import os
import time
import torch
from transformers import AutoTokenizer, AutoModel

from .base import BaseDLLM

MASK_TOKEN = "<|mdm_mask|>"
MASK_ID = 126336


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(noise.clamp(min=1e-20)))
    return logits + temperature * gumbel


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute unmasking budget evenly across steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)
    base = mask_num // steps
    remainder = mask_num % steps
    nt = base.expand(-1, steps).clone()
    for b in range(mask_num.size(0)):
        nt[b, : int(remainder[b].item())] += 1
    return nt  # (B, steps)


class LLaDAWrapper(BaseDLLM):
    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "auto"):
        if not os.path.isabs(model_name):
            cwd_path = os.path.abspath(model_name)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            root_path = os.path.join(repo_root, model_name)
            if os.path.exists(cwd_path):
                model_name = cwd_path
            elif os.path.exists(root_path):
                model_name = root_path
        self.model_name = model_name

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(
                lambda self: {k: None for k in (self._tied_weights_keys or [])}
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(resolved_device)
        self.model.eval()

        check = self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        if check is None or check == self.tokenizer.unk_token_id:
            self.mask_token_id = MASK_ID
        else:
            self.mask_token_id = int(check)

    def _build_chat_input(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _block_decode(
        self,
        x: torch.Tensor,
        prompt_len: int,
        gen_length: int,
        steps: int,
        block_length: int,
        temperature: float,
        remasking: str,
    ) -> None:
        """
        Block-wise semi-autoregressive masked diffusion decoding (in-place).
        Matches the official LLaDA generate.py logic.
        """
        mask_id = self.mask_token_id
        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = block_start + block_length

            # Mask index within this block
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            nt = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step in range(steps_per_block):
                mask_index = (x == mask_id)
                logits = self.model(x).logits  # (1, seq_len, vocab)

                if temperature > 0.0:
                    logits_for_sample = _add_gumbel_noise(logits, temperature)
                    x0 = logits_for_sample.argmax(dim=-1)
                else:
                    x0 = logits.argmax(dim=-1)

                if remasking == "low_confidence":
                    p = torch.softmax(logits.to(torch.float32), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Only unmask within the current block
                x0_p[:, :block_start] = float("inf")
                x0_p[:, block_end:] = float("inf")

                # Positions that are still masked
                x0_p = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("inf")))

                num_to_unmask = int(nt[0, step].item())
                if num_to_unmask > 0:
                    _, transfer_idx = x0_p.topk(num_to_unmask, dim=-1, largest=False)
                    x[0].scatter_(0, transfer_idx[0], x0[0, transfer_idx[0]])

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        steps: int = 256,
        temperature: float = 0.0,
        block_length: int = 32,
        remasking: str = "low_confidence",
        **kwargs,
    ) -> dict:
        chat_input = self._build_chat_input(prompt)
        prompt_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids.to(self.model.device)
        prompt_len = prompt_ids.shape[1]

        # Round gen_length up to a multiple of block_length
        gen_length = max_new_tokens
        if gen_length % block_length != 0:
            gen_length = ((gen_length // block_length) + 1) * block_length
        num_blocks = gen_length // block_length
        # Round steps up to a multiple of num_blocks
        if steps % num_blocks != 0:
            steps = ((steps // num_blocks) + 1) * num_blocks

        x = torch.full(
            (1, prompt_len + gen_length),
            self.mask_token_id,
            dtype=torch.long,
            device=self.model.device,
        )
        x[:, :prompt_len] = prompt_ids

        t0 = time.time()
        with torch.no_grad():
            self._block_decode(
                x,
                prompt_len=prompt_len,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
            )
        latency = time.time() - t0

        gen_ids = x[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return {
            "text": text,
            "latency": latency,
            "num_steps": steps,
            "num_forwards": steps,
            "metadata": {"gen_length": gen_length, "block_length": block_length},
        }

    def _encode_masked_text(self, text_with_masks: str) -> torch.Tensor:
        """
        Tokenize text that contains MASK_TOKEN placeholders.
        Each placeholder is replaced by a single mask_id token.
        Returns shape (1, seq_len).
        """
        parts = text_with_masks.split(MASK_TOKEN)
        token_ids = []
        for i, part in enumerate(parts):
            if part:
                ids = self.tokenizer(part, add_special_tokens=False).input_ids
                token_ids.extend(ids)
            if i < len(parts) - 1:
                token_ids.append(self.mask_token_id)
        return torch.tensor([token_ids], dtype=torch.long)

    def infill(
        self,
        prompt: str,
        text_with_masks: str,
        steps: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        chat_input = self._build_chat_input(prompt)
        prompt_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids.to(self.model.device)
        prompt_len = prompt_ids.shape[1]

        gen_ids = self._encode_masked_text(text_with_masks).to(self.model.device)
        gen_length = gen_ids.shape[1]

        num_masks = int((gen_ids == self.mask_token_id).sum().item())
        if num_masks == 0:
            text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            return {
                "text": text,
                "latency": 0.0,
                "num_steps": 0,
                "num_forwards": 0,
                "metadata": {"no_masks": True},
            }

        x = torch.cat([prompt_ids, gen_ids], dim=1)

        t0 = time.time()
        with torch.no_grad():
            # Treat the whole generation as one block for infill
            self._block_decode(
                x,
                prompt_len=prompt_len,
                gen_length=gen_length,
                steps=steps,
                block_length=gen_length,
                temperature=temperature,
                remasking="low_confidence",
            )
        latency = time.time() - t0

        out_ids = x[0, prompt_len:]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        return {
            "text": text,
            "latency": latency,
            "num_steps": steps,
            "num_forwards": steps,
            "metadata": {"gen_length": gen_length, "num_masks": num_masks},
        }
