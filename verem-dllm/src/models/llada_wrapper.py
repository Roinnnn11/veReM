import time
import torch
from transformers import AutoTokenizer, AutoModel

from .base import BaseDLLM

MASK_TOKEN = "[MASK]"


class LLaDAWrapper(BaseDLLM):
    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)

    def _build_chat_input(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        steps: int = 64,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        chat_input = self._build_chat_input(prompt)
        input_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        prompt_len = input_ids.shape[1]

        # Build masked output sequence
        mask_ids = torch.full(
            (1, max_new_tokens), self.mask_token_id, dtype=torch.long, device=self.model.device
        )
        x = torch.cat([input_ids, mask_ids], dim=1)

        t0 = time.time()
        with torch.no_grad():
            output = self._mdlm_decode(x, prompt_len, steps, temperature)
        latency = time.time() - t0

        gen_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return {
            "text": text,
            "latency": latency,
            "num_steps": steps,
            "num_forwards": steps,
            "metadata": {},
        }

    def infill(
        self,
        prompt: str,
        text_with_masks: str,
        steps: int = 32,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        text_with_masks: the generation text where some spans are replaced with MASK_TOKEN strings.
        We re-encode the full sequence (prompt + masked generation) and run masked diffusion.
        """
        chat_input = self._build_chat_input(prompt)
        prompt_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        # Encode the masked generation text
        gen_ids = self._encode_masked_text(text_with_masks)
        x = torch.cat([prompt_ids.to(self.model.device), gen_ids.to(self.model.device)], dim=1)

        t0 = time.time()
        with torch.no_grad():
            output = self._mdlm_decode(x, prompt_len, steps, temperature)
        latency = time.time() - t0

        out_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        return {
            "text": text,
            "latency": latency,
            "num_steps": steps,
            "num_forwards": steps,
            "metadata": {},
        }

    def _encode_masked_text(self, text_with_masks: str) -> torch.Tensor:
        """
        Tokenize text that may contain MASK_TOKEN placeholders.
        Each MASK_TOKEN becomes a single mask_token_id.
        """
        parts = text_with_masks.split(MASK_TOKEN)
        ids = []
        for i, part in enumerate(parts):
            if part:
                part_ids = self.tokenizer(part, add_special_tokens=False).input_ids
                ids.extend(part_ids)
            if i < len(parts) - 1:
                ids.append(self.mask_token_id)
        return torch.tensor([ids], dtype=torch.long)

    def _mdlm_decode(
        self,
        x: torch.Tensor,
        prompt_len: int,
        steps: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Masked diffusion decoding loop.
        At each step, unmask the tokens with highest confidence.
        Only tokens in the generation region (after prompt_len) are unmasked.
        """
        mask_id = self.mask_token_id
        seq_len = x.shape[1]
        gen_len = seq_len - prompt_len

        for step in range(steps):
            logits = self.model(x).logits  # (1, seq_len, vocab)
            gen_logits = logits[0, prompt_len:]  # (gen_len, vocab)

            # Identify still-masked positions
            masked_positions = (x[0, prompt_len:] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                break

            num_to_unmask = max(1, len(masked_positions) // (steps - step))

            if temperature == 0.0:
                probs = torch.softmax(gen_logits, dim=-1)
                confidence, predicted = probs.max(dim=-1)
            else:
                probs = torch.softmax(gen_logits / temperature, dim=-1)
                predicted = torch.multinomial(probs, num_samples=1).squeeze(-1)
                confidence = probs[torch.arange(gen_len), predicted]

            # Only consider masked positions
            masked_confidence = confidence[masked_positions]
            _, top_idx = masked_confidence.topk(min(num_to_unmask, len(masked_positions)))
            positions_to_unmask = masked_positions[top_idx]

            x = x.clone()
            x[0, prompt_len + positions_to_unmask] = predicted[positions_to_unmask]

        return x
