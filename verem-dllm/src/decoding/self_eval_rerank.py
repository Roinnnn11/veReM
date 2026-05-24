import time
import torch

from .base import BaseDecoder
from ..verifiers.gsm8k_verifier import extract_answer


class SelfEvalRerankDecoder(BaseDecoder):
    """Honest verifier: ask the model to judge its own candidates.

    For each candidate c, build a probe prompt:

        Problem: {question}
        Solution: {c}
        Is the solution above correct? Answer with a single word.

    Run a single forward pass with a mask at the answer position, take the
    logits at that position, and use logprob('Yes') - logprob('No') as the
    score. Pick the candidate with the highest score; ties broken by
    candidate index.

    This is honest because the verifier never sees the gold answer.
    """

    YES_TOKENS = ("Yes", " Yes", "yes", " yes", "YES")
    NO_TOKENS = ("No", " No", "no", " no", "NO")

    def __init__(
        self,
        model,
        verifier,
        max_new_tokens=256,
        steps=256,
        temperature=0.1,
        block_length=32,
        num_candidates=5,
    ):
        self.model = model
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.block_length = block_length
        self.num_candidates = num_candidates

        tok = model.tokenizer
        self._yes_ids = self._collect_token_ids(tok, self.YES_TOKENS)
        self._no_ids = self._collect_token_ids(tok, self.NO_TOKENS)

    @staticmethod
    def _collect_token_ids(tok, words):
        ids = set()
        for w in words:
            try:
                tids = tok(w, add_special_tokens=False).input_ids
                if len(tids) == 1:
                    ids.add(tids[0])
                ids.add(tids[0])  # leading subword anyway
            except Exception:
                pass
        return list(ids)

    def _score_candidate(self, prompt: str, candidate_text: str) -> float:
        """Single forward pass; return logprob(Yes) - logprob(No) at mask pos."""
        tok = self.model.tokenizer
        mask_id = self.model.mask_token_id
        device = self.model.model.device

        probe_user = (
            f"Problem:\n{prompt}\n\n"
            f"Proposed solution:\n{candidate_text}\n\n"
            f"Is the proposed solution correct? Answer with a single word: Yes or No."
        )
        chat = self.model._build_chat_input(probe_user)
        prompt_ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Append a single mask token where the answer goes.
        mask = torch.full((1, 1), mask_id, dtype=torch.long, device=device)
        x = torch.cat([prompt_ids, mask], dim=1)

        with torch.no_grad():
            logits = self.model.model(x).logits  # (1, L, V)

        last_logits = logits[0, -1].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        yes_lp = torch.logsumexp(log_probs[self._yes_ids], dim=0).item() if self._yes_ids else float("-inf")
        no_lp = torch.logsumexp(log_probs[self._no_ids], dim=0).item() if self._no_ids else float("-inf")

        return yes_lp - no_lp

    def decode(self, example: dict) -> dict:
        t0 = time.time()

        candidates = []
        for _ in range(self.num_candidates):
            result = self.model.generate(
                prompt=example["prompt"],
                max_new_tokens=self.max_new_tokens,
                steps=self.steps,
                temperature=self.temperature,
                block_length=self.block_length,
            )
            candidates.append(result["text"])

        initial_output = candidates[0]
        initial_correct = self.verifier(initial_output, example["gold"])

        scores = [self._score_candidate(example["prompt"], c) for c in candidates]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        final_output = candidates[best_idx]
        final_correct = self.verifier(final_output, example["gold"])

        latency = time.time() - t0
        # +1 forward pass per candidate for the self-eval probe
        n_forward = self.steps * self.num_candidates + self.num_candidates

        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "gold": example["gold"],
            "initial_output": initial_output,
            "final_output": final_output,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "method": "self_eval_rerank",
            "latency": latency,
            "num_model_forwards": n_forward,
            "num_verifier_calls": self.num_candidates,
            "num_remasked_spans": 0,
            "revision_trace": [
                {
                    "candidate_idx": i,
                    "output": c,
                    "answer": extract_answer(c),
                    "self_eval_score": s,
                    "selected": (i == best_idx),
                }
                for i, (c, s) in enumerate(zip(candidates, scores))
            ],
        }
