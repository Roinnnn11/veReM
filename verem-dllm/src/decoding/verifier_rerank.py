import time

from .base import BaseDecoder
from ..verifiers.gsm8k_verifier import extract_answer
from collections import Counter


class VerifierRerankDecoder(BaseDecoder):
    """Generate N candidates with low temperature, verifier picks the best one.

    Selection priority:
    1. First candidate that passes the verifier (oracle upper bound)
    2. Majority vote among extracted answers (no verifier signal)
    3. First candidate (fallback)
    """

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

        # Verifier reranking: pick first passing candidate
        final_output = None
        for c in candidates:
            if self.verifier(c, example["gold"]):
                final_output = c
                break

        # Fallback: majority vote
        if final_output is None:
            answers = [extract_answer(c) for c in candidates]
            valid = [a for a in answers if a is not None]
            if valid:
                majority = Counter(valid).most_common(1)[0][0]
                final_output = next(
                    c for c, a in zip(candidates, answers) if a == majority
                )
            else:
                final_output = candidates[0]

        final_correct = self.verifier(final_output, example["gold"])
        latency = time.time() - t0

        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "gold": example["gold"],
            "initial_output": initial_output,
            "final_output": final_output,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "method": "verifier_rerank",
            "latency": latency,
            "num_model_forwards": self.steps * self.num_candidates,
            "num_verifier_calls": self.num_candidates,
            "num_remasked_spans": 0,
            "revision_trace": [
                {
                    "candidate_idx": i,
                    "output": c,
                    "answer": extract_answer(c),
                    "verifier_pass": self.verifier(c, example["gold"]),
                }
                for i, c in enumerate(candidates)
            ],
        }
