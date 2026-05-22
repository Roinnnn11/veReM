import time
from collections import Counter

from .base import BaseDecoder
from ..verifiers.gsm8k_verifier import extract_answer


class SelfConsistencyDecoder(BaseDecoder):
    """Generate N samples with temperature > 0, take majority-vote answer."""

    def __init__(
        self,
        model,
        verifier,
        max_new_tokens=256,
        steps=256,
        temperature=0.5,
        block_length=32,
        num_samples=5,
    ):
        self.model = model
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.block_length = block_length
        self.num_samples = num_samples

    def decode(self, example: dict) -> dict:
        t0 = time.time()

        outputs = []
        for _ in range(self.num_samples):
            result = self.model.generate(
                prompt=example["prompt"],
                max_new_tokens=self.max_new_tokens,
                steps=self.steps,
                temperature=self.temperature,
                block_length=self.block_length,
            )
            outputs.append(result["text"])

        # Majority vote on extracted answers
        answers = [extract_answer(o) for o in outputs]
        valid = [a for a in answers if a is not None]

        if valid:
            majority_answer = Counter(valid).most_common(1)[0][0]
            # Pick the output whose extracted answer matches majority
            final_output = next(
                o for o, a in zip(outputs, answers) if a == majority_answer
            )
        else:
            final_output = outputs[0]

        # Initial = first sample (greedy equivalent for comparison)
        initial_output = outputs[0]
        initial_correct = self.verifier(initial_output, example["gold"])
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
            "method": "self_consistency",
            "latency": latency,
            "num_model_forwards": self.steps * self.num_samples,
            "num_verifier_calls": self.num_samples,
            "num_remasked_spans": 0,
            "revision_trace": [
                {"sample_idx": i, "output": o, "answer": a}
                for i, (o, a) in enumerate(zip(outputs, answers))
            ],
        }
