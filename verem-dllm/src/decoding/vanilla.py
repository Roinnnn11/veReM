import time

from .base import BaseDecoder


class VanillaDecoder(BaseDecoder):
    def __init__(self, model, verifier, max_new_tokens=256, steps=256, temperature=0.0, block_length=32):
        self.model = model
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.block_length = block_length

    def decode(self, example: dict) -> dict:
        t0 = time.time()
        result = self.model.generate(
            prompt=example["prompt"],
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            block_length=self.block_length,
        )
        latency = time.time() - t0

        output_text = result["text"]
        correct = self.verifier(output_text, example["gold"])

        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "gold": example["gold"],
            "initial_output": output_text,
            "final_output": output_text,
            "initial_correct": correct,
            "final_correct": correct,
            "method": "vanilla",
            "latency": latency,
            "num_model_forwards": self.steps,
            "num_verifier_calls": 1,
            "num_remasked_spans": 0,
            "revision_trace": [],
        }
