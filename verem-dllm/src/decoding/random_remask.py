import random
import time

from .base import BaseDecoder
from ..spans.sentence_splitter import split_sentences
from ..spans.math_span import is_candidate_span, mask_span


class RandomRemaskDecoder(BaseDecoder):
    def __init__(
        self,
        model,
        verifier,
        max_new_tokens=512,
        steps=64,
        infill_steps=32,
        temperature=0.0,
        max_revision_rounds=1,
        max_spans_per_example=3,
        samples_per_span=1,
        seed=42,
    ):
        self.model = model
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.infill_steps = infill_steps
        self.temperature = temperature
        self.max_revision_rounds = max_revision_rounds
        self.max_spans_per_example = max_spans_per_example
        self.samples_per_span = samples_per_span
        self.rng = random.Random(seed)

    def decode(self, example: dict) -> dict:
        t0 = time.time()

        # Initial generation
        init_result = self.model.generate(
            prompt=example["prompt"],
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
        )
        initial_output = init_result["text"]
        initial_correct = self.verifier(initial_output, example["gold"])
        num_forwards = self.steps
        num_verifier_calls = 1

        final_output = initial_output
        final_correct = initial_correct
        revision_trace = []

        # Only attempt revision on initially wrong examples
        if not initial_correct:
            all_spans = split_sentences(initial_output)
            candidates = [s for s in all_spans if is_candidate_span(s.text)]

            if candidates:
                selected = self.rng.sample(
                    candidates, min(self.max_spans_per_example, len(candidates))
                )

                for span in selected:
                    for _ in range(self.samples_per_span):
                        masked = mask_span(initial_output, span)
                        infill_result = self.model.infill(
                            prompt=example["prompt"],
                            text_with_masks=masked,
                            steps=self.infill_steps,
                            temperature=self.temperature,
                        )
                        revised = infill_result["text"]
                        num_forwards += self.infill_steps
                        passed = self.verifier(revised, example["gold"])
                        num_verifier_calls += 1

                        revision_trace.append({
                            "round": 1,
                            "span_id": span.span_id,
                            "span_text": span.text,
                            "masked_output": masked,
                            "revised_output": revised,
                            "verifier_pass": passed,
                            "latency": infill_result["latency"],
                        })

                        if passed:
                            final_output = revised
                            final_correct = True
                            break

                    if final_correct:
                        break

        latency = time.time() - t0
        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "gold": example["gold"],
            "initial_output": initial_output,
            "final_output": final_output,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "method": "random_remask",
            "latency": latency,
            "num_model_forwards": num_forwards,
            "num_verifier_calls": num_verifier_calls,
            "num_remasked_spans": len(revision_trace),
            "revision_trace": revision_trace,
        }
