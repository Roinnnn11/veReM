import time

from .base import BaseDecoder
from ..spans.math_span import get_math_candidate_spans, mask_span


class VeReMDecoder(BaseDecoder):
    """
    Verifier-Guided Span Remasking decoder.

    For each initially wrong example:
      - Propose candidate spans (sorted by heuristic math score)
      - For each span, remask + infill + verify
      - Accept the first revision that passes the verifier
      - Budget: max_spans_per_example * samples_per_span infill calls
    """

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
        samples_per_span=2,
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

    def decode(self, example: dict) -> dict:
        t0 = time.time()

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

        if not initial_correct:
            for revision_round in range(1, self.max_revision_rounds + 1):
                current_output = final_output
                candidates = get_math_candidate_spans(current_output)
                candidates = candidates[: self.max_spans_per_example]

                round_fixed = False
                for span in candidates:
                    for sample_idx in range(self.samples_per_span):
                        masked = mask_span(current_output, span)
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
                            "round": revision_round,
                            "span_id": span.span_id,
                            "span_text": span.text,
                            "sample_idx": sample_idx,
                            "masked_output": masked,
                            "revised_output": revised,
                            "verifier_pass": passed,
                            "latency": infill_result["latency"],
                        })

                        if passed:
                            final_output = revised
                            final_correct = True
                            round_fixed = True
                            break

                    if round_fixed:
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
            "method": "verem",
            "latency": latency,
            "num_model_forwards": num_forwards,
            "num_verifier_calls": num_verifier_calls,
            "num_remasked_spans": len(revision_trace),
            "revision_trace": revision_trace,
        }
