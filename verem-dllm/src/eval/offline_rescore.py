"""
Offline rescoring: take an existing run jsonl (with revision_trace containing
N candidates per example) and recompute final accuracy under different
selection strategies.

Strategies:
  - oracle      : use is_correct(c, gold) — the (dishonest) upper bound
  - majority    : majority vote on extract_answer (Self-Consistency)
  - format      : pick highest format_score, tie-break by candidate index
  - arithmetic  : pick highest arithmetic_consistency_score
  - honest      : weighted combo (0.4 format + 0.6 arithmetic)
  - first       : always take candidate[0] (greedy-equivalent baseline)
  - pass_at_n   : at least one candidate is correct (oracle Pass@N)

Usage:
    python -m src.eval.offline_rescore \
        --input outputs/generations/gsm8k_vrerank_t01_n5_200_seed42.jsonl \
        --strategies oracle majority format arithmetic honest first pass_at_n
"""

import argparse
import json
import os
from collections import Counter

from ..verifiers.gsm8k_verifier import extract_answer, is_correct
from ..verifiers.honest_verifiers import (
    format_score,
    arithmetic_consistency_score,
    honest_score,
)


def get_candidate_texts(record: dict) -> list[str]:
    """Pull candidate output strings from revision_trace, regardless of method."""
    trace = record.get("revision_trace", [])
    if not trace:
        # No trace — only the final_output is available.
        return [record.get("final_output", "")]
    return [t.get("output", "") for t in trace]


def select_oracle(cands: list[str], gold: str) -> str:
    """First candidate that matches gold; else first candidate. (Dishonest.)"""
    for c in cands:
        if is_correct(c, gold):
            return c
    return cands[0]


def select_majority(cands: list[str], _gold: str) -> str:
    """Self-Consistency style: pick candidate whose answer matches majority."""
    answers = [extract_answer(c) for c in cands]
    valid = [a for a in answers if a is not None]
    if not valid:
        return cands[0]
    majority = Counter(valid).most_common(1)[0][0]
    for c, a in zip(cands, answers):
        if a == majority:
            return c
    return cands[0]


def _pick_max(cands: list[str], scorer) -> str:
    """Pick the candidate with the highest scorer(c). Ties: lowest index."""
    best = cands[0]
    best_s = scorer(cands[0])
    for c in cands[1:]:
        s = scorer(c)
        if s > best_s:
            best, best_s = c, s
    return best


def select_format(cands: list[str], _gold: str) -> str:
    return _pick_max(cands, format_score)


def select_arithmetic(cands: list[str], _gold: str) -> str:
    return _pick_max(cands, arithmetic_consistency_score)


def select_honest(cands: list[str], _gold: str) -> str:
    return _pick_max(cands, honest_score)


def select_first(cands: list[str], _gold: str) -> str:
    return cands[0]


def _self_eval_scores(record: dict) -> list[float] | None:
    """Pull self_eval_score from trace, if present."""
    trace = record.get("revision_trace", [])
    if not trace or "self_eval_score" not in trace[0]:
        return None
    return [t.get("self_eval_score", 0.0) for t in trace]


def select_self_eval(record: dict, _gold: str) -> str:
    """Pick candidate with highest self_eval_score. Falls back to first."""
    cands = get_candidate_texts(record)
    scores = _self_eval_scores(record)
    if scores is None:
        return cands[0]
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    return cands[best_i]


def select_weighted_sc(record: dict, _gold: str) -> str:
    """Weighted self-consistency: each candidate votes for its extracted
    answer with weight sigmoid(self_eval_score). Pick candidate matching
    the answer with highest accumulated weight. Ties broken by candidate
    index. Falls back to plain majority if scores unavailable."""
    import math
    cands = get_candidate_texts(record)
    answers = [extract_answer(c) for c in cands]
    scores = _self_eval_scores(record)
    if scores is None:
        return select_majority(cands, _gold)
    weights = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    bucket: dict[str, float] = {}
    for a, w in zip(answers, weights):
        if a is None:
            continue
        bucket[a] = bucket.get(a, 0.0) + w
    if not bucket:
        return cands[0]
    top = max(bucket.items(), key=lambda kv: kv[1])[0]
    for c, a in zip(cands, answers):
        if a == top:
            return c
    return cands[0]


SELECTORS = {
    "oracle": select_oracle,
    "majority": select_majority,
    "format": select_format,
    "arithmetic": select_arithmetic,
    "honest": select_honest,
    "first": select_first,
}

# Selectors that need access to the full record (not just candidate texts),
# e.g. to read self_eval_score from the trace.
RECORD_SELECTORS = {
    "self_eval": select_self_eval,
    "weighted_sc": select_weighted_sc,
}


def evaluate(records: list[dict], strategy: str) -> dict:
    n = len(records)
    correct = 0
    initial_correct = 0
    pass_at_n = 0  # oracle "is at least one candidate correct"
    selector = SELECTORS.get(strategy)
    record_selector = RECORD_SELECTORS.get(strategy)

    for r in records:
        cands = get_candidate_texts(r)
        gold = r["gold"]
        # Pass@N: any candidate correct?
        any_correct = any(is_correct(c, gold) for c in cands)
        if any_correct:
            pass_at_n += 1
        # Initial = first candidate
        if is_correct(cands[0], gold):
            initial_correct += 1

        if strategy == "pass_at_n":
            if any_correct:
                correct += 1
        elif record_selector is not None:
            chosen = record_selector(r, gold)
            if is_correct(chosen, gold):
                correct += 1
        else:
            chosen = selector(cands, gold)
            if is_correct(chosen, gold):
                correct += 1

    return {
        "strategy": strategy,
        "n_examples": n,
        "n_candidates": len(get_candidate_texts(records[0])),
        "accuracy": correct / n,
        "initial_accuracy": initial_correct / n,
        "pass_at_n": pass_at_n / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["first", "oracle", "majority", "format", "arithmetic", "honest", "self_eval", "weighted_sc", "pass_at_n"],
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"  candidates per record: {len(get_candidate_texts(records[0]))}")
    print()

    rows = []
    for s in args.strategies:
        m = evaluate(records, s)
        rows.append(m)
        print(
            f"  {s:12s}  acc={m['accuracy']:.3f}  initial={m['initial_accuracy']:.3f}  "
            f"pass@N={m['pass_at_n']:.3f}"
        )

    out = args.output or args.input.replace(".jsonl", "_rescored.json").replace(
        "generations", "metrics"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"input": args.input, "rows": rows}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
