"""
Self-evaluation failure mode analysis.

For each candidate, we have:
  - is_correct(c, gold)  — ground truth label for the candidate
  - self_eval_score      — logprob(Yes) - logprob(No), the model's "is correct?" probe

Question: does self_eval_score discriminate correct vs incorrect candidates?

Outputs:
  - mean / median score for correct vs incorrect candidates
  - AUC of self_eval_score as a binary classifier of correctness
  - Spearman correlation
  - Per-example: was the highest-scoring candidate also the correct one?
"""

import argparse
import json
import math
import sys
from collections import Counter

from src.verifiers.gsm8k_verifier import is_correct, extract_answer


def auc_score(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC: P(score(positive) > score(negative))."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    n_pairs = 0
    n_correct = 0
    for p in pos:
        for n in neg:
            n_pairs += 1
            if p > n:
                n_correct += 1
            elif p == n:
                n_correct += 0.5
    return n_correct / n_pairs


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (no scipy)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _ranks(xs: list[float]) -> list[float]:
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[sorted_idx[j + 1]] == xs[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def analyse(records: list[dict]) -> dict:
    all_scores = []
    all_labels = []
    per_example = []

    for r in records:
        gold = r["gold"]
        trace = r.get("revision_trace", [])
        if not trace or "self_eval_score" not in trace[0]:
            continue

        scores = [t["self_eval_score"] for t in trace]
        labels = [int(is_correct(t["output"], gold)) for t in trace]

        all_scores.extend(scores)
        all_labels.extend(labels)

        if any(labels) and not all(labels):
            top_idx = max(range(len(scores)), key=lambda i: scores[i])
            top_correct = labels[top_idx] == 1
            # Was there at least one correct candidate the model could have picked?
            per_example.append({
                "n_correct": sum(labels),
                "n_total": len(labels),
                "top_picked_correct": top_correct,
                "score_gap": scores[top_idx] - max(s for s, l in zip(scores, labels) if l != labels[top_idx]) if any(l != labels[top_idx] for l in labels) else 0.0,
            })

    n = len(all_scores)
    if n == 0:
        return {"error": "no self_eval_score in records"}

    correct_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    wrong_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]

    mean_correct = sum(correct_scores) / len(correct_scores) if correct_scores else float("nan")
    mean_wrong = sum(wrong_scores) / len(wrong_scores) if wrong_scores else float("nan")

    auc = auc_score(all_scores, all_labels)
    rho = spearman_corr(all_scores, all_labels)

    # Among ambiguous examples (some correct, some wrong), how often does
    # the top-scored candidate happen to be correct?
    ambiguous = [e for e in per_example if 0 < e["n_correct"] < e["n_total"]]
    if ambiguous:
        rate_top_correct = sum(e["top_picked_correct"] for e in ambiguous) / len(ambiguous)
    else:
        rate_top_correct = float("nan")

    return {
        "n_candidates_total": n,
        "n_correct_candidates": len(correct_scores),
        "n_wrong_candidates": len(wrong_scores),
        "mean_score_correct": mean_correct,
        "mean_score_wrong": mean_wrong,
        "score_gap": mean_correct - mean_wrong,
        "roc_auc": auc,
        "spearman_rho": rho,
        "n_ambiguous_examples": len(ambiguous),
        "p_top_picked_correct_given_ambiguous": rate_top_correct,
        "if_random_baseline": (
            sum(e["n_correct"] / e["n_total"] for e in ambiguous) / len(ambiguous)
            if ambiguous else float("nan")
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    print(f"{'file':<55}  {'AUC':>6}  {'Δscore':>7}  {'ρ':>6}  {'P(top|amb)':>10}  {'rand':>5}")
    print("-" * 100)
    for fp in args.inputs:
        records = [json.loads(l) for l in open(fp) if l.strip()]
        m = analyse(records)
        if "error" in m:
            print(f"{fp.split('/')[-1]}: {m['error']}")
            continue
        print(
            f"{fp.split('/')[-1]:<55}  "
            f"{m['roc_auc']:>6.3f}  "
            f"{m['score_gap']:>7.3f}  "
            f"{m['spearman_rho']:>6.3f}  "
            f"{m['p_top_picked_correct_given_ambiguous']:>10.3f}  "
            f"{m['if_random_baseline']:>5.3f}"
        )


if __name__ == "__main__":
    main()
