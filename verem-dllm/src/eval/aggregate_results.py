"""
Aggregate results from multiple JSONL files into a comparison table.

Usage:
    python -m src.eval.aggregate_results \
        --inputs outputs/generations/gsm8k_vanilla_200.jsonl \
                  outputs/generations/gsm8k_random_200.jsonl \
                  outputs/generations/gsm8k_heuristic_200.jsonl \
                  outputs/generations/gsm8k_verem_200.jsonl \
        --output outputs/metrics/gsm8k_main_table_200.csv
"""

import argparse
import csv
import json
import os

from ..utils.io import read_jsonl


def compute_metrics(records: list, steps_baseline: int = 64) -> dict:
    n = len(records)
    if n == 0:
        return {}

    initial_correct = sum(r["initial_correct"] for r in records)
    final_correct = sum(r["final_correct"] for r in records)
    initially_wrong = [r for r in records if not r["initial_correct"]]
    fixed = [r for r in initially_wrong if r["final_correct"]]
    latencies = sorted(r["latency"] for r in records)
    avg_forwards = sum(r["num_model_forwards"] for r in records) / n

    return {
        "method": records[0].get("method", "unknown"),
        "n": n,
        "initial_acc": f"{initial_correct / n:.4f}",
        "final_acc": f"{final_correct / n:.4f}",
        "fix_rate": f"{len(fixed) / len(initially_wrong):.4f}" if initially_wrong else "N/A",
        "avg_extra_forwards": f"{avg_forwards - steps_baseline:.2f}",
        "avg_verifier_calls": f"{sum(r['num_verifier_calls'] for r in records) / n:.2f}",
        "avg_latency": f"{sum(latencies) / n:.2f}",
        "p50_latency": f"{latencies[int(n * 0.5)]:.2f}",
        "p95_latency": f"{latencies[int(n * 0.95)]:.2f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps_baseline", type=int, default=64)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        records = read_jsonl(path)
        metrics = compute_metrics(records, args.steps_baseline)
        rows.append(metrics)
        print(f"{path}: {metrics}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nTable saved to: {args.output}")


if __name__ == "__main__":
    main()
