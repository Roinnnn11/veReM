"""
Main GSM8K evaluation runner.

Usage:
    python -m src.eval.run_gsm8k \
        --model_name GSAI-ML/LLaDA-8B-Instruct \
        --method verem \
        --num_samples 200 \
        --steps 64 \
        --infill_steps 32 \
        --max_revision_rounds 1 \
        --max_spans_per_example 3 \
        --samples_per_span 2 \
        --seed 42 \
        --output outputs/generations/gsm8k_verem_200.jsonl
"""

import argparse
import json
import os
import time

from tqdm import tqdm

from ..models.llada_wrapper import LLaDAWrapper
from ..datasets.gsm8k_loader import load_gsm8k
from ..verifiers.gsm8k_verifier import is_correct
from ..decoding.vanilla import VanillaDecoder
from ..decoding.random_remask import RandomRemaskDecoder
from ..decoding.heuristic_remask import HeuristicRemaskDecoder
from ..decoding.verifier_remask import VeReMDecoder
from ..utils.io import append_jsonl, write_jsonl
from ..utils.seeds import set_seed


def build_decoder(method: str, model, verifier, args):
    common = dict(
        model=model,
        verifier=verifier,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
    )
    remask_common = dict(
        **common,
        infill_steps=args.infill_steps,
        max_revision_rounds=args.max_revision_rounds,
        max_spans_per_example=args.max_spans_per_example,
        samples_per_span=args.samples_per_span,
    )
    if method == "vanilla":
        return VanillaDecoder(**common)
    elif method == "random_remask":
        return RandomRemaskDecoder(**remask_common, seed=args.seed)
    elif method == "heuristic_remask":
        return HeuristicRemaskDecoder(**remask_common)
    elif method == "verem":
        return VeReMDecoder(**remask_common)
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_metrics(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {}

    initial_correct = sum(r["initial_correct"] for r in records)
    final_correct = sum(r["final_correct"] for r in records)

    initially_wrong = [r for r in records if not r["initial_correct"]]
    fixed = [r for r in initially_wrong if r["final_correct"]]
    fix_rate = len(fixed) / len(initially_wrong) if initially_wrong else 0.0

    latencies = [r["latency"] for r in records]
    latencies_sorted = sorted(latencies)
    avg_latency = sum(latencies) / n
    p50 = latencies_sorted[int(n * 0.5)]
    p95 = latencies_sorted[int(n * 0.95)]

    avg_forwards = sum(r["num_model_forwards"] for r in records) / n
    avg_verifier_calls = sum(r["num_verifier_calls"] for r in records) / n

    return {
        "n": n,
        "initial_accuracy": initial_correct / n,
        "final_accuracy": final_correct / n,
        "fix_rate": fix_rate,
        "initially_wrong": len(initially_wrong),
        "fixed": len(fixed),
        "avg_extra_forwards": avg_forwards - records[0].get("steps_baseline", args_steps_baseline),
        "avg_model_forwards": avg_forwards,
        "avg_verifier_calls": avg_verifier_calls,
        "avg_latency": avg_latency,
        "p50_latency": p50,
        "p95_latency": p95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--method", default="verem",
                        choices=["vanilla", "random_remask", "heuristic_remask", "verem"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--infill_steps", type=int, default=32)
    parser.add_argument("--max_revision_rounds", type=int, default=1)
    parser.add_argument("--max_spans_per_example", type=int, default=3)
    parser.add_argument("--samples_per_span", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading model: {args.model_name}")
    model = LLaDAWrapper(model_name=args.model_name, device=args.device)

    print(f"Loading GSM8K ({args.split}, {args.num_samples} samples)")
    examples = load_gsm8k(split=args.split, num_samples=args.num_samples, seed=args.seed)

    decoder = build_decoder(args.method, model, is_correct, args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records = []
    t_start = time.time()
    for ex in tqdm(examples, desc=f"[{args.method}]"):
        result = decoder.decode(ex)
        result["steps_baseline"] = args.steps
        records.append(result)
        append_jsonl(args.output, result)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s. Output: {args.output}")

    # Compute and save metrics
    metrics = {
        "method": args.method,
        "model": args.model_name,
        "num_samples": len(records),
        "steps": args.steps,
        "infill_steps": args.infill_steps,
        "max_spans_per_example": args.max_spans_per_example,
        "samples_per_span": args.samples_per_span,
        "seed": args.seed,
        "total_time": total_time,
    }

    n = len(records)
    initial_correct = sum(r["initial_correct"] for r in records)
    final_correct = sum(r["final_correct"] for r in records)
    initially_wrong = [r for r in records if not r["initial_correct"]]
    fixed = [r for r in initially_wrong if r["final_correct"]]
    latencies = sorted(r["latency"] for r in records)

    metrics.update({
        "initial_accuracy": initial_correct / n,
        "final_accuracy": final_correct / n,
        "fix_rate": len(fixed) / len(initially_wrong) if initially_wrong else 0.0,
        "initially_wrong": len(initially_wrong),
        "fixed": len(fixed),
        "avg_model_forwards": sum(r["num_model_forwards"] for r in records) / n,
        "avg_verifier_calls": sum(r["num_verifier_calls"] for r in records) / n,
        "avg_latency": sum(latencies) / n,
        "p50_latency": latencies[int(n * 0.5)],
        "p95_latency": latencies[int(n * 0.95)],
    })

    metrics_path = args.output.replace(".jsonl", "_metrics.json").replace(
        "generations", "metrics"
    )
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Initial Accuracy : {metrics['initial_accuracy']:.3f}")
    print(f"Final Accuracy   : {metrics['final_accuracy']:.3f}")
    print(f"Fix Rate         : {metrics['fix_rate']:.3f}")
    print(f"Avg Forwards     : {metrics['avg_model_forwards']:.1f}")
    print(f"Avg Latency      : {metrics['avg_latency']:.2f}s")
    print(f"Metrics saved to : {metrics_path}")


# module-level fallback for aggregate_metrics
args_steps_baseline = 64

if __name__ == "__main__":
    main()
