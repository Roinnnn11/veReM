"""
Export case studies from JSONL results.

Usage:
    python -m src.analysis.case_study \
        --input outputs/generations/gsm8k_verem_200.jsonl \
        --output_dir outputs/case_studies \
        --n_fixed 10 \
        --n_failed 5
"""

import argparse
import os

from ..utils.io import read_jsonl


def format_case(record: dict, idx: int) -> str:
    lines = [
        f"## Case {idx + 1} — ID: {record['id']}",
        "",
        f"**Method:** {record['method']}",
        f"**Initial Correct:** {record['initial_correct']}  |  **Final Correct:** {record['final_correct']}",
        f"**Latency:** {record['latency']:.2f}s  |  **Forwards:** {record['num_model_forwards']}  |  **Verifier Calls:** {record['num_verifier_calls']}",
        "",
        "### Prompt",
        f"```\n{record['prompt'][:500]}\n```",
        "",
        "### Gold Answer",
        f"```\n{record['gold']}\n```",
        "",
        "### Initial Output",
        f"```\n{record['initial_output']}\n```",
    ]

    if record["final_output"] != record["initial_output"]:
        lines += [
            "",
            "### Final Output (after revision)",
            f"```\n{record['final_output']}\n```",
        ]

    if record.get("revision_trace"):
        lines += ["", "### Revision Trace"]
        for step in record["revision_trace"]:
            lines += [
                f"- **Round {step['round']} | Span {step['span_id']}**: `{step['span_text'][:80]}`",
                f"  - Verifier pass: {step['verifier_pass']}",
            ]

    lines.append("\n---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs/case_studies")
    parser.add_argument("--n_fixed", type=int, default=10)
    parser.add_argument("--n_failed", type=int, default=5)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    fixed = [r for r in records if not r["initial_correct"] and r["final_correct"]]
    failed = [r for r in records if not r["initial_correct"] and not r["final_correct"]]

    fixed_path = os.path.join(args.output_dir, "fixed_cases.md")
    with open(fixed_path, "w", encoding="utf-8") as f:
        f.write(f"# Fixed Cases ({len(fixed)} total, showing {min(args.n_fixed, len(fixed))})\n\n")
        for i, r in enumerate(fixed[: args.n_fixed]):
            f.write(format_case(r, i))
    print(f"Fixed cases: {fixed_path}")

    failed_path = os.path.join(args.output_dir, "failed_cases.md")
    with open(failed_path, "w", encoding="utf-8") as f:
        f.write(f"# Failed Cases ({len(failed)} total, showing {min(args.n_failed, len(failed))})\n\n")
        for i, r in enumerate(failed[: args.n_failed]):
            f.write(format_case(r, i))
    print(f"Failed cases: {failed_path}")


if __name__ == "__main__":
    main()
