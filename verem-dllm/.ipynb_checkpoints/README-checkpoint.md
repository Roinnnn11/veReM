# VeReM: Verifier-Guided Span Remasking for dLLM Reasoning

Training-free decoder that selectively revises reasoning spans in diffusion language models (dLLMs) using a task verifier as the revision signal.

## Quick Start

```bash
pip install -r requirements.txt

# Smoke test (20 samples, all 4 methods)
bash scripts/smoke_test.sh

# Full 200-sample run
bash scripts/run_gsm8k_all.sh 200 42
```

## Project Structure

```
verem-dllm/
  src/
    models/          # LLaDA wrapper + base interface
    spans/           # Span splitting and masking utilities
    decoding/        # Vanilla, Random, Heuristic, VeReM decoders
    verifiers/       # GSM8K answer extraction and correctness check
    datasets/        # GSM8K loader
    eval/            # run_gsm8k.py, aggregate_results.py
    analysis/        # case_study.py
    utils/           # io.py, seeds.py
  configs/           # YAML configs for each experiment
  scripts/           # Shell scripts to run experiments
  outputs/           # Generated results (gitignored)
```

## Methods

| Method | Description |
|---|---|
| `vanilla` | Standard dLLM decoding, no revision |
| `random_remask` | Randomly select a candidate span, remask + infill |
| `heuristic_remask` | Select spans by math relevance score (numbers, operators) |
| `verem` | Verifier-guided: try spans in order of math score, accept first that passes verifier |

All three remasking methods only attempt revision on initially wrong examples.

## Key Design Decisions

**Span proposal**: Split generation by newlines first, fall back to sentence boundaries. Filter to spans containing digits or operators, exclude the `####` final answer line.

**Infill**: Replace the selected span with a single `[MASK]` token, then run the masked diffusion decode loop over only the masked positions. Non-masked positions are frozen.

**Verifier (GSM8K)**: Extract the number after `####`, fall back to last number in text. Normalize commas, trailing dots, and integer floats (`3.0` → `3`).

**Revision trace**: Every remask attempt is logged with span text, masked output, revised output, and verifier pass/fail. This is the primary asset for case studies and analysis.

## Running Individual Methods

```bash
# Vanilla
python -m src.eval.run_gsm8k \
  --method vanilla --num_samples 200 --steps 64 \
  --output outputs/generations/gsm8k_vanilla_200.jsonl

# VeReM
python -m src.eval.run_gsm8k \
  --method verem --num_samples 200 --steps 64 \
  --infill_steps 32 --max_spans_per_example 3 --samples_per_span 2 \
  --output outputs/generations/gsm8k_verem_200.jsonl
```

## Aggregating Results

```bash
python -m src.eval.aggregate_results \
  --inputs outputs/generations/gsm8k_vanilla_200.jsonl \
            outputs/generations/gsm8k_verem_200.jsonl \
  --output outputs/metrics/gsm8k_main_table.csv
```

## Exporting Case Studies

```bash
python -m src.analysis.case_study \
  --input outputs/generations/gsm8k_verem_200.jsonl \
  --output_dir outputs/case_studies
```

## Output Format

Each JSONL line contains:

```json
{
  "id": "gsm8k_test_0",
  "prompt": "...",
  "gold": "...",
  "initial_output": "...",
  "final_output": "...",
  "initial_correct": false,
  "final_correct": true,
  "method": "verem",
  "latency": 12.4,
  "num_model_forwards": 128,
  "num_verifier_calls": 3,
  "num_remasked_spans": 2,
  "revision_trace": [
    {
      "round": 1,
      "span_id": 2,
      "span_text": "So 3 * 4 = 14",
      "masked_output": "...[MASK]...",
      "revised_output": "...",
      "verifier_pass": true,
      "latency": 4.1
    }
  ]
}
```

## Metrics

- `initial_accuracy` / `final_accuracy`: fraction correct before/after revision
- `fix_rate`: fraction of initially wrong examples that were fixed
- `avg_extra_forwards`: average additional model forward passes vs vanilla
- `avg_verifier_calls`: average verifier calls per example
- `avg_latency` / `p50_latency` / `p95_latency`

## Known Limitations

- Batch size is fixed at 1. Throughput optimization is deferred.
- The infill interface encodes masked text by splitting on `[MASK]` string — this means the span boundary is at the character level, not token level. Minor token boundary misalignment is expected.
- `temperature=0.0` uses greedy confidence-based unmasking. Set `temperature > 0` for stochastic infill.
