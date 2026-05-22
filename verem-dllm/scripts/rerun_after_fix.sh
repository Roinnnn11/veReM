#!/bin/bash
# Quick re-run after the mask_id / block-decode fix.
# Defaults to smoke (N=20) then full N=200 run.
# Override with: MODEL=/path/to/model N=50 bash scripts/rerun_after_fix.sh

set -e

MODEL="${MODEL:-/root/autodl-fs/LLaDA-8B-Instruct}"
N="${N:-200}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."

echo "=== Smoke test (20 samples, vanilla only) ==="
python -m src.eval.run_gsm8k \
  --model_name "$MODEL" \
  --method vanilla \
  --num_samples 20 \
  --steps 256 \
  --max_new_tokens 256 \
  --block_length 32 \
  --seed "$SEED" \
  --output outputs/generations/smoke_vanilla_20_seed${SEED}.jsonl

echo ""
echo "=== Smoke result ==="
python -c "
import json, sys
records = [json.loads(l) for l in open('outputs/generations/smoke_vanilla_20_seed${SEED}.jsonl')]
acc = sum(r['final_correct'] for r in records) / len(records)
print(f'Smoke vanilla acc: {acc:.3f}  ({sum(r[\"final_correct\"] for r in records)}/{len(records)})')
print('Sample output:', records[0]['final_output'][:200])
"

echo ""
echo "=== Full run: N=${N}, seed=${SEED} ==="
bash scripts/run_gsm8k_all.sh "$N" "$SEED"
