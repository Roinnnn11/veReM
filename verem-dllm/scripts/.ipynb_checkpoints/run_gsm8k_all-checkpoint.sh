#!/bin/bash
# Run all four GSM8K methods and aggregate results
# Usage: bash scripts/run_gsm8k_all.sh [num_samples] [seed]

NUM=${1:-200}
SEED=${2:-42}
MODEL="/mnt/models/LLaDA-8B-Instruct"
STEPS=64
INFILL_STEPS=32
MAX_TOKENS=512
ROUNDS=1
SPANS=3

echo "=== VeReM GSM8K Experiment: N=${NUM}, seed=${SEED} ==="

python -m src.eval.run_gsm8k \
  --model_name $MODEL \
  --method vanilla \
  --num_samples $NUM \
  --steps $STEPS \
  --max_new_tokens $MAX_TOKENS \
  --seed $SEED \
  --output outputs/generations/gsm8k_vanilla_${NUM}_s${STEPS}_seed${SEED}.jsonl

python -m src.eval.run_gsm8k \
  --model_name $MODEL \
  --method random_remask \
  --num_samples $NUM \
  --steps $STEPS \
  --infill_steps $INFILL_STEPS \
  --max_revision_rounds $ROUNDS \
  --max_spans_per_example $SPANS \
  --samples_per_span 1 \
  --seed $SEED \
  --output outputs/generations/gsm8k_random_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n1_seed${SEED}.jsonl

python -m src.eval.run_gsm8k \
  --model_name $MODEL \
  --method heuristic_remask \
  --num_samples $NUM \
  --steps $STEPS \
  --infill_steps $INFILL_STEPS \
  --max_revision_rounds $ROUNDS \
  --max_spans_per_example $SPANS \
  --samples_per_span 1 \
  --seed $SEED \
  --output outputs/generations/gsm8k_heuristic_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n1_seed${SEED}.jsonl

python -m src.eval.run_gsm8k \
  --model_name $MODEL \
  --method verem \
  --num_samples $NUM \
  --steps $STEPS \
  --infill_steps $INFILL_STEPS \
  --max_revision_rounds $ROUNDS \
  --max_spans_per_example $SPANS \
  --samples_per_span 2 \
  --seed $SEED \
  --output outputs/generations/gsm8k_verem_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n2_seed${SEED}.jsonl

echo "=== Aggregating results ==="

python -m src.eval.aggregate_results \
  --inputs \
    outputs/generations/gsm8k_vanilla_${NUM}_s${STEPS}_seed${SEED}.jsonl \
    outputs/generations/gsm8k_random_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n1_seed${SEED}.jsonl \
    outputs/generations/gsm8k_heuristic_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n1_seed${SEED}.jsonl \
    outputs/generations/gsm8k_verem_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n2_seed${SEED}.jsonl \
  --output outputs/metrics/gsm8k_main_table_${NUM}_seed${SEED}.csv

echo "=== Exporting case studies ==="

python -m src.analysis.case_study \
  --input outputs/generations/gsm8k_verem_${NUM}_s${STEPS}_is${INFILL_STEPS}_r${ROUNDS}_k${SPANS}_n2_seed${SEED}.jsonl \
  --output_dir outputs/case_studies

echo "Done."
