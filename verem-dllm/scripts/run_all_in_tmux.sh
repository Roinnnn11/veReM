#!/bin/bash
# 一键在 tmux 中跑完所有 GSM8K 实验（跳过已有 vanilla 结果）
# 用法：bash scripts/run_all_in_tmux.sh [num_samples] [seed]
#
# 会创建 tmux session "verem_exp"，实验日志写到 outputs/run_v2.log
# 查看进度：tmux attach -t verem_exp
#           或 tail -f outputs/run_v2.log

set -e

NUM=${1:-200}
SEED=${2:-42}
SESSION="verem_exp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG="$REPO_DIR/outputs/run_v2.log"

mkdir -p "$REPO_DIR/outputs/generations" "$REPO_DIR/outputs/metrics"

# 找 tmux（conda 环境里的）
TMUX_BIN=$(which tmux 2>/dev/null || echo "/root/miniconda3/envs/verem/bin/tmux")
if [ ! -x "$TMUX_BIN" ]; then
  echo "找不到 tmux，请先运行: conda activate verem"
  exit 1
fi

# 如果 session 已存在就先删掉
$TMUX_BIN has-session -t "$SESSION" 2>/dev/null && $TMUX_BIN kill-session -t "$SESSION"

echo "启动 tmux session: $SESSION"
echo "日志文件: $LOG"
echo "查看进度: $TMUX_BIN attach -t $SESSION"
echo "         或 tail -f $LOG"
echo ""

$TMUX_BIN new-session -d -s "$SESSION" bash -c "
  set -e
  cd '$REPO_DIR'
  conda activate verem 2>/dev/null || true

  MODEL='\${MODEL:-/root/autodl-fs/LLaDA-8B-Instruct}'
  STEPS=256
  INFILL_STEPS=128
  MAX_TOKENS=256
  BLOCK_LENGTH=32
  NUM=$NUM
  SEED=$SEED

  exec > >(tee -a '$LOG') 2>&1

  echo '========================================'
  echo ' VeReM GSM8K 实验 N='\$NUM' seed='\$SEED
  echo ' 开始时间: '\$(date)
  echo '========================================'

  # --- vanilla（如果已有 200 条就跳过）---
  VANILLA_OUT=outputs/generations/gsm8k_vanilla_\${NUM}_s\${STEPS}_seed\${SEED}.jsonl
  VANILLA_COUNT=\$([ -f \"\$VANILLA_OUT\" ] && wc -l < \"\$VANILLA_OUT\" || echo 0)
  if [ \"\$VANILLA_COUNT\" -ge \"\$NUM\" ]; then
    echo \"[跳过] vanilla 已有 \$VANILLA_COUNT 条结果\"
  else
    echo \"[1/4] vanilla ...\"
    python -m src.eval.run_gsm8k \\
      --model_name \$MODEL --method vanilla \\
      --num_samples \$NUM --steps \$STEPS \\
      --max_new_tokens \$MAX_TOKENS --block_length \$BLOCK_LENGTH \\
      --seed \$SEED --output \$VANILLA_OUT
  fi

  # --- random_remask ---
  echo \"[2/4] random_remask ...\"
  python -m src.eval.run_gsm8k \\
    --model_name \$MODEL --method random_remask \\
    --num_samples \$NUM --steps \$STEPS --infill_steps \$INFILL_STEPS \\
    --max_new_tokens \$MAX_TOKENS --block_length \$BLOCK_LENGTH \\
    --max_revision_rounds 1 --max_spans_per_example 3 --samples_per_span 1 \\
    --seed \$SEED \\
    --output outputs/generations/gsm8k_random_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n1_seed\${SEED}.jsonl

  # --- heuristic_remask ---
  echo \"[3/4] heuristic_remask ...\"
  python -m src.eval.run_gsm8k \\
    --model_name \$MODEL --method heuristic_remask \\
    --num_samples \$NUM --steps \$STEPS --infill_steps \$INFILL_STEPS \\
    --max_new_tokens \$MAX_TOKENS --block_length \$BLOCK_LENGTH \\
    --max_revision_rounds 1 --max_spans_per_example 3 --samples_per_span 1 \\
    --seed \$SEED \\
    --output outputs/generations/gsm8k_heuristic_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n1_seed\${SEED}.jsonl

  # --- verem ---
  echo \"[4/4] verem ...\"
  python -m src.eval.run_gsm8k \\
    --model_name \$MODEL --method verem \\
    --num_samples \$NUM --steps \$STEPS --infill_steps \$INFILL_STEPS \\
    --max_new_tokens \$MAX_TOKENS --block_length \$BLOCK_LENGTH \\
    --max_revision_rounds 1 --max_spans_per_example 3 --samples_per_span 2 \\
    --seed \$SEED \\
    --output outputs/generations/gsm8k_verem_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n2_seed\${SEED}.jsonl

  # --- 汇总结果 ---
  echo ''
  echo '========================================'
  echo ' 汇总结果'
  echo '========================================'
  python -c \"
import json, sys
sys.path.insert(0, '.')
from src.verifiers.gsm8k_verifier import extract_answer

files = [
  ('vanilla',   'outputs/generations/gsm8k_vanilla_\${NUM}_s\${STEPS}_seed\${SEED}.jsonl'),
  ('random',    'outputs/generations/gsm8k_random_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n1_seed\${SEED}.jsonl'),
  ('heuristic', 'outputs/generations/gsm8k_heuristic_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n1_seed\${SEED}.jsonl'),
  ('verem',     'outputs/generations/gsm8k_verem_\${NUM}_s\${STEPS}_is\${INFILL_STEPS}_r1_k3_n2_seed\${SEED}.jsonl'),
]
print(f'{\\'Method\\':<12} {\\'N\\':<6} {\\'Init Acc\\':<10} {\\'Final Acc\\':<11} {\\'Fix Rate\\'}')
print('-' * 55)
for name, path in files:
    try:
        records = [json.loads(l) for l in open(path)]
        n = len(records)
        init_c = sum(extract_answer(r.get(\\'initial_output\\', r[\\'final_output\\'])) == extract_answer(r[\\'gold\\'])
                     for r in records)
        final_c = sum(extract_answer(r[\\'final_output\\']) == extract_answer(r[\\'gold\\'])
                      for r in records)
        wrong = [r for r in records if extract_answer(r.get(\\'initial_output\\', r[\\'final_output\\'])) != extract_answer(r[\\'gold\\'])]
        fixed = [r for r in wrong if extract_answer(r[\\'final_output\\']) == extract_answer(r[\\'gold\\'])]
        fix_rate = len(fixed)/len(wrong) if wrong else 0
        print(f'{name:<12} {n:<6} {init_c/n:<10.3f} {final_c/n:<11.3f} {fix_rate:.3f}')
    except FileNotFoundError:
        print(f'{name:<12} (文件不存在)')
\"

  echo ''
  echo '完成时间:' \$(date)
  echo '按任意键退出...'
  read -n1
"

echo "已在后台启动，用以下命令查看："
echo "  $TMUX_BIN attach -t $SESSION"
