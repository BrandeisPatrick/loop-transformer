#!/bin/bash
# Ouro-1.4B MMLU 5-shot (log-likelihood) at T=1..4 with lm-evaluation-harness — the paper's per-step
# ablation (Table 10: 41.21 / 60.43 / 66.71 / 67.45). --limit 0.05 = ~700 questions (SE ~1.7 pt).
# Run only when nothing else holds the GPU (loads its own 3 GB copy of Ouro). Logs to results/mmlu/.
cd "$(dirname "$0")/.."
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }   # memory guard
mkdir -p results/mmlu
LIMIT=${LIMIT:-0.05}
for T in ${TS:-1 4 2 3}; do
  scripts/memwait.sh || { echo "### skipping T=$T: not enough free memory"; continue; }
  echo "### $(date '+%H:%M:%S') MMLU T=$T limit=$LIMIT"
  .venv-tf4/bin/lm_eval --model hf \
    --model_args "pretrained=ByteDance/Ouro-1.4B,trust_remote_code=True,dtype=bfloat16,total_ut_steps=$T" \
    --tasks mmlu --num_fewshot 5 --device mps --batch_size 1 --limit $LIMIT \
    --output_path results/mmlu/ouro-1.4b_T$T 2>&1 | grep -vE "Warning|warn|it/s\]|%\|" | tail -8
done
echo "### $(date '+%H:%M:%S') MMLU SWEEP DONE"
