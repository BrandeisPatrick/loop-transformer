#!/bin/bash
# Matched baselines, run through the identical harness and protocols as the looped models.
#   Qwen3-4B-Base  -> Ouro's larger base comparator (paper Table 7: 72.86 vs Ouro-1.4B 78.92), base protocol
#   Qwen3.5-4B     -> Nanbeige4.2-3B's own card comparator, instruct protocol, thinking off
cd "$(dirname "$0")/.."
PY=.venv/bin/python; OLL=http://127.0.0.1:11434/v1
run() { echo "### $(date '+%H:%M:%S') $*"; "$@" 2>&1 | grep -vE "Warning|warn" | tail -4; }
for m in "hf.co/mradermacher/Qwen3-4B-Base-GGUF:Q4_K_M" "qwen3.5:4b"; do
  echo "### $(date '+%H:%M:%S') pull $m"; ollama pull "$m" 2>&1 | tr '\r' '\n' | grep -E "success|error|Error" | tail -1
done
ollama list
run $PY eval/run_eval.py --api ollama --base-url $OLL --model "hf.co/mradermacher/Qwen3-4B-Base-GGUF:Q4_K_M" --benchmark gsm8k --mode completion --shots 3 --n 200 --max-tokens 256 --out results/gsm8k_qwen3-4b-base_3shot.jsonl
run $PY eval/run_eval.py --api ollama --think off --base-url $OLL --model qwen3.5:4b --benchmark gsm8k --n 200 --max-tokens 1024 --out results/gsm8k_qwen3.5-4b_nothink.jsonl
echo "### $(date '+%H:%M:%S') BASELINES DONE"; $PY eval/report.py >/dev/null; $PY eval/summarize.py
