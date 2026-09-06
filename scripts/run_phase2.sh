#!/bin/bash
# Phase 2 — run AFTER scripts/run_evals.sh has exited (it needs the memory).
# 1) LoopUS (Qwen3-1.7B looped) vs the Qwen3-1.7B-Base already measured in phase 1.
#    This is an original measurement, not a replication: the LoopUS paper reports no GSM8K/MATH.
# 2) Ouro-1.4B MMLU 5-shot at T=1..4 — the paper's per-step ablation (41.21/60.43/66.71/67.45).
set -u
cd "$(dirname "$0")/.."
echo "=== $(date '+%H:%M:%S') phase 2 start ==="
# Re-run the phase-1 Ollama evals: everything already answered is skipped, so this only retries
# items that hit a transient 500. Cheap, and it removes a silent penalty from the accuracy.
NB="hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M"
.venv/bin/python eval/run_eval.py --api ollama --think off --base-url http://127.0.0.1:11434/v1 \
  --model "$NB" --benchmark gsm8k --n 200 --max-tokens 1024 \
  --out results/gsm8k_nanbeige4.2-3b_loops2_nothink.jsonl 2>&1 | grep -vE "Warning" | tail -3
pkill -f "serve/shim.py" 2>/dev/null; sleep 3          # free the Ouro shim's ~3 GB
curl -s -m 3 http://127.0.0.1:11434/api/ps >/dev/null && echo "(ollama up; models unload after keep_alive)"
bash scripts/loopus_smoke.sh
echo "=== $(date '+%H:%M:%S') LoopUS done; MMLU sweep next ==="
TS="4 1" LIMIT=0.03 bash scripts/run_mmlu_sweep.sh
echo "=== $(date '+%H:%M:%S') phase 2 done ==="
.venv/bin/python eval/report.py > /dev/null && echo "REPORT.md updated"
