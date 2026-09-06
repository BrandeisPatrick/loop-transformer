#!/bin/bash
# Phase 2 — run AFTER scripts/run_evals.sh has exited (it needs the memory).
# 1) LoopUS (Qwen3-1.7B looped) vs the Qwen3-1.7B-Base already measured in phase 1.
#    This is an original measurement, not a replication: the LoopUS paper reports no GSM8K/MATH.
# 2) Ouro-1.4B MMLU 5-shot at T=1..4 — the paper's per-step ablation (41.21/60.43/66.71/67.45).
set -u
cd "$(dirname "$0")/.."
echo "=== $(date '+%H:%M:%S') phase 2 start ==="
pkill -f "serve/shim.py" 2>/dev/null; sleep 3          # free the Ouro shim's ~3 GB
curl -s -m 3 http://127.0.0.1:11434/api/ps >/dev/null && echo "(ollama up; models unload after keep_alive)"
bash scripts/loopus_smoke.sh
echo "=== $(date '+%H:%M:%S') LoopUS done; MMLU sweep next ==="
TS="4 1" LIMIT=0.03 bash scripts/run_mmlu_sweep.sh
echo "=== $(date '+%H:%M:%S') phase 2 done ==="
.venv/bin/python eval/report.py > /dev/null && echo "REPORT.md updated"
