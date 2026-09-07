#!/bin/bash
# One-time, user-approved: run the MMLU depth-4 point under a 20% free-memory bound (it peaked at 29%
# under the default 30% twice, with zero swap growth), then restore the 30% bound. Waits for any
# running sweep to finish first so only one model is ever loaded.
cd "$(dirname "$0")/.."
while pgrep -f "^(/bin/)?bash scripts/run_mmlu_sweep.sh" >/dev/null; do sleep 30; done
echo "### $(date '+%H:%M:%S') depth-1/2 sweep finished; relaxing guard to 20% for the depth-4 run"
pkill -f "[s]cripts/memguard.sh"; sleep 1
MEMGUARD_MIN_FREE=20 nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 2
scripts/memwait.sh && TS="4" LIMIT=0.03 bash scripts/run_mmlu_sweep.sh
echo "### $(date '+%H:%M:%S') depth-4 run finished; restoring guard to 30%"
pkill -f "[s]cripts/memguard.sh"; sleep 1
nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 2
tail -1 results/memguard.log
