#!/bin/bash
# Wait for the baseline chain to exit, then run phase 2 (LoopUS test + MMLU depth sweep) while memory is free.
cd "$(dirname "$0")/.."
while pgrep -f run_baselines.sh >/dev/null; do sleep 60; done
echo "=== $(date '+%H:%M:%S') baselines finished; starting phase 2 ==="
exec scripts/run_phase2.sh
