#!/bin/bash
# Block until free memory is at least MEMGUARD_START_FREE percent (default 45) before loading a model.
# Trades time for safety: a load that would push the machine toward the kill bound simply waits.
NEED=${MEMGUARD_START_FREE:-45}; t0=$(date +%s)
while true; do
  F=$(memory_pressure 2>/dev/null | awk '/free percentage/{gsub(/%/,"",$5); print $5}'); F=${F:-100}
  [ "$F" -ge "$NEED" ] && exit 0
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "memwait: still only ${F}% free after 15 min; giving up this step" >&2; exit 1; }
  sleep 5
done
