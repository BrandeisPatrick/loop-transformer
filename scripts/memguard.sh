#!/bin/bash
# Memory guard: polls system memory and, if it crosses a bound, kills THIS PROJECT'S jobs
# (never anything else) so the machine never gets into a swap storm. Everything it kills is
# resumable, so the cost of a false positive is a re-run, not lost work.
#
#   MEMGUARD_MIN_FREE   kill when free memory drops below this percent   (default 20)
#   MEMGUARD_MAX_SWAP   kill when swap in use exceeds this many MB       (default 2500)
#   MEMGUARD_INTERVAL   seconds between checks                           (default 3)
#
# Kill order (all in one sweep, biggest memory first):
#   1. model servers I started: serve/shim.py (HF looped models on MPS), lm_eval
#   2. every model Ollama has loaded  (ollama stop <name>) — frees GPU-wired memory immediately
#   3. the eval drivers (run_eval.py, run_*.sh, after_baselines.sh) so nothing reloads a model
# Log: results/memguard.log. Also posts a macOS notification.
cd "$(dirname "$0")/.."
MIN_FREE=${MEMGUARD_MIN_FREE:-20}; MAX_SWAP=${MEMGUARD_MAX_SWAP:-2500}; INTERVAL=${MEMGUARD_INTERVAL:-3}
LOG=results/memguard.log; mkdir -p results
if pgrep -f "scripts/memguard.sh" | grep -v "^$$\$" | grep -qv "^$$$"; then :; fi
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/{gsub(/%/,"",$5); print $5}'; }
swap_mb()  { sysctl -n vm.swapusage 2>/dev/null | awk '{gsub(/M/,"",$6); printf "%d", $6}'; }
loaded_models() { curl -s -m 2 http://127.0.0.1:11434/api/ps 2>/dev/null | python3 -c "import sys,json
try: print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))
except Exception: pass" 2>/dev/null; }
sweep() {
  local why="$1"
  log "KILL: $why (free=${F}% swap=${S}MB, bounds: free>=${MIN_FREE}% swap<=${MAX_SWAP}MB)"
  pkill -f "serve/shim.py" 2>/dev/null && log "  killed shim"
  pkill -f "lm_eval" 2>/dev/null && log "  killed lm_eval"
  for m in $(loaded_models); do ollama stop "$m" >/dev/null 2>&1 && log "  ollama stop $m"; done
  pkill -f "eval/run_eval.py" 2>/dev/null && log "  killed run_eval.py"
  pkill -f "scripts/run_baselines.sh|scripts/run_evals.sh|scripts/run_phase2.sh|scripts/after_baselines.sh|scripts/loopus_smoke.sh|scripts/run_mmlu_sweep.sh" 2>/dev/null && log "  killed eval chain scripts"
  osascript -e "display notification \"free ${F}%, swap ${S} MB — killed looplm jobs\" with title \"looplm memguard\"" >/dev/null 2>&1
}
log "memguard start: kill below ${MIN_FREE}% free or above ${MAX_SWAP}MB swap; every ${INTERVAL}s"
LASTWARN=0
while true; do
  F=$(free_pct); S=$(swap_mb); F=${F:-100}; S=${S:-0}
  if [ "$F" -lt "$MIN_FREE" ] || [ "$S" -gt "$MAX_SWAP" ]; then
    sweep "memory bound crossed"; sleep 10
  elif [ "$F" -lt $((MIN_FREE+10)) ] && [ $(( $(date +%s) - LASTWARN )) -gt 60 ]; then
    log "WARN: free=${F}% swap=${S}MB (approaching bound)"; LASTWARN=$(date +%s)
  fi
  sleep "$INTERVAL"
done
