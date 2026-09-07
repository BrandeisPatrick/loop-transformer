#!/bin/bash
# Memory guard: polls system memory and, if it crosses a bound, kills THIS PROJECT'S jobs
# (never anything else) so the machine never gets into a swap storm. Everything it kills is
# resumable, so the cost of a false positive is a re-run, not lost work.
#
#   MEMGUARD_MIN_FREE   kill when free memory drops below this percent   (default 30)
#   MEMGUARD_SWAP_GROWTH kill when swap has GROWN by this many MB since the guard started (default 800)
#                       (absolute swap is not used: macOS does not drain swap after memory is freed,
#                        so an absolute bound would keep killing with nothing left to gain)
#   MEMGUARD_INTERVAL   seconds between checks                           (default 2)
#
# Kill order (all in one sweep, biggest memory first):
#   1. model servers I started: serve/shim.py (HF looped models on MPS), lm_eval
#   2. every model Ollama has loaded  (ollama stop <name>) — frees GPU-wired memory immediately
#   3. the eval drivers (run_eval.py, run_*.sh, after_baselines.sh) so nothing reloads a model
# Log: results/memguard.log. Also posts a macOS notification.
cd "$(dirname "$0")/.."
MIN_FREE=${MEMGUARD_MIN_FREE:-30}; SWAP_GROWTH=${MEMGUARD_SWAP_GROWTH:-800}; INTERVAL=${MEMGUARD_INTERVAL:-2}
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
  log "KILL: $why (free=${F}% swap=${S}MB, base swap ${S0}MB; bounds: free>=${MIN_FREE}%, swap growth<=${SWAP_GROWTH}MB)"
  # Patterns are anchored to the start of the command line so they match only the real
  # processes, never a shell whose command text merely mentions these names.
  pkill -f "^[^ ]*python serve/shim.py" 2>/dev/null && log "  killed shim"
  pkill -f "^[^ ]*(python|lm_eval) [^ ]*lm_eval" 2>/dev/null && log "  killed lm_eval"
  pkill -f "^[^ ]*/lm_eval " 2>/dev/null
  for m in $(loaded_models); do ollama stop "$m" >/dev/null 2>&1 && log "  ollama stop $m"; done
  pkill -f "^[^ ]*python eval/run_eval.py" 2>/dev/null && log "  killed run_eval.py"
  pkill -f "^(/bin/)?bash scripts/(run_baselines|run_evals|run_phase2|after_baselines|loopus_smoke|run_mmlu_sweep)\.sh" 2>/dev/null && log "  killed eval chain scripts"
  osascript -e "display notification \"free ${F}%, swap ${S} MB — killed looplm jobs\" with title \"looplm memguard\"" >/dev/null 2>&1
}
S0=$(swap_mb); S0=${S0:-0}
log "memguard start: kill below ${MIN_FREE}% free or if swap grows >${SWAP_GROWTH}MB from ${S0}MB; every ${INTERVAL}s"
LASTWARN=0
while true; do
  F=$(free_pct); S=$(swap_mb); F=${F:-100}; S=${S:-0}
  if [ "$F" -lt "$MIN_FREE" ] || [ $((S - S0)) -gt "$SWAP_GROWTH" ]; then
    sweep "memory bound crossed"; sleep 10; S0=$(swap_mb); S0=${S0:-$S0}   # re-baseline after a sweep
  elif [ "$F" -lt $((MIN_FREE+10)) ] && [ $(( $(date +%s) - LASTWARN )) -gt 60 ]; then
    log "WARN: free=${F}% swap=${S}MB (approaching bound)"; LASTWARN=$(date +%s)
  fi
  sleep "$INTERVAL"
done
