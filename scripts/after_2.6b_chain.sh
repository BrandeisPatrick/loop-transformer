#!/bin/bash
# Runs after the 2.6B GSM8K eval finishes: logit parity (item 1) then wikitext-2 perplexity (item 2),
# one model resident at a time, downloads deleted as soon as they are no longer needed (disk is tight).
set -u
cd "$(dirname "$0")/.."
LC=third_party/llama.cpp/build-clt/bin
HF=https://huggingface.co/BrandeisPatrick/Ouro-1.4B-GGUF/resolve/main
CHUNKS=${CHUNKS:-100}
log() { echo "### $(date '+%H:%M:%S') $*"; }

# 1. wait for the eval chain
while pgrep -f "scripts/llamacpp_eval_2.6b.sh" >/dev/null; do sleep 30; done
log "2.6B eval finished: $(tail -1 results/llamacpp_eval_2.6b.log)"
pkill -f "build/bin/llama-server" 2>/dev/null; sleep 2
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }

# 2. swap files: drop the 2.6B GGUF, fetch the 1.4B F16
rm -f models/Ouro-2.6B-Q8_0.gguf
curl -sS -L --fail -o models/Ouro-1.4B-F16.gguf $HF/Ouro-1.4B-F16.gguf || { log "F16 download failed"; exit 1; }
log "F16 ready, disk free $(df -h / | tail -1 | awk '{print $4}')"

# 3. logit parity at 1, 2, 4 loops (reference first, then GGUF; the script keeps one model resident)
scripts/memwait.sh || { log "parity: not enough memory"; exit 1; }
.venv-tf4/bin/python scripts/logit_parity.py --gguf models/Ouro-1.4B-F16.gguf --loops 1 2 4 \
    --out results/logit_parity.json 2>&1 | grep -v -i -E "warn|deprecat" | tee results/logit_parity.log
rm -rf results/parity_work ~/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B
log "parity done, disk free $(df -h / | tail -1 | awk '{print $4}')"

# 4. perplexity: F16 at 4/2/1 loops, then Q8_0 and Q4_K_M at 4 loops
curl -sS -L --fail -o models/Ouro-1.4B-Q8_0.gguf   $HF/Ouro-1.4B-Q8_0.gguf
curl -sS -L --fail -o models/Ouro-1.4B-Q4_K_M.gguf $HF/Ouro-1.4B-Q4_K_M.gguf
ppl() {  # $1 = gguf, $2 = loops, $3 = tag
  scripts/memwait.sh || { log "ppl $3: not enough memory"; return; }
  log "perplexity $3 (chunks=$CHUNKS)"
  $LC/llama-perplexity -m "$1" -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks $CHUNKS -ngl 99 \
      --override-kv ouro.num_loops=int:$2 > "results/ppl_$3.log" 2>&1
  grep -E "Final estimate|estimating|tokens|^\[1\]" "results/ppl_$3.log" | tail -2
}
ppl models/Ouro-1.4B-F16.gguf   4 F16_L4
ppl models/Ouro-1.4B-F16.gguf   2 F16_L2
ppl models/Ouro-1.4B-F16.gguf   1 F16_L1
ppl models/Ouro-1.4B-Q8_0.gguf  4 Q8_0_L4
ppl models/Ouro-1.4B-Q4_K_M.gguf 4 Q4_K_M_L4
rm -f models/Ouro-1.4B-F16.gguf models/Ouro-1.4B-Q8_0.gguf models/Ouro-1.4B-Q4_K_M.gguf
pkill -f "scripts/memguard.sh"
log "CHAIN DONE, disk free $(df -h / | tail -1 | awk '{print $4}')"
