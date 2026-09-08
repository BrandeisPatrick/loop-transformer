#!/bin/bash
# Reproduce the Ouro depth curve through llama.cpp from ONE GGUF, using the runtime override
# --override-kv ouro.num_loops=int:N (the loader consults overrides before the file, so no
# reconversion is needed). Compares against the transformers reference measured in this repo:
#   depth 1 = 23.0%, depth 2 = 64.0%, depth 3 = 72.0%, depth 4 = 80.0%
set -u
cd "$(dirname "$0")/.."
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }
LC=third_party/llama.cpp/build/bin; MODEL=${MODEL:-models/Ouro-1.4B-F16.gguf}; N=${N:-100}
for L in ${DEPTHS:-1 2 4}; do
  scripts/memwait.sh || { echo "### skipping depth $L: not enough free memory"; continue; }
  pkill -f "build/bin/llama-server"; sleep 2
  echo "### $(date '+%H:%M:%S') depth $L"
  nohup $LC/llama-server -m "$MODEL" --override-kv ouro.num_loops=int:$L \
      -c 768 -ngl 99 --host 127.0.0.1 --port 8080 --no-warmup > /tmp/llama_server_d$L.log 2>&1 &
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 3; done
  .venv/bin/python eval/run_eval.py --base-url http://127.0.0.1:8080/v1 --model "ouro-gguf-L$L" \
      --benchmark gsm8k --mode completion --shots 3 --n "$N" --max-tokens 256 \
      --out "results/gsm8k_ouro-1.4b_llamacpp_L$L.jsonl" 2>&1 | grep -vE "Warning" | tail -3
done
pkill -f "build/bin/llama-server"
echo "### $(date '+%H:%M:%S') DEPTH SWEEP DONE"
