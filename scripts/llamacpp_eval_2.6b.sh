#!/bin/bash
# GSM8K 3-shot for Ouro-2.6B through llama.cpp (Q8_0, 4 loops), same 200-problem subset and protocol as
# the 1.4B runs, so the result is comparable to the paper's 2.6B number and to the 1.4B port numbers.
# Memory: 48 physical layers x 4 loops = 192 KV slots with no GQA (16 KV heads), so the cache is
# quantized to q8_0 (needs flash attention) and the context kept at 1024. Guarded by memguard (30%).
set -u
cd "$(dirname "$0")/.."
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }
LC=third_party/llama.cpp/build/bin
MODEL=${MODEL:-models/Ouro-2.6B-Q8_0.gguf}; N=${N:-200}; L=${L:-4}
OUT=${OUT:-results/gsm8k_ouro-2.6b_llamacpp_Q8_L$L.jsonl}
scripts/memwait.sh || { echo "### not enough free memory"; exit 1; }
pkill -f "build/bin/llama-server"; sleep 2
echo "### $(date '+%H:%M:%S') Ouro-2.6B Q8_0 depth $L -> $OUT"
nohup $LC/llama-server -m "$MODEL" --override-kv ouro.num_loops=int:$L \
    -c 1024 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8080 --no-warmup \
    > results/llama_server_2.6b.log 2>&1 &
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 3; done
grep -E "print_info: (arch|n_layer |model params)" results/llama_server_2.6b.log | sed 's/.*print_info: //' | tr -s ' ' | tr '\n' ';'; echo
.venv/bin/python eval/run_eval.py --base-url http://127.0.0.1:8080/v1 --model "ouro-2.6b-q8-L$L" \
    --benchmark gsm8k --mode completion --shots 3 --n "$N" --max-tokens 256 \
    --out "$OUT" 2>&1 | grep -vE "Warning" | tail -3
pkill -f "build/bin/llama-server"
echo "### $(date '+%H:%M:%S') DONE"
