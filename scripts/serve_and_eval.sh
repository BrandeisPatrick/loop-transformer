#!/bin/bash
# Serve a GGUF with llama-server and run a GSM8K eval against it, safely:
#  - waits for real readiness ("status":"ok"), not just a responding socket (503 while loading)
#  - never pkills a pattern that could match the calling shell (bracket guard)
#  - resumable: re-running continues the same results file
# usage: scripts/serve_and_eval.sh <model.gguf> <out.jsonl> [n] [--override-kv k=int:v ...]
set -u
cd "$(dirname "$0")/.."
MODEL="$1"; OUT="$2"; N="${3:-100}"; shift 3 2>/dev/null || shift $#
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }
pkill -f "[b]uild/bin/llama-server"; sleep 2
scripts/memwait.sh || { echo "not enough free memory"; exit 1; }
nohup third_party/llama.cpp/build/bin/llama-server -m "$MODEL" "$@" \
    -c 768 -ngl 99 --host 127.0.0.1 --port 8080 --no-warmup > /tmp/serve_$$.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"' && break
  sleep 3
done
curl -s http://127.0.0.1:8080/health | grep -q '"ok"' || { echo "server never became ready; see /tmp/serve_$$.log"; exit 1; }
echo "serving $(basename "$MODEL") $*"
.venv/bin/python eval/run_eval.py --base-url http://127.0.0.1:8080/v1 --model "$(basename "$MODEL" .gguf)" \
    --benchmark gsm8k --mode completion --shots 3 --n "$N" --max-tokens 256 --out "$OUT" 2>&1 | grep -vE "Warning" | tail -3
pkill -f "[b]uild/bin/llama-server"
