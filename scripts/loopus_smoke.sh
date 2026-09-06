#!/bin/bash
# Bring up LoopUS (Qwen3-1.7B looped, custom lds arch) behind the shim on port 11436, smoke it, run a small
# GSM8K few-shot at two recursion depths, then shut it down. Needs ~4 GB (bf16 on MPS) — run when the Ouro shim is down.
cd "$(dirname "$0")/.."
S=${SCRATCH:-/tmp}; N=${N:-30}
pkill -f "port 11436" 2>/dev/null; sleep 1
nohup .venv/bin/python serve/shim.py --model Thrillcrazyer/Qwen3_1.7B_LoopUS --backend lds --name loopus-qwen3-1.7b --loops ${LOOPS:-8} --dtype ${DTYPE:-bfloat16} --port 11436 > $S/shim_loopus.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:11436/api/version >/dev/null 2>&1 && break; sleep 3; done
grep -E "^\[shim\]|Error|error" $S/shim_loopus.log | tail -3
echo "== smoke: few-shot completion at N=8 and N=1 =="
for L in 8 1; do .venv/bin/python - $L <<'PY'
import sys, requests
sys.path.insert(0, 'eval'); from run_eval import fewshot_prompt, extract_strict
L=int(sys.argv[1]); q="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
r=requests.post("http://127.0.0.1:11436/v1/completions", json={"model":"x","prompt":fewshot_prompt(q,3),"max_tokens":96,"temperature":0,"stop":["Q:"],"num_loops":L,"exit_threshold":1.0}, timeout=900).json()
t=r["choices"][0]["text"]; print(f"[N={L}] strict={extract_strict(t)!r} | {r['usage'].get('looplm')} | {t.strip()[:160]!r}")
PY
done
echo "== GSM8K n=$N at N=8 (q_threshold 0.9 default halting) and N=1 =="
.venv/bin/python eval/run_eval.py --base-url http://127.0.0.1:11436/v1 --model loopus-qwen3-1.7b --benchmark gsm8k --mode completion --shots 3 --n $N --max-tokens 256 --extra-body '{"num_loops": 8}' --out results/gsm8k_loopus-qwen3-1.7b_3shot_N8.jsonl 2>&1 | grep -vE "Warning" | tail -3
.venv/bin/python eval/run_eval.py --base-url http://127.0.0.1:11436/v1 --model loopus-qwen3-1.7b --benchmark gsm8k --mode completion --shots 3 --n $N --max-tokens 256 --extra-body '{"num_loops": 1}' --out results/gsm8k_loopus-qwen3-1.7b_3shot_N1.jsonl 2>&1 | grep -vE "Warning" | tail -3
pkill -f "port 11436"; echo "loopus shim stopped"
