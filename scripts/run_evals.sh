#!/bin/bash
# Sequential eval chain (each run is resumable; re-run to continue). Logs to results/run_evals.log
# Memory policy: Ollama is started with OLLAMA_MAX_LOADED_MODELS=1 / KEEP_ALIVE=1m; the Ouro shim is
# killed as soon as the Ouro runs are done so at most one model is resident at any time.
cd "$(dirname "$0")/.."
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }   # memory guard: kills our jobs if free RAM < 20% or swap > 2.5 GB
PY=.venv/bin/python
OLL=http://127.0.0.1:11434/v1
SHIM=http://127.0.0.1:11435/v1
NB="hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M"
run() { echo "### $(date '+%H:%M:%S') $*"; "$@" 2>&1 | grep -vE "Warning|warn" | tail -4; }
# --- Phase 1: Ouro-1.4B via shim. GSM8K base-model protocol: 3-shot CoT, strict "The answer is N", greedy, 256 tok
run $PY eval/run_eval.py --base-url $SHIM --model ouro-1.4b --benchmark gsm8k --mode completion --shots 3 --n 200 --max-tokens 256 --extra-body '{"num_loops": 4}' --out results/gsm8k_ouro-1.4b_3shot_T4.jsonl
run $PY eval/run_eval.py --base-url $SHIM --model ouro-1.4b --benchmark gsm8k --mode completion --shots 3 --n 100 --max-tokens 256 --extra-body '{"num_loops": 1}' --out results/gsm8k_ouro-1.4b_3shot_T1.jsonl
run $PY eval/run_eval.py --base-url $SHIM --model ouro-1.4b --benchmark gsm8k --mode completion --shots 3 --n 100 --max-tokens 256 --extra-body '{"num_loops": 2}' --out results/gsm8k_ouro-1.4b_3shot_T2.jsonl
run $PY eval/run_eval.py --base-url $SHIM --model ouro-1.4b --benchmark gsm8k --mode completion --shots 3 --n 100 --max-tokens 256 --extra-body '{"num_loops": 3}' --out results/gsm8k_ouro-1.4b_3shot_T3.jsonl
echo "### $(date '+%H:%M:%S') Ouro runs done -> stopping shim to free memory"; pkill -f "serve/shim.py"; sleep 2
# --- Phase 2: Ollama-hosted models. Instruct protocol: zero-shot CoT + boxed, thinking off, greedy
run $PY eval/run_eval.py --api ollama --think off --base-url $OLL --model "$NB" --benchmark gsm8k --n 200 --max-tokens 1024 --out results/gsm8k_nanbeige4.2-3b_loops2_nothink.jsonl
run $PY eval/run_eval.py --api ollama --think off --base-url $OLL --model qwen3:1.7b --benchmark gsm8k --n 200 --max-tokens 1024 --out results/gsm8k_qwen3-1.7b_nothink.jsonl
run $PY eval/run_eval.py --api ollama --think off --base-url $OLL --model nanbeige4.2:loops1 --benchmark gsm8k --n 30 --max-tokens 256 --out results/gsm8k_nanbeige4.2-3b_loops1_nothink.jsonl
# --- Phase 3: Nanbeige native thinking mode (its published setting), small subset
run $PY eval/run_eval.py --api ollama --think on --base-url $OLL --model "$NB" --benchmark gsm8k --n 50 --max-tokens 4096 --out results/gsm8k_nanbeige4.2-3b_loops2_think.jsonl
echo "### $(date '+%H:%M:%S') ALL DONE"; $PY eval/summarize.py
