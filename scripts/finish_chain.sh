#!/bin/bash
# After after_2.6b_chain.sh: (1) finish the 2.6B GSM8K run (resumes from its JSONL; one server slot),
# (2) re-run logit parity against an fp32 reference on CPU, so the comparison has no bf16 rounding in it.
set -u
cd "$(dirname "$0")/.."
HF26=https://huggingface.co/BrandeisPatrick/Ouro-2.6B-GGUF/resolve/main
HF14=https://huggingface.co/BrandeisPatrick/Ouro-1.4B-GGUF/resolve/main
log() { echo "### $(date '+%H:%M:%S') $*"; }
while pgrep -f "scripts/after_2.6b_chain.sh" >/dev/null; do sleep 30; done
log "previous chain finished: $(tail -1 results/after_2.6b_chain.log)"

# 1. finish the 2.6B run
curl -sS -L --fail -o models/Ouro-2.6B-Q8_0.gguf $HF26/Ouro-2.6B-Q8_0.gguf || { log "2.6B download failed"; exit 1; }
scripts/llamacpp_eval_2.6b.sh 2>&1 | tail -4
rm -f models/Ouro-2.6B-Q8_0.gguf
.venv/bin/python -c "
import json; rows=[json.loads(l) for l in open('results/gsm8k_ouro-2.6b_llamacpp_Q8_L4.jsonl')]
c=sum(r['correct'] for r in rows); print(f'2.6B GSM8K: {c}/{len(rows)} = {100*c/len(rows):.1f}%')"
log "2.6B done, disk free $(df -h / | tail -1 | awk '{print $4}')"

# 2. parity against fp32
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }
curl -sS -L --fail -o models/Ouro-1.4B-F16.gguf $HF14/Ouro-1.4B-F16.gguf || { log "F16 download failed"; exit 1; }
scripts/memwait.sh || { log "parity fp32: not enough memory"; exit 1; }
.venv-tf4/bin/python scripts/logit_parity.py --gguf models/Ouro-1.4B-F16.gguf --loops 1 2 4 \
    --device cpu --ref-dtype float32 --out results/logit_parity_fp32.json 2>&1 \
    | grep -v -i -E "warn|deprecat" | tee results/logit_parity_fp32.log
rm -rf results/parity_work ~/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B models/Ouro-1.4B-F16.gguf
pkill -f "scripts/memguard.sh"
log "FINISH CHAIN DONE, disk free $(df -h / | tail -1 | awk '{print $4}')"
