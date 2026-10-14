#!/bin/bash
# Re-check the published Ouro-2.6B-Thinking Q8_0 on the PR-head build, two ways:
#   A. plain llama-cli greedy smoke prompt (the publish gate)
#   B. llama-server with the flag set a tester reported garbage with (adapted: -c 4096 instead of 65536,
#      which would need ~50 GB of KV for 192 layers; Metal instead of CUDA)
set -u
cd "$(dirname "$0")/.."
LC=third_party/llama.cpp/build-clt/bin
F=models/Ouro-2.6B-Thinking-Q8_0.gguf
OUT=results/check_2.6b_thinking.log
free_pct() { memory_pressure | awk '/free percentage/ {gsub("%","",$NF); print $NF}'; }
watch() { local pid=$1 minf=100; while kill -0 $pid 2>/dev/null; do p=$(free_pct); [ -n "$p" ] && [ "$p" -lt "$minf" ] && minf=$p; [ -n "$p" ] && [ "$p" -lt 30 ] && { echo "WATCHDOG kill at ${p}%"; kill $pid; }; sleep 2; done; echo "min free ${minf}%"; }
: > $OUT
echo "### build $(git -C third_party/llama.cpp log -1 --format=%h) | free before $(free_pct)%" | tee -a $OUT
curl -sS -L --fail -o $F https://huggingface.co/BrandeisPatrick/Ouro-2.6B-Thinking-GGUF/resolve/main/Ouro-2.6B-Thinking-Q8_0.gguf || { echo "DOWNLOAD FAILED" | tee -a $OUT; exit 1; }
echo "downloaded $(ls -lh $F | awk '{print $5}')" | tee -a $OUT

echo "### A. llama-cli greedy smoke" | tee -a $OUT
$LC/llama-cli -m $F --single-turn --temp 0 -n 700 -c 2048 --no-warmup -v -p "What is 17 + 26? Answer briefly." > results/_a.out 2> results/_a.err &
watch $! | tee -a $OUT
grep -E "print_info: (arch|n_layer |model params)" results/_a.err | sed 's/.*print_info: //' | tr -s ' ' | tee -a $OUT
sed -n '/Answer briefly/,$p' results/_a.out | head -c 1500 | tee -a $OUT; echo | tee -a $OUT
grep -q "43" results/_a.out && echo "A RESULT: contains 43" | tee -a $OUT || echo "A RESULT: 43 NOT FOUND" | tee -a $OUT

echo "### B. llama-server with the tester's flags (adapted)" | tee -a $OUT
$LC/llama-server -m $F --host 127.0.0.1 --port 8081 -nkvo --fit off -ngl 48 -c 4096 \
    --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 --spec-type ngram-mod --spec-ngram-mod-n-max 128 \
    --cache-ram 512 --ctx-checkpoints 4 --checkpoint-min-step 1024 -fa on --threads 6 --no-warmup \
    > results/_b.err 2>&1 &
SPID=$!
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8081/health | grep -q ok && break; sleep 3; done
grep -E "print_info: (arch|n_layer |model params)|offloaded|spec" results/_b.err | sed 's/.*print_info: //' | tr -s ' ' | head -6 | tee -a $OUT
curl -s -m 600 http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"ouro","temperature":0,"max_tokens":500,
  "messages":[{"role":"user","content":"What is 17 + 26? Answer briefly."}]}' > results/_b.json &
watch $! | tee -a $OUT
python3 -I -c "
import json; d=json.load(open('results/_b.json')); m=d['choices'][0]['message']
txt=(m.get('reasoning_content') or m.get('reasoning') or '')+'\n---content---\n'+(m.get('content') or '')
print(txt[:1500]); print('B RESULT:', 'contains 43' if '43' in txt else '43 NOT FOUND', '| finish:', d['choices'][0].get('finish_reason'))" | tee -a $OUT
kill $SPID 2>/dev/null; sleep 1
rm -f $F results/_a.out results/_a.err results/_b.err results/_b.json
echo "### DONE, file removed, free $(free_pct)%" | tee -a $OUT
