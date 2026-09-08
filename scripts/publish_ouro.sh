#!/bin/bash
# End-to-end: HF checkpoint -> GGUF -> quants -> validate -> upload to a Hugging Face repo.
# Deletes the source snapshot right after conversion and the F16 right after quantising, so the
# peak disk need is roughly (snapshot + F16) rather than the sum of everything.
#   usage: scripts/publish_ouro.sh ByteDance/Ouro-2.6B-Thinking BrandiesPatrick/Ouro-2.6B-Thinking-GGUF
set -u
cd "$(dirname "$0")/.."
SRC="$1"; DEST="$2"; NAME=$(basename "$SRC")
LC=third_party/llama.cpp; PY=.venv/bin/python
pgrep -f "scripts/memguard.sh" >/dev/null || { nohup scripts/memguard.sh >/dev/null 2>&1 & sleep 1; }

echo "### $(date '+%H:%M:%S') [1/5] download $SRC"
SNAP=$($PY - "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.py","*.txt","*.model"]))
PY
) || { echo "download failed"; exit 1; }
echo "  snapshot: $SNAP  ($(du -sh "$SNAP" | cut -f1))"

echo "### $(date '+%H:%M:%S') [2/5] convert to F16"
PYTHONPATH=$LC/gguf-py $PY $LC/convert_hf_to_gguf.py "$SNAP" --outfile "models/$NAME-F16.gguf" --outtype f16 2>&1 \
  | grep -E "num_loops|block_count|successfully|Traceback|Error" | tail -4
[ -f "models/$NAME-F16.gguf" ] || { echo "conversion failed"; exit 1; }
echo "  freeing the source snapshot"
$PY -c "
from huggingface_hub import scan_cache_dir
import sys
c=scan_cache_dir()
for r in c.repos:
    if r.repo_id=='$SRC':
        c.delete_revisions(*[rev.commit_hash for rev in r.revisions]).execute(); print('  deleted HF cache for $SRC')
"

echo "### $(date '+%H:%M:%S') [3/5] quantise"
for q in Q8_0 Q4_K_M; do
  $LC/build/bin/llama-quantize "models/$NAME-F16.gguf" "models/$NAME-$q.gguf" $q 2>&1 | tail -1
done
ls -lh models/$NAME*.gguf | awk '{print "  ",$5,$9}'

echo "### $(date '+%H:%M:%S') [4/5] verify (gates the upload)"
# This is a GATE, not a courtesy. The 2.6B-Thinking run uploaded unverified because the memory
# guard killed the smoke test and the script carried on regardless. Now a failure aborts.
scripts/memwait.sh || { echo "VERIFY SKIPPED: no memory. Not uploading unverified files."; exit 1; }
SHAPE=$($LC/build/bin/llama-cli -m "models/$NAME-Q8_0.gguf" -n 1 -c 512 --no-warmup --single-turn -p "hi" -v 2>&1 \
        | grep -E "print_info: (arch|n_layer |model params)" | sed 's/.*I //' | tr '\n' ' ')
echo "  $SHAPE"
echo "$SHAPE" | grep -q "arch *= *ouro" || { echo "VERIFY FAILED: architecture is not ouro. Not uploading."; exit 1; }
# Verify on the SMALLEST quant with a small context: the point is to exercise the graph, and a
# lighter load is far less likely to be killed by the memory guard mid-test.
# An empty answer means the verifier was killed, not that the model is wrong -- retry those; only a
# non-empty WRONG answer is a real failure.
VERIFY_MODEL="models/$NAME-Q4_K_M.gguf"; [ -f "$VERIFY_MODEL" ] || VERIFY_MODEL="models/$NAME-Q8_0.gguf"
OK=0
for attempt in 1 2 3; do
  scripts/memwait.sh || { echo "  attempt $attempt: waiting for memory"; sleep 30; continue; }
  OUT=$($LC/build/bin/llama-cli -m "$VERIFY_MODEL" --single-turn --temp 0 -n 64 -c 512 --no-warmup \
        -p "What is 17 + 26? Answer briefly." 2>/dev/null | tail -6)
  BODY=$(echo "$OUT" | tr '\n' ' ' | sed 's/.*Answer briefly\.//;s/\[ Prompt.*//' | tr -d ' ')
  if echo "$OUT" | grep -q "43"; then echo "  answer contains 43 — verification passed"; OK=1; break; fi
  if [ -z "$BODY" ]; then echo "  attempt $attempt produced no output (verifier killed?); retrying"; sleep 20; continue; fi
  echo "VERIFY FAILED: model answered '$BODY', expected 43. Not uploading."; exit 1
done
[ "$OK" = 1 ] || { echo "VERIFY INCONCLUSIVE after 3 attempts (never produced output). Not uploading."; exit 1; }

echo "### $(date '+%H:%M:%S') [5/5] upload to $DEST"
$PY - "$DEST" "$NAME" <<'PY'
import sys, os, time
from huggingface_hub import HfApi
dest, name = sys.argv[1], sys.argv[2]
api = HfApi(); api.create_repo(repo_id=dest, repo_type="model", exist_ok=True, private=False)
for f in [f"{name}-Q8_0.gguf", f"{name}-Q4_K_M.gguf", f"{name}-F16.gguf"]:
    p = os.path.join("models", f)
    if not os.path.exists(p): continue
    t = time.time(); api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=dest, repo_type="model")
    print(f"  uploaded {f} ({os.path.getsize(p)/1e9:.2f} GB) in {time.time()-t:.0f}s", flush=True)
print("  UPLOADS DONE", flush=True)
PY
echo "### $(date '+%H:%M:%S') DONE — https://huggingface.co/$DEST"
