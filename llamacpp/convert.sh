#!/bin/bash
# Convert an Ouro checkpoint to GGUF and (optionally) quantise it.
# Usage: llamacpp/convert.sh ByteDance/Ouro-1.4B [outdir] [quant...]
#   e.g. llamacpp/convert.sh ByteDance/Ouro-1.4B models Q4_K_M Q8_0
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/.."
REPO="${1:-ByteDance/Ouro-1.4B}"; OUT="${2:-$ROOT/models}"; shift 2 2>/dev/null || shift $# 
LC="$ROOT/third_party/llama.cpp"; PY="$ROOT/.venv/bin/python"
NAME=$(basename "$REPO")
mkdir -p "$OUT"
SNAP=$("$PY" - "$REPO" <<'PY'
import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.py","*.txt","*.model"]))
PY
)
echo "==> snapshot: $SNAP"
PYTHONPATH="$LC/gguf-py" "$PY" "$LC/convert_hf_to_gguf.py" "$SNAP" --outfile "$OUT/$NAME-F16.gguf" --outtype f16
for q in "$@"; do
  echo "==> quantising $q"
  "$LC/build/bin/llama-quantize" "$OUT/$NAME-F16.gguf" "$OUT/$NAME-$q.gguf" "$q"
done
ls -lh "$OUT"/$NAME*.gguf
