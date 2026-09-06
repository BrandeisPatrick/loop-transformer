#!/bin/bash
# Create Ollama models that are byte-identical copies of a looped GGUF with a different `<arch>.num_loops`
# metadata value, so the loop count can be swept from inside Ollama (llama.cpp reads num_loops at load and
# unrolls the shared layers that many times).
#   usage: scripts/make_loop_variants.sh <ollama-model-name> <short-name> <loops...>
#   e.g.:  scripts/make_loop_variants.sh hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M nanbeige4.2 1 2 3
set -euo pipefail
SRC_MODEL="$1"; SHORT="$2"; shift 2
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HERE/.venv/bin/python"
mkdir -p "$HERE/models"
BLOB=$(ollama show --modelfile "$SRC_MODEL" | awk '/^FROM /{print $2; exit}')
[ -f "$BLOB" ] || { echo "cannot find GGUF blob for $SRC_MODEL (got: $BLOB)"; exit 1; }
ARCH=$("$PY" "$HERE/scripts/gguf_header.py" "$BLOB" 4000000 | awk -F' = ' '/^general.architecture/{print $2}')
echo "source blob: $BLOB (arch=$ARCH, $(du -h "$BLOB" | cut -f1))"
TEMPLATE=$(ollama show --modelfile "$SRC_MODEL" | sed -n '/^TEMPLATE /,$p')
for L in "$@"; do
  OUT="$HERE/models/${SHORT}-loops${L}.gguf"
  if [ ! -f "$OUT" ]; then cp "$BLOB" "$OUT"; fi
  "$PY" -m gguf.scripts.gguf_set_metadata "$OUT" "${ARCH}.num_loops" "$L" --force >/dev/null
  "$PY" "$HERE/scripts/gguf_header.py" "$OUT" 4000000 | grep -E "num_loops"
  MF="$HERE/models/${SHORT}-loops${L}.Modelfile"
  { echo "FROM $OUT"; echo "$TEMPLATE"; } > "$MF"
  ollama create "${SHORT}:loops${L}" -f "$MF" && rm -f "$OUT"   # blob now lives in ~/.ollama; drop the working copy
done
ollama list | grep -E "^${SHORT}"
