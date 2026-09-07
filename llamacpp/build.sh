#!/bin/bash
# Build llama.cpp with the Ouro architecture, on top of the upstream commit this was developed against.
# Usage: llamacpp/build.sh [target-dir]   (default: third_party/llama.cpp)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HERE/../third_party/llama.cpp}"
BASE=67672dc5b76f8bc17785a19d3dc6d1463fc2902c   # upstream master at time of writing

if [ ! -d "$DEST/.git" ]; then
  echo "==> cloning llama.cpp into $DEST"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
cd "$DEST"
git fetch --depth 1 origin "$BASE" 2>/dev/null || git fetch origin
git checkout -B ouro-arch "$BASE"
echo "==> applying the Ouro patch"
git am "$HERE/0001-model-add-Ouro.patch"
echo "==> building (Metal, no tests/examples)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
      --target llama-cli llama-server llama-quantize
echo "==> done: $DEST/build/bin/{llama-cli,llama-server,llama-quantize}"
