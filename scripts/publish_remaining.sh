#!/bin/bash
# Publish the remaining Ouro variants one at a time, clearing local GGUFs between them.
cd "$(dirname "$0")/.."
for pair in "ByteDance/Ouro-1.4B-Thinking BrandiesPatrick/Ouro-1.4B-Thinking-GGUF" \
            "ByteDance/Ouro-2.6B BrandiesPatrick/Ouro-2.6B-GGUF"; do
  set -- $pair
  echo "══════════ $1 ══════════"
  scripts/publish_ouro.sh "$1" "$2" || echo "!!! FAILED: $1 (nothing uploaded for it)"
  rm -f models/$(basename "$1")-*.gguf
  df -h /System/Volumes/Data | tail -1 | awk '{print "disk free:",$4}'
done
echo "══════════ ALL REMAINING MODELS DONE ══════════"
