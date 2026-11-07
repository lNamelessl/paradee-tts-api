#!/usr/bin/env bash
# Bake the Paradee 1.2 weights into the image from a pinned Hugging Face commit.
#
# The tag `v1.2` is deliberately NOT used: tags move, commits do not. Both files
# are verified by size and sha256, so a changed or truncated download fails the
# build instead of producing a subtly different voice at runtime.
set -euo pipefail

DEST="${MODELS_DIR:-/app/models}"
REPO="sahilmahendrakar/Paradee-8M-v1.2"
REV="ed2934d4df8a4fd891e29584c5e1104649409765"   # tag v1.2

ONNX_SHA="a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1"
ONNX_SIZE="8462119"
CFG_SHA="002535cf8dd08a434cdddb13372da8a7b4844152bdf9b5b4e7342e12ec30bcfb"
CFG_SIZE="1827"

mkdir -p "$DEST/model"

fetch() { # url dest sha size label
  local url="$1" dest="$2" sha="$3" size="$4" label="$5"
  echo "[models] fetching $label"
  curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -o "$dest" "$url"
  local got_size got_sha
  got_size="$(wc -c < "$dest" | tr -d ' ')"
  got_sha="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [ "$got_size" != "$size" ]; then
    echo "[models] FAIL $label size $got_size != $size" >&2
    exit 1
  fi
  if [ "$got_sha" != "$sha" ]; then
    echo "[models] FAIL $label sha256 $got_sha != $sha" >&2
    exit 1
  fi
  echo "[models] ok $label ($got_size bytes, sha256 ${got_sha:0:12}…)"
}

BASE="https://huggingface.co/${REPO}/resolve/${REV}"
fetch "${BASE}/onnx/paradee_int8.onnx" "${DEST}/paradee_int8.onnx" "$ONNX_SHA" "$ONNX_SIZE" "onnx/paradee_int8.onnx"
fetch "${BASE}/config.json"           "${DEST}/config.json"        "$CFG_SHA"  "$CFG_SIZE"  "config.json"

rmdir "$DEST/model" 2>/dev/null || true
echo "[models] baked into $DEST"
