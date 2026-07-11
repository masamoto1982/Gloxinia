#!/usr/bin/env bash
# Build the Gloxinia WASM module and emit JS/TS bindings into ../web/src/wasm.
# Used identically by local dev and CI, so the artifact is reproducible.
set -euo pipefail

cd "$(dirname "$0")"
OUT="../web/src/wasm"

echo "==> cargo build (wasm32-unknown-unknown, release)"
cargo build --target wasm32-unknown-unknown --release

echo "==> wasm-bindgen -> $OUT"
wasm-bindgen \
  target/wasm32-unknown-unknown/release/gloxinia.wasm \
  --out-dir "$OUT" \
  --target web

echo "==> done"
