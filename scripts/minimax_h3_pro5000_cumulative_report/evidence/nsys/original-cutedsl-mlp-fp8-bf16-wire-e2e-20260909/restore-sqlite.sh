#!/usr/bin/env bash
set -euo pipefail

archive="minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite.zst"
database="minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite"
expected="53e017aef441ea5e8e5fed39030d55faa92072f7ef653b62cc84e4883d09f144"

zstd -d -f "$archive" -o "$database"
printf '%s  %s\n' "$expected" "$database" | sha256sum -c -

