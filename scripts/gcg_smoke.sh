#!/usr/bin/env bash
# GCG attack — SMOKE TEST: 2 behaviors, 1 target, blocking. Proves the attack path works end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"

$MODAL run src/experiments_replication/modal_gcg.py \
    --mode gennothink --n-behaviors 2 --target-count 1 \
    --gpu B300 --timeout 5400
