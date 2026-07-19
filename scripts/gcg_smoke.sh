#!/usr/bin/env bash
# GCG attack — SMOKE TEST: 2 behaviors, 1 target, blocking. Proves the attack path works end-to-end.
# 25 optimization steps (vs the 250 of a real attack) so it finishes in minutes; a suffix that
# actually jailbreaks usually needs the full run, so don't read success/failure into this.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"

$MODAL run src/experiments_replication/modal_gcg.py::entry \
    --mode gennothink --n-behaviors 2 --target-count 1 \
    --num-steps 25 --search-width 128 --early-stop \
    --gpu B300 --timeout 5400
