#!/usr/bin/env bash
# Qwen2 replication (instruct track) — SMOKE TEST: 1 data point per dataset, infer+eval, blocking.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"

$MODAL run src/experiments_replication/modal_run.py \
    --runs "qwen:7b:0:1" \
    --stages "infer,eval" \
    --config configs/bucketing/bucket_config_clean.json \
    --gpu A100 --timeout 3600
