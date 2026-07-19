#!/usr/bin/env bash
# Thinking track (Qwen3.5) — SMOKE TEST: 1 data point per dataset, infer+eval only, blocking.
# The row slice is the `left:right` part of --runs, so 0:1 == one example per dataset.
# Only infer+eval: at 1 row/dataset the buckets are too small for acts/fig (some families empty).
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"

$MODAL run src/experiments_replication/modal_run.py \
    --runs "qwen35:9b:0:1" \
    --thinking \
    --stages "infer,eval" \
    --config configs/bucketing/bucket_config_qwen35_think.json \
    --sampling-config configs/sampling/sampling_config_qwen3.5-9b.json \
    --batch-size 1 --max-len 512 \
    --gpu A100 --timeout 3600
