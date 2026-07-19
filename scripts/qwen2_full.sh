#!/usr/bin/env bash
# Qwen2 replication (instruct track) — FULL RUN: infer, eval, acts, gen_buckets, fig, fig3.
#   ./scripts/qwen2_full.sh           launch detached
#   ./scripts/qwen2_full.sh collect   pull results
# NB: the GPU stages for qwen7b are already committed; only re-run to regenerate from scratch.
#     Model/size alternatives: llama3:8b (Llama-3), llama:7b (Llama-2).
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"
PHASE="${1:-launch}"

RUNS="qwen:7b:0:100000"                       # 100000 == "all rows"
CONFIG="configs/bucketing/bucket_config_clean.json"
STAGES="infer,eval,acts,gen_buckets,fig,fig3"

COMMON=(--runs "$RUNS" --stages "$STAGES" --config "$CONFIG")

if [ "$PHASE" = "collect" ]; then
    $MODAL run src/experiments_replication/modal_run.py "${COMMON[@]}" --collect-only
else
    $MODAL run --detach src/experiments_replication/modal_run.py "${COMMON[@]}" \
        --gpu A100 --timeout 18000 --no-wait
    echo "Launched detached. When it finishes, collect with:  ./scripts/qwen2_full.sh collect"
fi
