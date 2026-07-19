#!/usr/bin/env bash
# Thinking track (Qwen3.5) — FULL RUN: infer, eval, acts, gen_buckets, fig.
#   ./scripts/thinking_full.sh           launch detached (survives closing the laptop)
#   ./scripts/thinking_full.sh collect   pull results once the run has finished
# NB: fig3 is instruct-only; _main_thinking plots figure 2 only, so it is omitted here.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"
PHASE="${1:-launch}"

RUNS="qwen35:9b:0:512"
CONFIG="configs/bucketing/bucket_config_qwen35_think.json"
STAGES="infer,eval,acts,gen_buckets,fig"

COMMON=(--runs "$RUNS" --thinking --stages "$STAGES" --config "$CONFIG" \
        --sampling-config configs/sampling/sampling_config_qwen3.5-9b.json \
        --use-judge --judge-config configs/eval/judge_config_thinking.json)

if [ "$PHASE" = "collect" ]; then
    $MODAL run src/experiments_replication/modal_run.py "${COMMON[@]}" --collect-only
else
    $MODAL run --detach src/experiments_replication/modal_run.py "${COMMON[@]}" \
        --batch-size 8 --max-len 4096 --gpu B300 --timeout 18000 --no-wait
    echo "Launched detached. When it finishes, collect with:  ./scripts/thinking_full.sh collect"
fi
