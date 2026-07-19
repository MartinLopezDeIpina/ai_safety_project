#!/usr/bin/env bash
# GCG attack — FULL RUN: 15 behaviors, target >=10 acceptances, detached.
#   ./scripts/gcg_full.sh           launch detached
#   ./scripts/gcg_full.sh collect   pull the gcg-results volume
# After collecting, feed the jsonl into the pipeline with qwen35_jailbreak/gcg_to_generations.py
# (see scripts.md §4), then run the `acts` stage to produce gcg_advbench_<mode>_{accepted,refused}.pt.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
MODAL="${MODAL:-modal}"
PHASE="${1:-launch}"

COMMON=(--mode gennothink --n-behaviors 15 --target-count 10 \
        --num-steps 250 --search-width 512 --topk 256 \
        --dataset advbench.json --seed 42)

if [ "$PHASE" = "collect" ]; then
    $MODAL run src/experiments_replication/modal_gcg.py "${COMMON[@]}" --collect-only
else
    $MODAL run --detach src/experiments_replication/modal_gcg.py "${COMMON[@]}" \
        --gpu B300 --timeout 18000 --no-wait
    echo "Launched detached. When it finishes, collect with:  ./scripts/gcg_full.sh collect"
fi
