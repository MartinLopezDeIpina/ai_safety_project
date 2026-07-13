#!/usr/bin/env bash
# Collect results from the detached intervention runs launched by run_interventions_modal.sh.
#
# Each --collect-only call pulls from the Modal 'intervention-results' volume into
# src/experiments_replication/intervention_outputs/, auto-indexing (_2, _3, …) so
# previous local outputs are never overwritten.
#
# Usage:  bash collect_interventions_modal.sh

set -euo pipefail

SCRIPT="src/experiments_replication/modal_intervene.py"
RUNS="qwen:7b:0:500"
# Sleep between modal run calls to avoid "App creation failed: rate limit exceeded".
COLLECT_DELAY=5

echo "=== Collecting 1. Reverse harmfulness direction on advbench (less-harm) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "hf" --reverse-intervention 1 \
    --intervene-context-only 1 --intervene-all 0 --collect-only
echo "Sleeping to avoid exceeding rate limit..."
sleep "$COLLECT_DELAY"

echo "=== Collecting 2. Reverse refusal direction on advbench (less-refusal) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "refusal" --reverse-intervention 1 \
    --intervene-context-only 0 --intervene-all 1 --collect-only
echo "Sleeping to avoid exceeding rate limit..."
sleep "$COLLECT_DELAY"

echo "=== Collecting 3. Refusal direction on advbench (more-refusal) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "refusal" --reverse-intervention 0 \
    --intervene-context-only 0 --intervene-all 1 --collect-only
echo "Sleeping to avoid exceeding rate limit..."
sleep "$COLLECT_DELAY"

echo "=== Collecting 4. Harmfulness direction on alpaca (more-harm) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "hf" --reverse-intervention 0 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 1 --intervene-all 0 --collect-only
echo "Sleeping to avoid exceeding rate limit..."
sleep "$COLLECT_DELAY"

echo "=== Collecting 5. Refusal direction on alpaca (more-refusal) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "refusal" --reverse-intervention 0 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 0 --intervene-all 1 --collect-only
echo "Sleeping to avoid exceeding rate limit..."
sleep "$COLLECT_DELAY"

echo "=== Collecting 6. Reverse refusal direction on alpaca (less-refusal) ==="
modal run "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "refusal" --reverse-intervention 1 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 0 --intervene-all 1 --collect-only

echo ""
echo "Done. Results are in src/experiments_replication/intervention_outputs/"
