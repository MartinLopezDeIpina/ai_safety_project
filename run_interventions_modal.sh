#!/usr/bin/env bash
# Intervention experiments on Modal — steering variants on advbench & alpaca (Qwen 7B, first 500 examples).
#
#   advbench:
#     1. hf-reverse    — reverse the harmfulness vector  (less-harm steering)
#     2. refusal-rev   — reverse the refusal vector      (less-refusal steering)
#     3. refusal-fwd   — refusal vector, not reversed    (more-refusal steering)
#   alpaca:
#     4. hf-fwd        — harmfulness vector, not reversed (more-harm steering)
#     5. refusal-fwd   — refusal vector, not reversed     (more-refusal steering)
#     6. refusal-rev   — reverse the refusal vector       (less-refusal steering)
#
# Each runs detached; collect results afterwards with the --collect-only commands at the bottom.
#
# Usage:  bash run_interventions_modal.sh

set -euo pipefail

SCRIPT="src/experiments_replication/modal_intervene.py"
RUNS="qwen:7b:0:500"
# Sleep between modal run calls to avoid "App creation failed: rate limit exceeded".
LAUNCH_DELAY=5

echo "=== 1. Reverse harmfulness direction on advbench (less-harm) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "hf" --reverse-intervention 1 \
    --intervene-context-only 1 --intervene-all 0 --no-wait
echo "Sleeping to avoid exceeding rate limit..."
sleep "$LAUNCH_DELAY"

echo "=== 2. Reverse refusal direction on advbench (less-refusal) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "refusal" --reverse-intervention 1 \
    --intervene-context-only 0 --intervene-all 1 --no-wait
echo "Sleeping to avoid exceeding rate limit..."
sleep "$LAUNCH_DELAY"

echo "=== 3. Refusal direction on advbench (more-refusal) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "advbench" --vectors "refusal" --reverse-intervention 0 \
    --intervene-context-only 0 --intervene-all 1 --no-wait
echo "Sleeping to avoid exceeding rate limit..."
sleep "$LAUNCH_DELAY"

echo "=== 4. Harmfulness direction on alpaca (more-harm) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "hf" --reverse-intervention 0 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 1 --intervene-all 0 --no-wait
echo "Sleeping to avoid exceeding rate limit..."
sleep "$LAUNCH_DELAY"

echo "=== 5. Refusal direction on alpaca (more-refusal) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "refusal" --reverse-intervention 0 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 0 --intervene-all 1 --no-wait
echo "Sleeping to avoid exceeding rate limit..."
sleep "$LAUNCH_DELAY"

echo "=== 6. Reverse refusal direction on alpaca (less-refusal) ==="
modal run --detach "$SCRIPT" --runs "$RUNS" \
    --datasets "alpaca_data_instruction" --vectors "refusal" --reverse-intervention 1 \
    --arg-key-prompt "instruction" \
    --intervene-context-only 0 --intervene-all 1 --no-wait

echo ""
echo "All six runs launched detached."
echo "Collect results later with:  bash collect_interventions_modal.sh"
