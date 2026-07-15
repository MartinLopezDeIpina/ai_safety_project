#!/usr/bin/env bash
#
# Run the 5 experiments that haven't been collected yet.
# Each command is standalone — run them one at a time.
#
set -euo pipefail

echo "=== Launching missing experiments (detached) ==="
echo

modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 1 \
    --intervene-all 0 \
    --use-inversion 1 \
    --no-wait
sleep 10

modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --no-wait
sleep 10

modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors refusal \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --no-wait
sleep 10

modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:100 \
    --datasets alpaca_data_instruction \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --no-wait
# sleep 10

modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:100 \
    --datasets alpaca_data_instruction \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --no-wait

echo
echo "=== All launched detached ==="
echo "Wait for them to finish, then collect with collect_experiments.sh"
