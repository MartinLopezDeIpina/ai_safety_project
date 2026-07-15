#!/usr/bin/env bash
#
# Collect intervention experiments from Modal.
# Each command is standalone — run them one at a time, or comment out
# experiments you've already collected.
#
# Usage:
#   bash src/experiments_replication/collect_experiments.sh   # collect all 14
#   # or run individual lines below in any order
#
set -euo pipefail

echo "--- Using runs=qwen:7b:0:500 ---"
echo

# ── Harmful datasets ────────────────────────────────────────────

modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets advbench \
    --vectors hf \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 1 \
    --intervene-all 0 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets advbench \
    --vectors refusal \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets advbench \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets jbb \
    --vectors hf \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 1 \
    --intervene-all 0 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets jbb \
    --vectors refusal \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets jbb \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


# ── Harmless datasets ───────────────────────────────────────────

modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets alpaca_data_instruction \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 1 \
    --intervene-all 0 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets alpaca_data_instruction \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets alpaca_data_instruction \
    --vectors refusal \
    --reverse-intervention 1 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 1 \
    --intervene-all 0 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets xstest-harmless \
    --vectors refusal \
    --reverse-intervention 1 \
    --arg-key-prompt bad_q \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 1 \
    --collect-only


# ── No-inversion runs (alpaca, no inversion prompt) ────────────

modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets alpaca_data_instruction \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --collect-only


modal run src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:500 \
    --datasets alpaca_data_instruction \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --collect-only

echo
echo "=== All done ==="
