#!/bin/bash
# Compute harmfulness and refusal intervention vectors for Qwen 7B
# and save them to steering_vectors/qwen-7b/.

# Run the script from the root folder!

set -e

ACT_DIR="src/experiments_replication/activations_qwen"  # change this for your setup
OUT_DIR="src/experiments_replication/steering_vectors/qwen7b"

python "src/experiments_replication/get_intervene_vectors.py" \
  --act-dir "${ACT_DIR}" \
  --out-dir "${OUT_DIR}"
