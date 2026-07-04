#!/bin/bash
#SBATCH --job-name=repl_00_03_pipeline
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/00_03_pipeline_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/00_03_pipeline_%j.err

# Runs the Figure-2 pipeline end to end in a single job: 00 -> 01 -> 02 -> 03.
#   00  collect behaviors (model inference on all datasets; the slow stage)
#   01  extract hidden states + cluster centers + directions (forward passes)
#   02  Table 1 (post-instruction-token ablation; model inference)
#   03  Figure 2 clustering (pure analysis of saved .pt tensors, no GPU)
# Stages run sequentially and share the GPU; `set -e` aborts on the first failure.

set -euo pipefail

REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
EXPDIR=$REPO/src/experiments_replication
VENV=$REPO/venv

# HF_HOME / HF_HUB_OFFLINE from the project .env (compute nodes have no internet).
set -a
source "$REPO/.env"
set +a

source "$VENV/bin/activate"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

mkdir -p "$EXPDIR/logs" "$EXPDIR/results"
cd "$EXPDIR"

echo "===== [00] collect_behaviors  $(date) ====="
python 00_collect_behaviors.py

echo "===== [01] extract_directions  $(date) ====="
python 01_extract_directions.py

echo "===== [02] table1_post_inst_tokens  $(date) ====="
python 02_table1_post_inst_tokens.py

echo "===== [03] figure2_clustering  $(date) ====="
python 03_figure2_clustering.py

echo "===== pipeline complete  $(date) ====="
