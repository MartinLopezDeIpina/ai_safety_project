#!/bin/bash
#SBATCH --job-name=repl_05_figure4
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/05_figure4_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/05_figure4_%j.err

# 7B model + per-layer activation-addition. One A100 (80GB) has ample
# headroom - the original 2x A4000 was only to reach enough total VRAM.
REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
EXPDIR=$REPO/src/experiments_replication
VENV=$REPO/venv

# HF_HOME / HF_HUB_OFFLINE from the project .env (compute nodes have no internet).
set -a
source "$PROJECT/.env"
set +a

source "$VENV/bin/activate"

export PYTHONPATH="$REPO/src:$PYTHONPATH"

mkdir -p "$EXPDIR/logs" "$EXPDIR/results"
cd "$EXPDIR"

python 05_figure4_steering_layers.py
