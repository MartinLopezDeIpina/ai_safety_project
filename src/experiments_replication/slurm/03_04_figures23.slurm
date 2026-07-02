#!/bin/bash
#SBATCH --job-name=repl_0304_figures23
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/03_04_figures23_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/03_04_figures23_%j.err

# Figures 2 and 3 are pure analysis (no model inference - reads saved .pt tensors).
REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
EXPDIR=$REPO/src/experiments_replication
VENV=$REPO/venv

# HF_HOME / HF_HUB_OFFLINE from the project .env (compute nodes have no internet).
set -a
source "$REPO/.env"
set +a

source "$VENV/bin/activate"

export PYTHONPATH="$REPO/src:$PYTHONPATH"

mkdir -p "$EXPDIR/logs" "$EXPDIR/results"
cd "$EXPDIR"

python 03_figure2_clustering.py

python 04_figure3_scatter.py
