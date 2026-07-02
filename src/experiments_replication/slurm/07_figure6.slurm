#!/bin/bash
#SBATCH --job-name=repl_07_figure6
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/07_figure6_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/07_figure6_%j.err

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

python 07_figure6_jailbreak.py
