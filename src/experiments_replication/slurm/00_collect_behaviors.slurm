#!/bin/bash
#SBATCH --job-name=repl_00_behaviors
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/00_collect_behaviors_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/00_collect_behaviors_%j.err

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

python 00_collect_behaviors.py
