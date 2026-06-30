#!/bin/bash
#SBATCH --job-name=repl_05_figure4
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/05_figure4.out
#SBATCH --error=logs/05_figure4.err

# 2x A4000 (32 GB total): 7B model uses ~14 GB; activation-addition
# memory spikes during per-layer steering require the extra headroom.

BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

source $BASE/.env
export HF_TOKEN
export PYTHONPATH=$BASE/src:$PYTHONPATH

mkdir -p $EXPDIR/logs $EXPDIR/results

cd $EXPDIR
conda run -n llm_separate --no-capture-output \
    python 05_figure4_steering_layers.py
