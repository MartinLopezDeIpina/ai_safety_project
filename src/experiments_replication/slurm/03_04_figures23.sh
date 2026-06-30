#!/bin/bash
#SBATCH --job-name=repl_0304_figures23
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/03_04_figures23.out
#SBATCH --error=logs/03_04_figures23.err

# Figures 2 and 3 are pure analysis (no model inference — reads saved .pt tensors).

BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

source $BASE/.env
export HF_TOKEN
export PYTHONPATH=$BASE/src:$PYTHONPATH

mkdir -p $EXPDIR/logs $EXPDIR/results

cd $EXPDIR
conda run -n llm_separate --no-capture-output \
    python 03_figure2_clustering.py

conda run -n llm_separate --no-capture-output \
    python 04_figure3_scatter.py
