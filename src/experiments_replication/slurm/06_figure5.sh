#!/bin/bash
#SBATCH --job-name=repl_06_figure5
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/06_figure5.out
#SBATCH --error=logs/06_figure5.err

# 2x A4000 (32 GB total): Figure 5 runs 4 conditions x 28 layers x N_TEST samples.

BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

source $BASE/.env
export HF_TOKEN
export PYTHONPATH=$BASE/src:$PYTHONPATH

mkdir -p $EXPDIR/logs $EXPDIR/results

cd $EXPDIR
conda run -n llm_separate --no-capture-output \
    python 06_figure5_reply_inversion.py
