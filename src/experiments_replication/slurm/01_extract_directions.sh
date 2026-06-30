#!/bin/bash
#SBATCH --job-name=repl_01_directions
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/01_extract_directions.out
#SBATCH --error=logs/01_extract_directions.err

BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

source $BASE/.env
export HF_TOKEN
export PYTHONPATH=$BASE/src:$PYTHONPATH

mkdir -p $EXPDIR/logs $EXPDIR/results

cd $EXPDIR
conda run -n llm_separate --no-capture-output \
    python 01_extract_directions.py
