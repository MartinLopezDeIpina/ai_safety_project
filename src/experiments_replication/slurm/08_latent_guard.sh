#!/bin/bash
#SBATCH --job-name=repl_08_latent_guard
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/08_latent_guard.out
#SBATCH --error=logs/08_latent_guard.err

# Pure analysis — reads saved .pt tensors and computes delta_harmful thresholds.
# Model inference only needed if jailbreak activations are not cached.

BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

source $BASE/.env
export HF_TOKEN
export PYTHONPATH=$BASE/src:$PYTHONPATH

mkdir -p $EXPDIR/logs $EXPDIR/results

cd $EXPDIR
conda run -n llm_separate --no-capture-output \
    python 08_latent_guard.py
