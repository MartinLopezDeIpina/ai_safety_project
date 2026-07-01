#!/bin/bash
#SBATCH --job-name=repl_05_figure4
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/05_figure4_%j.out
#SBATCH --error=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/05_figure4_%j.err

# 2x A4000 (32 GB total): 7B model uses ~14 GB; activation-addition
# memory spikes during per-layer steering require the extra headroom.
BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
EXPDIR=$BASE/src/experiments_replication

# conda install for this cluster (verified: conda info --base)
CONDA_ROOT=/mnt/beegfs/home/stud127/miniconda
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate llm_separate

source "$BASE/.env"
export HF_TOKEN
export PYTHONPATH="$BASE/src:$PYTHONPATH"

mkdir -p "$EXPDIR/logs" "$EXPDIR/results"
cd "$EXPDIR"

python 05_figure4_steering_layers.py
