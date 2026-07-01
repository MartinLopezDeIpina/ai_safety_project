#!/bin/bash
#SBATCH --job-name=repl_06_figure5
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/06_figure5_%j.out
#SBATCH --error=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/06_figure5_%j.err

# 2x A4000 (32 GB total): Figure 5 runs 4 conditions x 28 layers x N_TEST samples.
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

python 06_figure5_reply_inversion.py
