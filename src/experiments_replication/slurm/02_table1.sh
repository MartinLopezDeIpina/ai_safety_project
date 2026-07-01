#!/bin/bash
#SBATCH --job-name=repl_02_table1
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/02_table1_%j.out
#SBATCH --error=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/02_table1_%j.err

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

python 02_table1_post_inst_tokens.py
