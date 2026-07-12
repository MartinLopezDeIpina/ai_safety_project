#!/usr/bin/env bash
#SBATCH --job-name=asc2026-main
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$SLURM_SUBMIT_DIR/src/experiments_replication"
cd "$SCRIPT_DIR"

python main.py