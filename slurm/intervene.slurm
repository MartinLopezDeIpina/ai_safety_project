#!/usr/bin/env bash
#SBATCH --job-name=asc2026-intervene
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen7b-advbench-inv-lessharm_%j.out
#SBATCH --error=logs/qwen7b-advbench-inv-lessharm_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

# SCRIPT_DIR="$SLURM_SUBMIT_DIR/src/experiments_replication"
# cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"

OUTPUT_DIR="src/experiments_replication/qwen7b/inversion_outputs"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

python -u "$SLURM_SUBMIT_DIR/src/intervention.py" \
	--test_data_pth "data/advbench.json" \
	--output_pth "$OUTPUT_DIR/advbench_lessharm.json" \
	--intervention_vector "src/experiments_replication/steering_vectors/qwen-7b/hf.pt" \
	--reverse_intervention 1 \
	--intervene_context_only 1 \
	--arg_key_prompt bad_q \
	--model qwen \
	--model_size 7b \
	--left 0 \
	--right 50 \
	--layer_s 0 \
	--layer_e 28 \
	--coeff_select 1 \
	--max_token_generate 100 \
	--max_decode_step_while_intervene 1 \
	--use_inversion 1 \
	--inversion_prompt_idx 1
