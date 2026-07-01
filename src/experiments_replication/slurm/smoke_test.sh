#!/bin/bash
#SBATCH --job-name=repl_smoke
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/smoke_%j.out
#SBATCH --error=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately/src/experiments_replication/logs/smoke_%j.err

# Quick environment check for the beegfs/conda cluster.
# Verifies: conda activation works in a batch job, torch sees the GPU,
# transformers imports, HF_TOKEN is set, PYTHONPATH resolves the src package.
# No model download, no heavy compute — should finish in seconds once scheduled.

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

echo "=== which python ==="
which python

echo "=== python / torch / gpu ==="
python - <<'PY'
import sys
print("interpreter:", sys.executable)
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
import transformers
print("transformers:", transformers.__version__)
PY

echo "=== HF_TOKEN set? (length only, not printed) ==="
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN present, length ${#HF_TOKEN}"
else
    echo "WARNING: HF_TOKEN is empty"
fi

echo "=== smoke test complete ==="
