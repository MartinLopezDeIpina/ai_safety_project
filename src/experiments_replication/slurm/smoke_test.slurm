#!/bin/bash
#SBATCH --job-name=repl_smoke
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/smoke_%j.out
#SBATCH --error=/user/m.lopezdeipinamuno/u27880/ai_safety_project/src/experiments_replication/logs/smoke_%j.err

# Quick environment check for KISSKI + personal venv.
# Verifies: venv activates in a batch job, torch sees the A100, transformers
# imports, HF_TOKEN/HF_HOME resolve, PYTHONPATH finds the src package.
# No model download, no heavy compute.

REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
EXPDIR=$REPO/src/experiments_replication
VENV=$REPO/venv

# Load HF_HOME, HF_HUB_OFFLINE, etc. from the project .env (same one the course
# smoke test uses). $PROJECT is set by GWDG on both login and compute nodes.
set -a
source "$PROJECT/.env"
set +a

source "$VENV/bin/activate"

export PYTHONPATH="$REPO/src:$PYTHONPATH"

mkdir -p "$EXPDIR/logs" "$EXPDIR/results"
cd "$EXPDIR"

echo "=== which python ==="
which python

echo "=== python / torch / gpu / transformers ==="
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

echo "=== HF env ==="
echo "HF_HOME=${HF_HOME:-<unset>}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN present, length ${#HF_TOKEN}"
else
    echo "note: HF_TOKEN empty (fine if HF_HUB_OFFLINE=1 and models pre-downloaded)"
fi

echo "=== smoke test complete ==="
