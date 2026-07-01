#!/bin/bash
# ============================================================================
# ONE-TIME SETUP — run this on the LOGIN NODE, not in a job.
#   bash setup_venv.sh
#
# KISSKI compute nodes have NO outbound internet, so the venv must be built
# and all pip installs done here on the login node (which does have internet).
# ============================================================================
set -euo pipefail

REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
VENV=$REPO/venv

echo "=== Creating personal venv at $VENV ==="
# Use the same Python the cluster provides. python3.11 matches the working
# course venv (Python 3.11.13). Adjust if your repo needs a different version.
python3 -m venv "$VENV"

source "$VENV/bin/activate"
echo "venv python: $(which python)"
python --version

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing requirements ==="
# requirements.txt lives at the repo root.
pip install -r "$REPO/requirements.txt"

echo ""
echo "=== Verifying key packages ==="
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda build:", torch.version.cuda)
PY

echo ""
echo "=== Setup complete. venv is at: $VENV ==="
echo "Next: pre-download models on the login node (see download note in README),"
echo "then submit smoke_test.slurm."
