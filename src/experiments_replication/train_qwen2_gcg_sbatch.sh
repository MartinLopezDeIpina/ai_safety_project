#!/usr/bin/env bash
#SBATCH --job-name=train_qwen2_gcg
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --exclude=tcml-node26
#SBATCH --nodelist=tcml-node18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_qwen2_gcg-%j.out
#SBATCH --error=logs/slurm/train_qwen2_gcg-%j.err

set -e
set -o pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/stud127/ai_safety_project}"
EXP_DIR="${PROJECT_DIR}/src/experiments_replication"

cd "${EXP_DIR}"
mkdir -p logs/slurm "${PROJECT_DIR}/data"

source /mnt/beegfs/home/stud127/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-ai_safety}"

if [ "${INSTALL_NANOGCG:-0}" = "1" ]; then
  python -m pip install nanogcg
fi

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("nanogcg") else 1)
PY
then
  echo "nanogcg is not installed in this environment."
  echo "Resubmit with INSTALL_NANOGCG=1, or install once with: pip install nanogcg"
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2-7B-Instruct}"
GCG_INPUT="${GCG_INPUT:-${PROJECT_DIR}/data/advbench.json}"
GCG_OUTPUT="${GCG_OUTPUT:-${PROJECT_DIR}/data/qwen2-gcg-advsuffix.json}"
GCG_LEFT="${GCG_LEFT:-0}"
GCG_RIGHT="${GCG_RIGHT:-20}"
GCG_STEPS="${GCG_STEPS:-80}"
GCG_SEARCH_WIDTH="${GCG_SEARCH_WIDTH:-32}"
GCG_BATCH_SIZE="${GCG_BATCH_SIZE:-8}"
GCG_TOPK="${GCG_TOPK:-32}"
GCG_BUFFER_SIZE="${GCG_BUFFER_SIZE:-4}"
GCG_SEED="${GCG_SEED:-42}"
GCG_FILTER_IDS="${GCG_FILTER_IDS:-0}"
GCG_OPTIM_STR_INIT="${GCG_OPTIM_STR_INIT:-! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !}"

echo "========================================="
echo "Training Qwen2 nanoGCG suffixes"
echo "Model: ${MODEL_ID}"
echo "Input: ${GCG_INPUT}"
echo "Output: ${GCG_OUTPUT}"
echo "Rows: [${GCG_LEFT}, ${GCG_RIGHT})"
echo "Steps/search_width/topk: ${GCG_STEPS}/${GCG_SEARCH_WIDTH}/${GCG_TOPK}"
echo "Filter ids: ${GCG_FILTER_IDS}"
echo "========================================="

python -u _04_train_qwen2_gcg.py \
  --model_id "${MODEL_ID}" \
  --input "${GCG_INPUT}" \
  --output "${GCG_OUTPUT}" \
  --left "${GCG_LEFT}" \
  --right "${GCG_RIGHT}" \
  --num_steps "${GCG_STEPS}" \
  --search_width "${GCG_SEARCH_WIDTH}" \
  --batch_size "${GCG_BATCH_SIZE}" \
  --topk "${GCG_TOPK}" \
  --buffer_size "${GCG_BUFFER_SIZE}" \
  --seed "${GCG_SEED}" \
  --filter_ids "${GCG_FILTER_IDS}" \
  --optim_str_init "${GCG_OPTIM_STR_INIT}"

echo ""
echo "Done. Next run:"
echo "  JAILBREAK_SET=qwen_gcg sbatch section4_jailbreak_sbatch.sh"
echo "or:"
echo "  JAILBREAK_SET=all_trained sbatch section4_jailbreak_sbatch.sh"
