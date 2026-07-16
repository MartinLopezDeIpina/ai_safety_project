#!/usr/bin/env bash
#SBATCH --job-name=section4_jailbreak
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --exclude=tcml-node26
#SBATCH --nodelist=tcml-node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/section4_jailbreak-%j.out
#SBATCH --error=logs/slurm/section4_jailbreak-%j.err

set -e
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/stud127/ai_safety_project}"
EXP_DIR="${PROJECT_DIR}/src/experiments_replication"
SRC_DIR="${PROJECT_DIR}/src"

cd "${EXP_DIR}"
mkdir -p logs/slurm output

source /mnt/beegfs/home/stud127/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-ai_safety}"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
LEFT="${LEFT:-0}"
RIGHT="${RIGHT:-50}"
BASELINE_LEFT="${BASELINE_LEFT:-0}"
BASELINE_RIGHT="${BASELINE_RIGHT:-100}"
BUCKET_CONFIG="${BUCKET_CONFIG:-bucket_config_clean.json}"
MAX_LEN="${MAX_LEN:-128}"
FORCE_INFER="${FORCE_INFER:-0}"

# Set RUN_BASELINE=1 if datasets_outputs/activations do not already exist for this model.
RUN_BASELINE="${RUN_BASELINE:-0}"

# default     = template jailbreak + evil persona
# all         = template jailbreak + evil persona + transferred GCG suffix + persuasion
# qwen_gcg    = Qwen2-trained GCG suffix only (after running train_qwen2_gcg_sbatch.sh)
# all_trained = template jailbreak + evil persona + transferred GCG + Qwen2-trained GCG + persuasion
# template / evil are also supported for single-family jobs.
JAILBREAK_SET="${JAILBREAK_SET:-default}"

echo "========================================="
echo "Starting Section 4 jailbreak analysis"
echo "Project: ${PROJECT_DIR}"
echo "Model: ${MODEL} ${MODEL_SIZE}"
echo "Jailbreak sample range: [${LEFT}, ${RIGHT})"
echo "Baseline sample range: [${BASELINE_LEFT}, ${BASELINE_RIGHT})"
echo "Bucket config: ${BUCKET_CONFIG}"
echo "Max generation length: ${MAX_LEN}"
echo "Run baseline first: ${RUN_BASELINE}"
echo "Jailbreak set: ${JAILBREAK_SET}"
echo "========================================="

baseline_dir="${EXP_DIR}/output/${MODEL}${MODEL_SIZE}/datasets_outputs/activations"
required_baseline=(
  "advbench_gentinst_accepted.pt"
  "jbb_gentinst_accepted.pt"
  "sorrybench_gentpost_accepted.pt"
  "advbench_gentinst_refused.pt"
  "jbb_gentinst_refused.pt"
  "advbench_gentpost_refused.pt"
  "jbb_gentpost_refused.pt"
  "alpaca_accepted.pt"
  "xstest_refused.pt"
)

missing_baseline=0
for file_name in "${required_baseline[@]}"; do
  if [ ! -f "${baseline_dir}/${file_name}" ]; then
    missing_baseline=1
    echo "Missing baseline activation: ${baseline_dir}/${file_name}"
  fi
done

if [ "${missing_baseline}" = "1" ]; then
  if [ "${RUN_BASELINE}" = "1" ]; then
    echo ""
    echo "STEP 0: baseline activations missing, running infer/eval/acts first"
    python -u -c "import main; main.main('${MODEL}', '${MODEL_SIZE}', left=${BASELINE_LEFT}, right=${BASELINE_RIGHT}, stages=('infer', 'eval', 'acts'), bucket_config='${BUCKET_CONFIG}')"
  else
    echo ""
    echo "Baseline activations are missing."
    echo "Resubmit with RUN_BASELINE=1, or run the Figure 2/3 activation pipeline first."
    exit 1
  fi
fi

jailbreak_args=()
case "${JAILBREAK_SET}" in
  default)
    ;;
  all)
    jailbreak_args=(--jailbreaks all)
    ;;
  qwen_gcg)
    jailbreak_args=(--jailbreaks "qwen2 gcg suffix")
    ;;
  all_trained)
    jailbreak_args=(--jailbreaks "template jailbreak" "evil persona" "gcg suffix" "qwen2 gcg suffix" "persuasion")
    ;;
  template)
    jailbreak_args=(--jailbreaks "template jailbreak")
    ;;
  evil)
    jailbreak_args=(--jailbreaks "evil persona")
    ;;
  *)
    echo "Unknown JAILBREAK_SET='${JAILBREAK_SET}'. Use default, all, template, or evil."
    exit 1
    ;;
esac

echo ""
echo "STEP 1: run jailbreak inference"
force_args=()
if [ "${FORCE_INFER}" = "1" ]; then
  force_args=(--force)
fi
python -u _04_4_jailbreak_plot.py "${MODEL}" "${MODEL_SIZE}" \
  --stages infer \
  --left "${LEFT}" \
  --right "${RIGHT}" \
  --max_len "${MAX_LEN}" \
  --bucket_config "${BUCKET_CONFIG}" \
  "${force_args[@]}" \
  "${jailbreak_args[@]}"

echo ""
echo "STEP 2: extract jailbreak activations"
python -u _04_4_jailbreak_plot.py "${MODEL}" "${MODEL_SIZE}" \
  --stages acts \
  --left "${LEFT}" \
  --right "${RIGHT}" \
  --bucket_config "${BUCKET_CONFIG}" \
  "${jailbreak_args[@]}"

echo ""
echo "STEP 3: plot Section 4 jailbreak scatter"
python -u _04_4_jailbreak_plot.py "${MODEL}" "${MODEL_SIZE}" \
  --stages plot \
  --left "${LEFT}" \
  --right "${RIGHT}" \
  --bucket_config "${BUCKET_CONFIG}" \
  "${jailbreak_args[@]}"

echo ""
echo "STEP 4: evaluate harmful compliance"
mapfile -t generation_files < <(find "${EXP_DIR}/output/${MODEL}${MODEL_SIZE}/section4_jailbreak/generations" -maxdepth 1 -name "*.json" | sort)
python -u _04_harmful_compliance_eval.py \
  --input "${generation_files[@]}" \
  --out_dir "${EXP_DIR}/output/${MODEL}${MODEL_SIZE}/section4_jailbreak/harmful_compliance_eval"

echo ""
echo "========================================="
echo "Section 4 jailbreak analysis completed"
echo "Figure:"
echo "  ${EXP_DIR}/output/${MODEL}${MODEL_SIZE}/section4_jailbreak/section4_jailbreak_scatter.png"
echo "Harmful compliance summary:"
echo "  ${EXP_DIR}/output/${MODEL}${MODEL_SIZE}/section4_jailbreak/harmful_compliance_eval/all_harmful_compliance_summary.json"
echo "========================================="
