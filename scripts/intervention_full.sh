#!/usr/bin/env bash
# Interventions (Fig 4 + Fig 5a + Fig 5b) — FULL RUN, ~31 detached runs, coeff ladder 2,4.
#   ./scripts/intervention_full.sh           build vectors + launch every run detached
#   ./scripts/intervention_full.sh collect   pull + score Fig5 + LAUNCH Fig4 judges  (run twice:
#                                            first pass scores Fig5 & fires the judges,
#                                            second pass -- after ~40 min -- pulls the Fig4 labels)
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
PY="${PY:-python}"
PHASE="${1:-launch}"

# Token pair + data. Change --hf-slot/--refusal-slot for a different pair; point --bucket-config
# and --harmful-dataset at your GCG set. genthink needs --thinking-mode genthink (token caps auto).
ARGS=(--hf-slot 1 --refusal-slot 20 \
      --thinking-mode gennothink \
      --bucket-config configs/bucketing/bucket_config_qwen35_nothink_intervene_gcg.json \
      --coeffs 2,4 --right 50 --gpu A100-80GB)

if [ "$PHASE" = "collect" ]; then
    $PY src/experiments_replication/launch_intervention.py "${ARGS[@]}" --mode collect --skip-vectors
else
    $PY src/experiments_replication/launch_intervention.py "${ARGS[@]}"
    echo "Launched detached (~31 runs, Modal queues past 10 GPUs)."
    echo "After generation drains (~35 min):  ./scripts/intervention_full.sh collect"
    echo "Then again after the judges finish (~40 min):  ./scripts/intervention_full.sh collect"
fi
