#!/usr/bin/env bash
# Interventions (Fig 4 / Fig 5) — SMOKE TEST: fig5a only, coeff 2, 8 rows.
# fig5a is the cheapest decisive probe: CPU-scored (no LLM judge), ~6 short runs.
#   ./scripts/intervention_smoke.sh          build vectors + launch
#   ./scripts/intervention_smoke.sh collect  pull + score + plot
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
PY="${PY:-python}"
PHASE="${1:-launch}"

ARGS=(--hf-slot 1 --refusal-slot 20 \
      --thinking-mode gennothink \
      --bucket-config configs/bucketing/bucket_config_qwen35_nothink_intervene_gcg.json \
      --figures fig5a --coeffs 2 --right 8 --gpu A100-80GB)

if [ "$PHASE" = "collect" ]; then
    $PY src/experiments_replication/launch_intervention.py "${ARGS[@]}" --mode collect --skip-vectors
else
    $PY src/experiments_replication/launch_intervention.py "${ARGS[@]}"
    echo "Launched. When generation finishes (~minutes):  ./scripts/intervention_smoke.sh collect"
fi
