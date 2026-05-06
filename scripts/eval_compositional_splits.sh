#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-checkpoints/predictor_structured_answer/best.pt}"
SAMPLES="${SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODE="${MODE:-pred}"

for curriculum in seen_single unseen_single seen_mixed unseen_mixed; do
  echo "=== ${curriculum} (${MODE}) ==="
  PYTHONPATH=src python -m ssl_reasoner.eval \
    --checkpoint "${CHECKPOINT}" \
    --samples "${SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --mode "${MODE}" \
    --curriculum "${curriculum}"
done
