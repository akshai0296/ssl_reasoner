#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}"
SAMPLES="${SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CURRICULUM="${CURRICULUM:-mixed}"
MODES="${MODES:-target pred latent_nn}"

for mode in ${MODES}; do
  echo "=== ${mode} ==="
  PYTHONPATH=src python -m ssl_reasoner.eval \
    --checkpoint "${CHECKPOINT}" \
    --mode "${mode}" \
    --samples "${SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --curriculum "${CURRICULUM}"
done
