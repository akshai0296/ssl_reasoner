#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/state_conditioned_latent/best.pt}"
SAMPLES="${SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CURRICULUM="${CURRICULUM:-mixed}"

for mode in state_conditioned state_conditioned_latent_nn pred latent_nn step_state_solver; do
  echo "=== ${mode} ==="
  PYTHONPATH=src python -m ssl_reasoner.eval \
    --checkpoint "${CHECKPOINT}" \
    --mode "${mode}" \
    --samples "${SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --curriculum "${CURRICULUM}"
done
