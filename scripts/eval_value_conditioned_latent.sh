#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/value_conditioned_joint_wide/best.pt}"
SAMPLES="${SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CURRICULUM="${CURRICULUM:-mixed}"

for mode in value_conditioned value_conditioned_latent_nn step_state_solver pred latent_nn; do
  echo "=== ${mode} ==="
  PYTHONPATH=src python -m ssl_reasoner.eval \
    --checkpoint "${CHECKPOINT}" \
    --mode "${mode}" \
    --samples "${SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --curriculum "${CURRICULUM}"
done
