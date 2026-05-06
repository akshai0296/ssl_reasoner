#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint checkpoints/stage3_joint_low_lr/best.pt \
  --verifier-checkpoint checkpoints/verifier_v1.pt \
  --mode verifier \
  --verifier-candidates 4 \
  --verifier-noise-scale 0.05 \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum mixed
