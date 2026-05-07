#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint "${CHECKPOINT:-checkpoints/predictor_reasoning_stage1_mixed_anticollapse/best.pt}" \
  --verifier-checkpoint "${VERIFIER:-checkpoints/verifier_reasoning_stage1_mixed_heavy.pt}" \
  --mode verifier \
  --verifier-candidates 8 \
  --verifier-noise-scale 0.05 \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum mixed
