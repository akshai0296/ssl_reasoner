#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --stages standalone_transition \
  --checkpoint "${CHECKPOINT:-checkpoints/standalone_transition_reductions/best.pt}" \
  --variable-reasoner-steps "${STEPS:-5000}" \
  --train-size "${TRAIN_SIZE:-60000}" \
  --val-size "${VAL_SIZE:-3000}" \
  --train-curriculum "${TRAIN_CURRICULUM:-multi_step_balanced}" \
  --val-curriculum "${VAL_CURRICULUM:-multi_step_balanced}" \
  --max-math-len "${MAX_MATH_LEN:-34}" \
  --max-variable-steps "${MAX_VARIABLE_STEPS:-16}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.0005}" \
  --use-math-features \
  --eval-every "${EVAL_EVERY:-500}" \
  --sample-count "${SAMPLE_COUNT:-3}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/standalone_transition_reductions}"
