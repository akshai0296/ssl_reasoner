#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --stages standalone_transition \
  --checkpoint "${CHECKPOINT:-checkpoints/variable_reasoner/best.pt}" \
  --variable-reasoner-steps "${STEPS:-3000}" \
  --train-size "${TRAIN_SIZE:-12000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-single_op_balanced}" \
  --val-curriculum "${VAL_CURRICULUM:-single_op_balanced}" \
  --max-math-len "${MAX_MATH_LEN:-34}" \
  --max-variable-steps "${MAX_VARIABLE_STEPS:-16}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.001}" \
  --use-math-features \
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/standalone_transition_single}"
