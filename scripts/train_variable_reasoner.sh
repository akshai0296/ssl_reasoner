#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --stages variable_reasoner \
  --variable-reasoner-steps "${STEPS:-2000}" \
  --train-size "${TRAIN_SIZE:-12000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-multi_step_balanced}" \
  --val-curriculum "${VAL_CURRICULUM:-multi_step_balanced}" \
  --max-math-len "${MAX_MATH_LEN:-10}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.001}" \
  --use-math-features \
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/variable_reasoner}"
