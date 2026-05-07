#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --stages answer_value_head \
  --answer-value-head-steps "${STEPS:-800}" \
  --train-size "${TRAIN_SIZE:-8000}" \
  --val-size "${VAL_SIZE:-500}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed}" \
  --train-easy-ratio "${TRAIN_EASY_RATIO:-0.35}" \
  --val-easy-ratio "${VAL_EASY_RATIO:-0.75}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.001}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --eval-every "${EVAL_EVERY:-200}" \
  --sample-count 5 \
  --output-dir "${OUTPUT_DIR:-checkpoints/answer_value_head}"
