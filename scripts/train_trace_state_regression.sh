#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --stages trace_state_head \
  --trace-ops-head-steps "${STEPS:-1200}" \
  --train-size "${TRAIN_SIZE:-12000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed_only}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed_only}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.001}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/trace_state_regression}"
