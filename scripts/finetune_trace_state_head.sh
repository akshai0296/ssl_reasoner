#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --stages trace_ops_head \
  --trace-ops-head-steps "${STEPS:-1000}" \
  --train-size "${TRAIN_SIZE:-10000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed_only}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed_only}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.0005}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --eval-every "${EVAL_EVERY:-250}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/trace_state_head_mixed}"
