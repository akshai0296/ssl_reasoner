#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/predictor_reasoning_stage1_mixed_anticollapse/best.pt}" \
  --stages trace_ops_head \
  --trace-ops-head-steps "${STEPS:-500}" \
  --train-size "${TRAIN_SIZE:-6000}" \
  --val-size "${VAL_SIZE:-500}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed_only}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed_only}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.001}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --eval-every "${EVAL_EVERY:-100}" \
  --sample-count 5 \
  --output-dir "${OUTPUT_DIR:-checkpoints/trace_ops_head_mixed}"
