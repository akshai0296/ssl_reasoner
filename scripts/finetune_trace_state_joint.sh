#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --stages 1 \
  --stage1-steps "${STAGE1_STEPS:-1200}" \
  --train-size "${TRAIN_SIZE:-12000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed_only}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed_only}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.00005}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --trace-weight "${TRACE_WEIGHT:-1.0}" \
  --trace-struct-weight "${TRACE_STRUCT_WEIGHT:-1.0}" \
  --reasoning-struct-weight "${REASONING_STRUCT_WEIGHT:-0.2}" \
  --structured-answer-weight "${STRUCTURED_ANSWER_WEIGHT:-0.5}" \
  --answer-value-weight "${ANSWER_VALUE_WEIGHT:-0.0}" \
  --answer-contrastive-weight "${ANSWER_CONTRASTIVE_WEIGHT:-0.0}" \
  --contrastive-weight "${CONTRASTIVE_WEIGHT:-0.2}" \
  --batch-diversity-weight "${BATCH_DIVERSITY_WEIGHT:-0.5}" \
  --slot-diversity-weight "${SLOT_DIVERSITY_WEIGHT:-0.1}" \
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/trace_state_joint}"
