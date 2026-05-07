#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --stages 1,2 \
  --stage1-steps "${STAGE1_STEPS:-1200}" \
  --stage2-steps "${STAGE2_STEPS:-400}" \
  --train-size "${TRAIN_SIZE:-10000}" \
  --val-size "${VAL_SIZE:-600}" \
  --train-curriculum "${TRAIN_CURRICULUM:-mixed}" \
  --val-curriculum "${VAL_CURRICULUM:-mixed}" \
  --train-easy-ratio "${TRAIN_EASY_RATIO:-0.35}" \
  --val-easy-ratio "${VAL_EASY_RATIO:-0.75}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.00005}" \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --trace-weight "${TRACE_WEIGHT:-0.5}" \
  --trace-struct-weight "${TRACE_STRUCT_WEIGHT:-1.0}" \
  --reasoning-struct-weight "${REASONING_STRUCT_WEIGHT:-0.5}" \
  --structured-answer-weight "${STRUCTURED_ANSWER_WEIGHT:-1.0}" \
  --answer-value-weight "${ANSWER_VALUE_WEIGHT:-2.0}" \
  --answer-contrastive-weight "${ANSWER_CONTRASTIVE_WEIGHT:-1.0}" \
  --contrastive-weight "${CONTRASTIVE_WEIGHT:-0.2}" \
  --batch-diversity-weight "${BATCH_DIVERSITY_WEIGHT:-1.0}" \
  --slot-diversity-weight "${SLOT_DIVERSITY_WEIGHT:-0.15}" \
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/predictor_answer_latent}"
