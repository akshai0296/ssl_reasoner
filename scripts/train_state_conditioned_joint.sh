#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint "${CHECKPOINT:-checkpoints/step_state_solver_mixed_only/best.pt}" \
  --stages state_conditioned_joint \
  --state-conditioned-steps "${STEPS:-1200}" \
  --state-conditioned-readout-weight "${READOUT_WEIGHT:-2.0}" \
  --train-size "${TRAIN_SIZE:-12000}" \
  --val-size "${VAL_SIZE:-600}" \
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
  --eval-every "${EVAL_EVERY:-300}" \
  --sample-count "${SAMPLE_COUNT:-5}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/state_conditioned_joint}"
