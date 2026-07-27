#!/usr/bin/env bash
set -euo pipefail

args=(
  --stages latent_reasoning_sequence
  --variable-reasoner-steps "${STEPS:-1000}"
  --train-size "${TRAIN_SIZE:-30000}"
  --val-size "${VAL_SIZE:-2000}"
  --train-curriculum "${TRAIN_CURRICULUM:-multi_step_balanced}"
  --val-curriculum "${VAL_CURRICULUM:-multi_step_balanced}"
  --max-math-len "${MAX_MATH_LEN:-34}"
  --max-variable-steps "${MAX_VARIABLE_STEPS:-16}"
  --batch-size "${BATCH_SIZE:-64}"
  --lr "${LR:-0.0003}"
  --use-math-features
  --copy-update-state-weight "${COPY_UPDATE_STATE_WEIGHT:-3.0}"
  --copy-update-slot-weight "${COPY_UPDATE_SLOT_WEIGHT:-2.0}"
  --copy-update-result-weight "${COPY_UPDATE_RESULT_WEIGHT:-2.0}"
  --copy-update-predicted-result-weight "${COPY_UPDATE_PREDICTED_RESULT_WEIGHT:-4.0}"
  --slot-digit-weight "${SLOT_DIGIT_WEIGHT:-1.0}"
  --predicted-slot-digit-weight "${PREDICTED_SLOT_DIGIT_WEIGHT:-1.0}"
  --slot-digit-carry-weight "${SLOT_DIGIT_CARRY_WEIGHT:-0.5}"
  --predicted-slot-digit-carry-weight "${PREDICTED_SLOT_DIGIT_CARRY_WEIGHT:-0.5}"
  --eval-every "${EVAL_EVERY:-250}"
  --sample-count "${SAMPLE_COUNT:-3}"
  --output-dir "${OUTPUT_DIR:-checkpoints/latent_reasoning_copy_update}"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
fi

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train "${args[@]}"
