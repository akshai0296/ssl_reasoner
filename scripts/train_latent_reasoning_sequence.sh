#!/usr/bin/env bash
set -euo pipefail

args=(
  --stages latent_reasoning_sequence \
  --variable-reasoner-steps "${STEPS:-2000}" \
  --train-size "${TRAIN_SIZE:-30000}" \
  --val-size "${VAL_SIZE:-2000}" \
  --train-curriculum "${TRAIN_CURRICULUM:-multi_step_balanced}" \
  --val-curriculum "${VAL_CURRICULUM:-multi_step_balanced}" \
  --max-math-len "${MAX_MATH_LEN:-34}" \
  --max-variable-steps "${MAX_VARIABLE_STEPS:-16}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.0005}" \
  --use-math-features \
  --eval-every "${EVAL_EVERY:-250}" \
  --sample-count "${SAMPLE_COUNT:-3}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/latent_reasoning_sequence}"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
fi

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train "${args[@]}"
