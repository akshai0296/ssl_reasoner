#!/usr/bin/env bash
# A/B: does the H-JEPA plan level help over the flat latent reasoning sequence?
#
# Both arms warm-start from the SAME L1 checkpoint and train for the same steps on
# the same data. The only difference:
#   arm A (baseline) trains the flat L1 sequence alone   (--stages latent_reasoning_sequence)
#   arm B (hjepa)    adds the L2 plan level on top of L1 (--stages hjepa)
# Then both are scored on every OOD preset: baseline via the slot-digit decode,
# hjepa via the same decode conditioned on the predicted plan latent (--hjepa-values).
#
# Override any knob via env var, e.g.  STEPS=4000 CHECKPOINT=checkpoints/foo/best.pt bash scripts/hjepa_ab.sh
set -euo pipefail

STEPS="${STEPS:-2000}"
TRAIN_SIZE="${TRAIN_SIZE:-30000}"
VAL_SIZE="${VAL_SIZE:-2000}"
CURRICULUM="${CURRICULUM:-multi_step_balanced}"
MAX_MATH_LEN="${MAX_MATH_LEN:-34}"
MAX_VARIABLE_STEPS="${MAX_VARIABLE_STEPS:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0003}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
DEVICE="${DEVICE:-auto}"
# Warm-start checkpoint shared by both arms (a trained flat L1 sequence).
CHECKPOINT="${CHECKPOINT:-checkpoints/latent_reasoning_sequence_recurrent_smoke/best.pt}"
BASELINE_DIR="${BASELINE_DIR:-checkpoints/ab_baseline_lrs}"
HJEPA_DIR="${HJEPA_DIR:-checkpoints/ab_hjepa}"

common_train=(
  --variable-reasoner-steps "$STEPS"
  --hjepa-steps "$STEPS"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --train-curriculum "$CURRICULUM"
  --val-curriculum "$CURRICULUM"
  --max-math-len "$MAX_MATH_LEN"
  --max-variable-steps "$MAX_VARIABLE_STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --device "$DEVICE"
  --use-math-features
  --checkpoint "$CHECKPOINT"
  --partial-checkpoint
)

echo "########## ARM A: baseline (flat L1) ##########"
PYTHONUNBUFFERED=1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -m ssl_reasoner.train \
  --stages latent_reasoning_sequence \
  "${common_train[@]}" \
  --output-dir "$BASELINE_DIR"

echo "########## ARM B: H-JEPA (plan + L1) ##########"
PYTHONUNBUFFERED=1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -m ssl_reasoner.train \
  --stages hjepa \
  "${common_train[@]}" \
  --output-dir "$HJEPA_DIR"

echo "########## EVAL: baseline vs hjepa on all presets ##########"
echo "===== ARM A baseline (--latent-reasoning-slot-digit-values) ====="
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint "$BASELINE_DIR/best.pt" --preset all --samples "$EVAL_SAMPLES" \
  --device "$DEVICE" --dump-errors 0 --latent-reasoning-slot-digit-values

echo "===== ARM B hjepa (--hjepa-values, + plan-order diagnostic) ====="
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint "$HJEPA_DIR/best.pt" --preset all --samples "$EVAL_SAMPLES" \
  --device "$DEVICE" --dump-errors 0 --hjepa-values --hjepa-plan-order
