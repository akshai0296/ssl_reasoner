#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_structured_answer/best.pt \
  --stages 3 \
  --stage3-steps 200 \
  --train-size 3000 \
  --val-size 500 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --train-easy-ratio 0.75 \
  --val-easy-ratio 0.75 \
  --batch-size 32 \
  --lr 1e-5 \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --joint-pred-weight 1.0 \
  --joint-contrastive-weight 0.1 \
  --joint-vicreg-weight 0.05 \
  --joint-token-weight 0.5 \
  --joint-length-weight 0.1 \
  --trace-weight 0.5 \
  --trace-struct-weight 1.0 \
  --structured-answer-weight 1.0 \
  --eval-every 50 \
  --sample-count 5 \
  --output-dir checkpoints/stage3_joint_low_lr
