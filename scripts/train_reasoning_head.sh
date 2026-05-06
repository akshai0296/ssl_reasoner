#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/stage3_joint_low_lr/best.pt \
  --stages reasoning_head \
  --reasoning-head-steps 1000 \
  --train-size 6000 \
  --val-size 500 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --train-easy-ratio 0.5 \
  --val-easy-ratio 0.75 \
  --batch-size 64 \
  --lr 1e-3 \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --eval-every 200 \
  --sample-count 5 \
  --output-dir checkpoints/reasoning_head
