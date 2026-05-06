#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_structured_answer/best.pt \
  --stages 1,2 \
  --stage1-steps 1000 \
  --stage2-steps 500 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --train-easy-ratio 0.6 \
  --val-easy-ratio 0.75 \
  --batch-size 64 \
  --lr 5e-5 \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --use-trace-fusion \
  --trace-weight 0.5 \
  --trace-struct-weight 1.0 \
  --reasoning-struct-weight 0.5 \
  --structured-answer-weight 1.0 \
  --eval-every 250 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_reasoning_stage1
