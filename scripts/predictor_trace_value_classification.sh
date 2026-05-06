#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_structured_trace/best.pt \
  --partial-checkpoint \
  --stages 1,2 \
  --stage1-steps 3000 \
  --stage2-steps 1000 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --batch-size 64 \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --use-reasoning-trace \
  --trace-weight 0.5 \
  --trace-struct-weight 1.0 \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_trace_value_classification
