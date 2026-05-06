#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_curriculum_mixed/best.pt \
  --stages 1 \
  --stage1-steps 3000 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum mixed_only \
  --val-curriculum mixed_only \
  --batch-size 64 \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_mixed_focus

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_mixed_focus/best.pt \
  --stages 1,2 \
  --stage1-steps 2000 \
  --stage2-steps 1000 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --batch-size 64 \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_mixed_focus_final
