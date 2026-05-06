#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/stage0_target_readout/best.pt \
  --partial-checkpoint \
  --stages 1,2 \
  --stage1-steps 5000 \
  --stage2-steps 1000 \
  --train-size 5000 \
  --val-size 500 \
  --batch-size 64 \
  --predictor-layers 6 \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_capacity_from_stage0
