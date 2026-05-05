#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --overfit \
  --stage0-steps 500 \
  --stage1-steps 1500 \
  --stage2-steps 1000 \
  --train-size 128 \
  --val-size 128 \
  --batch-size 32 \
  --d-model 128 \
  --num-slots 8 \
  --eval-every 100 \
  --sample-count 8 \
  --output-dir checkpoints/overfit_tiny
