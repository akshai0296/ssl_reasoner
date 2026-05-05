#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --steps 50 \
  --train-size 512 \
  --val-size 128 \
  --batch-size 32 \
  --d-model 64 \
  --num-slots 4
