#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m ssl_reasoner.verifier \
  --checkpoint checkpoints/stage3_joint_low_lr/best.pt \
  --output checkpoints/verifier_v1.pt \
  --train-size 3000 \
  --val-size 500 \
  --batch-size 64 \
  --steps 500 \
  --lr 1e-3 \
  --noise-scale 0.25 \
  --eval-every 100 \
  --curriculum mixed
