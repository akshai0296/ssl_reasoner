#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/stage0_target_readout/best.pt \
  --partial-checkpoint \
  --stages 1 \
  --stage1-steps 3000 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum single_op_balanced \
  --val-curriculum single_op_balanced \
  --batch-size 64 \
  --predictor-type cross_attn \
  --predictor-layers 4 \
  --use-math-features \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_curriculum_single_op

PYTHONPATH=src python -m ssl_reasoner.train \
  --checkpoint checkpoints/predictor_curriculum_single_op/best.pt \
  --stages 1,2 \
  --stage1-steps 3000 \
  --stage2-steps 1000 \
  --train-size 6000 \
  --val-size 600 \
  --train-curriculum mixed \
  --val-curriculum mixed \
  --batch-size 64 \
  --eval-every 500 \
  --sample-count 5 \
  --output-dir checkpoints/predictor_curriculum_mixed
