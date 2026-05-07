#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.solve \
  --checkpoint "${CHECKPOINT:-checkpoints/predictor_reasoning_stage1_mixed_anticollapse/best.pt}" \
  "$@"
