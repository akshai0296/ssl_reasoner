#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint "${CHECKPOINT:-checkpoints/variable_reasoner/best.pt}" \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  "$@"
