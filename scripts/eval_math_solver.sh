#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint "${CHECKPOINT:-checkpoints/predictor_reasoning_stage1_mixed_anticollapse/best.pt}" \
  --mode operation_fallback \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum "${CURRICULUM:-mixed}" \
  --dump-errors "${DUMP_ERRORS:-0}"
