#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_ops_head_mixed/best.pt}" \
  --mode operation_fallback \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum "${CURRICULUM:-mixed}" \
  --operation-confidence-threshold "${OPERATION_CONFIDENCE_THRESHOLD:-0.0}" \
  --dump-errors "${DUMP_ERRORS:-0}"
