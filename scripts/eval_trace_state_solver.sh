#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint "${CHECKPOINT:-checkpoints/trace_state_head_mixed/best.pt}" \
  --mode trace_state_solver \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum "${CURRICULUM:-mixed_only}"
