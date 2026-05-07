#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m ssl_reasoner.eval \
  --checkpoint "${CHECKPOINT:-checkpoints/step_state_solver_mixed_only/best.pt}" \
  --mode step_state_solver \
  --samples "${SAMPLES:-500}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --curriculum "${CURRICULUM:-mixed}"
