#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CHECKPOINT="checkpoints/trace_ops_head_mixed/best.pt"
for arg in "$@"; do
  if [[ "$arg" == "--debug-reasoning" ]]; then
    DEFAULT_CHECKPOINT="checkpoints/step_state_solver_mixed_only/best.pt"
    break
  fi
done

PYTHONPATH=src python -m ssl_reasoner.solve \
  --checkpoint "${CHECKPOINT:-$DEFAULT_CHECKPOINT}" \
  "$@"
