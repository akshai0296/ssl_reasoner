#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/trace_ops_head_mixed/best.pt}"
REPORT_PATH="${REPORT_PATH:-reports/trace_ops_head_repro.json}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SAMPLES="${SAMPLES:-500}"
mkdir -p "$(dirname "${REPORT_PATH}")"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  OUTPUT_DIR="$(dirname "${CHECKPOINT_PATH}")" bash scripts/finetune_trace_ops_head.sh
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_eval() {
  local curriculum="$1"
  local output_file="${tmp_dir}/${curriculum}.txt"
  CHECKPOINT="${CHECKPOINT_PATH}" CURRICULUM="${curriculum}" SAMPLES="${SAMPLES}" \
    bash scripts/eval_math_solver.sh > "${output_file}"
  local line
  line="$(grep 'operation_fallback_exact_match=' "${output_file}")"
  local accuracy
  accuracy="$(printf '%s\n' "${line}" | sed -E 's/.*=([0-9.]+) .*/\1/')"
  local counts
  counts="$(printf '%s\n' "${line}" | sed -E 's/.*\(([0-9]+)\/([0-9]+)\).*/\1 \2/')"
  local correct total
  correct="$(printf '%s\n' "${counts}" | awk '{print $1}')"
  total="$(printf '%s\n' "${counts}" | awk '{print $2}')"
  printf '    "%s": {"accuracy": %s, "correct": %s, "total": %s}' \
    "${curriculum}" "${accuracy}" "${correct}" "${total}"
}

{
  printf '{\n'
  printf '  "checkpoint": "%s",\n' "${CHECKPOINT_PATH}"
  printf '  "base_checkpoint": "%s",\n' \
    "${CHECKPOINT:-checkpoints/predictor_reasoning_stage1_mixed_anticollapse/best.pt}"
  printf '  "skip_train": %s,\n' "$( [[ "${SKIP_TRAIN}" == "1" ]] && printf true || printf false )"
  printf '  "samples": %s,\n' "${SAMPLES}"
  printf '  "eval_results": {\n'
  run_eval "mixed"
  printf ',\n'
  run_eval "mixed_only"
  printf ',\n'
  run_eval "seen_mixed"
  printf ',\n'
  run_eval "unseen_mixed"
  printf '\n'
  printf '  }\n'
  printf '}\n'
} > "${REPORT_PATH}"

cat "${REPORT_PATH}"
