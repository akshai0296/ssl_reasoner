#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-https://github.com/akshai0296/ssl_reasoner/releases/download/math-solver-v0.1.0/best.pt}"
OUTPUT="${OUTPUT:-checkpoints/trace_ops_head_mixed/best.pt}"
SHA256="${SHA256:-fcd81e049cf0b7c40a9f0f58918617d9f75aff5bd8cf62d507febbed8568e2ff}"

mkdir -p "$(dirname "${OUTPUT}")"
curl -L "${URL}" -o "${OUTPUT}"

actual="$(shasum -a 256 "${OUTPUT}" | awk '{print $1}')"
if [[ "${actual}" != "${SHA256}" ]]; then
  echo "checksum mismatch for ${OUTPUT}" >&2
  echo "expected ${SHA256}" >&2
  echo "actual   ${actual}" >&2
  exit 1
fi

echo "downloaded ${OUTPUT}"
