#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-gliner"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [ $# -lt 1 ]; then
  echo "usage: scripts/run_gliner_compat.sh <python_script> [args...]" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} is required but was not found on PATH." >&2
  echo "hint: install Python 3.11 or run with PYTHON_BIN=/path/to/python3.11." >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Creating GLiNER compatibility venv at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Syncing GLiNER compatibility dependencies..."
"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet --upgrade \
  "gliner==0.2.24" \
  "transformers==4.57.3" \
  "huggingface-hub<1.0" \
  "torch>=2.5.0"

PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV_DIR}/bin/python" \
  "$@"
