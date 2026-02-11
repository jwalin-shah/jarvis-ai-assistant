#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/run_gliner_compat.sh" \
  "${ROOT_DIR}/scripts/run_extractor_bakeoff.py" \
  "$@"
