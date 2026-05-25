#!/usr/bin/env bash
# Shared setup for Optimus star-dss batch scripts.
# Usage: source "$(dirname "$0")/../common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAR_DSS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPTIMUS_ROOT="${OPTIMUS_ROOT:-$(cd "${STAR_DSS_ROOT}/.." && pwd)}"

cd "${STAR_DSS_ROOT}"
export OPTIMUS_ROOT
export PYTHONPATH="${STAR_DSS_ROOT}:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llm_clone" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_clone
  fi
fi
