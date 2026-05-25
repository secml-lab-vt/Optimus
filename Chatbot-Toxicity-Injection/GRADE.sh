#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTIMUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRADE_ROOT="${OPTIMUS_ROOT}/GRADE"

cd "${GRADE_ROOT}/script/"
source activate grade_env
bash inference.sh "$1" "$2" "$3"
