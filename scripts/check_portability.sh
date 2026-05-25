#!/usr/bin/env bash
# Fail if tracked Python/shell sources contain non-portable absolute paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

failures=0

check_pattern() {
  local label="$1"
  local pattern="$2"
  local matches
  matches="$(git ls-files '*.py' '*.sh' 2>/dev/null \
    | grep -Ev '^Optimus_models\+datasets/.*\.txt$|^GRADE/texar-pytorch/|^scripts/check_portability\.sh$' \
    | xargs -r grep -nE "$pattern" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    echo "FAIL: found $label in tracked .py/.sh files:"
    echo "$matches"
    failures=$((failures + 1))
  fi
}

check_pattern "cluster path /projects/secml-cs-group" '/projects/secml-cs-group'
check_pattern "home directory /home/<user>" '/home/[a-zA-Z0-9._-]+'
check_pattern "chatsec2 reference" 'chatsec2'

if [[ "$failures" -gt 0 ]]; then
  echo
  echo "$failures portability check(s) failed."
  exit 1
fi

echo "Portability check passed."
