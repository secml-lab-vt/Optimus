#!/usr/bin/env bash
# One-time Optimus setup: move downloaded assets from Optimus_models+datasets/
# into the standard repo directories, then verify layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTIMUS_ROOT="${OPTIMUS_ROOT:-${SCRIPT_DIR}}"
BUNDLE="${OPTIMUS_ROOT}/Optimus_models+datasets"

remove_if_symlink() {
  local path="$1"
  if [[ -L "${path}" ]]; then
    rm "${path}"
    echo "removed symlink ${path}"
  fi
}

move_file() {
  local src="$1"
  local dest="$2"
  if [[ ! -e "${src}" ]]; then
    return 0
  fi
  remove_if_symlink "${dest}"
  mkdir -p "$(dirname "${dest}")"
  mv "${src}" "${dest}"
  echo "moved ${src} -> ${dest}"
}

move_dir() {
  local src="$1"
  local dest="$2"

  if [[ ! -e "${src}" ]]; then
    return 0
  fi

  remove_if_symlink "${dest}"

  if [[ ! -e "${dest}" ]]; then
    mkdir -p "$(dirname "${dest}")"
    mv "${src}" "${dest}"
    echo "moved ${src} -> ${dest}"
    return 0
  fi

  if [[ ! -d "${src}" || ! -d "${dest}" ]]; then
    echo "ERROR: cannot merge ${src} into ${dest}" >&2
    exit 1
  fi

  local item base
  shopt -s dotglob nullglob
  for item in "${src}"/*; do
    base="$(basename "${item}")"
    if [[ -e "${dest}/${base}" ]]; then
      if [[ -d "${item}" && -d "${dest}/${base}" ]]; then
        move_dir "${item}" "${dest}/${base}"
      else
        echo "skip (already exists): ${dest}/${base}"
      fi
    else
      mv "${item}" "${dest}/${base}"
      echo "moved ${item} -> ${dest}/${base}"
    fi
  done

  rmdir "${src}" 2>/dev/null || true
}

move_evaluation_module() {
  local module="$1"
  local src_module="${BUNDLE}/Evaluation/${module}"
  local dest_module="${OPTIMUS_ROOT}/Evaluation/${module}"

  if [[ ! -d "${src_module}" ]]; then
    return 0
  fi

  mkdir -p "${dest_module}"
  local item base
  shopt -s dotglob nullglob
  for item in "${src_module}"/*; do
    base="$(basename "${item}")"
    if [[ -d "${item}" ]]; then
      move_dir "${item}" "${dest_module}/${base}"
    else
      move_file "${item}" "${dest_module}/${base}"
    fi
  done
  rmdir "${src_module}" 2>/dev/null || true
}

link_relative() {
  local link="$1"
  local target="$2"
  local parent rel
  parent="$(dirname "${link}")"
  mkdir -p "${parent}"
  rel="$(python3 - "${parent}" "${target}" <<'PY'
import os, sys
print(os.path.relpath(sys.argv[2], start=sys.argv[1]))
PY
)"
  if [[ -L "${link}" ]]; then
    current="$(readlink "${link}")"
    if [[ "${current}" == "${rel}" ]]; then
      return 0
    fi
    rm "${link}"
  elif [[ -e "${link}" ]]; then
    return 0
  fi
  ln -s "${rel}" "${link}"
  echo "linked ${link} -> ${rel}"
}

if [[ ! -d "${BUNDLE}" ]]; then
  echo "ERROR: download bundle not found at ${BUNDLE}" >&2
  echo "Download Optimus_models+datasets/ and place it inside your Optimus clone." >&2
  echo "See Optimus_models+datasets/README.md for contents." >&2
  exit 1
fi

echo "Optimus root: ${OPTIMUS_ROOT}"
echo "Asset bundle: ${BUNDLE}"
echo

mkdir -p "${OPTIMUS_ROOT}/Models/Custom/model_runs"
mkdir -p "${OPTIMUS_ROOT}/logs_database"
mkdir -p "${OPTIMUS_ROOT}/Evaluation"
mkdir -p "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection"
mkdir -p "${OPTIMUS_ROOT}/GRADE"

echo "Moving assets from Optimus_models+datasets/ ..."

move_dir "${BUNDLE}/Datasets" "${OPTIMUS_ROOT}/Datasets"

move_dir "${BUNDLE}/GRADE/data" "${OPTIMUS_ROOT}/GRADE/data"
move_dir "${BUNDLE}/GRADE/tools" "${OPTIMUS_ROOT}/GRADE/tools"
move_dir "${BUNDLE}/GRADE/output" "${OPTIMUS_ROOT}/GRADE/output"
rmdir "${BUNDLE}/GRADE" 2>/dev/null || true

move_dir "${BUNDLE}/Chatbot-Toxicity-Injection/data" \
  "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/data"
move_dir "${BUNDLE}/Chatbot-Toxicity-Injection/saves" \
  "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/saves"
move_file "${BUNDLE}/Chatbot-Toxicity-Injection/Benign_Dataset.csv" \
  "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/Benign_Dataset.csv"
rmdir "${BUNDLE}/Chatbot-Toxicity-Injection" 2>/dev/null || true

move_dir "${BUNDLE}/Universal-Prompt-Injection/data" \
  "${OPTIMUS_ROOT}/Universal-Prompt-Injection/data"
rmdir "${BUNDLE}/Universal-Prompt-Injection" 2>/dev/null || true

move_evaluation_module "RTR_Evaluation_Classifier"
move_evaluation_module "RTR_Evaluation_Classifier_Heal"
rmdir "${BUNDLE}/Evaluation" 2>/dev/null || true

if [[ -d "${BUNDLE}/Models" ]]; then
  move_dir "${BUNDLE}/Models" "${OPTIMUS_ROOT}/Models"
fi

echo
echo "Asset move complete (Optimus_models+datasets/README.md kept as manifest)."
echo

# Legacy CTI scripts expect Chatbot-Toxicity-Injection/GRADE -> ../GRADE
if [[ -d "${OPTIMUS_ROOT}/GRADE" && ! -e "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/GRADE" ]]; then
  link_relative "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/GRADE" "${OPTIMUS_ROOT}/GRADE"
fi

echo "Running verify_paths checks..."
failed=0
for script in \
  "${OPTIMUS_ROOT}/GRADE/verify_paths.py" \
  "${OPTIMUS_ROOT}/Evaluation/verify_paths.py" \
  "${OPTIMUS_ROOT}/Chatbot-Toxicity-Injection/verify_paths.py"
do
  if [[ -f "${script}" ]]; then
    echo "--- ${script} ---"
    if ! python3 "${script}"; then
      failed=1
    fi
    echo
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "Setup completed with verification failures." >&2
  echo "Ensure Optimus_models+datasets/ is fully downloaded, then re-run setup_optimus.sh." >&2
  exit 1
fi

echo "Setup complete."
echo "Optional: export OPTIMUS_ROOT=\"${OPTIMUS_ROOT}\""
