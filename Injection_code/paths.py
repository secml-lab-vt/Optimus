"""Path helpers for Injection_code scripts (cwd-independent)."""
from __future__ import annotations

import sys
from pathlib import Path

_INJECTION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INJECTION_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import datasets_dir, model_runs_dir  # noqa: E402


def datasets(*parts: str) -> Path:
    return datasets_dir().joinpath(*parts)


def model_runs(*parts: str) -> Path:
    return model_runs_dir().joinpath(*parts)


def eval_csv(name: str) -> Path:
    stem = name[:-4] if name.endswith(".csv") else name
    return datasets("Evaluation", f"{stem}.csv")
