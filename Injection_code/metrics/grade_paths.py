"""Resolve Optimus/GRADE paths independent of the caller's working directory."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import grade_root, model_runs_dir, optimus_root  # noqa: E402

OPTIMUS_ROOT = optimus_root()
GRADE_ROOT = grade_root()
MODEL_RUNS_DIR = model_runs_dir()
