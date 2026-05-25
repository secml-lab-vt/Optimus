"""Path helpers for STAR-DSS + Optimus integration scripts."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import (  # noqa: E402
    datasets_dir,
    logs_db_path,
    model_runs_dir,
    optimus_root,
)

__all__ = [
    "datasets_dir",
    "logs_db_path",
    "model_runs_dir",
    "optimus_root",
]
