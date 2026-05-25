"""Central path helpers for the Optimus repository.

Resolution order for repo root:
1. OPTIMUS_ROOT environment variable
2. Walk up from this file looking for README.md + environments/

Assets (datasets, models, GRADE data, CTI saves, RTR checkpoints) are read from
their standard directories under the repo root. Download archives separately and
copy files into those paths (see README.md and Optimus_models+datasets/README.md).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_MARKERS = ("README.md", "environments")


@lru_cache(maxsize=1)
def optimus_root() -> Path:
    env_root = os.environ.get("OPTIMUS_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "README.md").is_file():
            raise FileNotFoundError(
                "OPTIMUS_ROOT is set but does not look like the Optimus repo: {}".format(root)
            )
        return root

    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not locate Optimus repo root. Set OPTIMUS_ROOT to your clone directory."
    )


def datasets_dir() -> Path:
    return optimus_root() / "Datasets"


def models_dir() -> Path:
    return optimus_root() / "Models"


def model_runs_dir() -> Path:
    return models_dir() / "Custom" / "model_runs"


def huggingface_models_dir() -> Path:
    override = os.environ.get("OPTIMUS_HF_MODELS")
    if override:
        return Path(override).expanduser().resolve()
    return models_dir() / "HuggingFace_direct" / "models"


def grade_root() -> Path:
    return optimus_root() / "GRADE"


def cti_root() -> Path:
    return optimus_root() / "Chatbot-Toxicity-Injection"


def evaluation_root() -> Path:
    return optimus_root() / "Evaluation"


def logs_db_path() -> Path:
    return optimus_root() / "logs_database" / "logs.db"


# Backward-compatible aliases (assets live in module dirs above, not a separate bucket).
def assets_root() -> Path:
    return optimus_root()


def grade_assets_dir() -> Path:
    return grade_root()


def cti_assets_dir() -> Path:
    return cti_root()


def evaluation_assets_dir() -> Path:
    return evaluation_root()


def join_under(base: Path, *parts: str) -> Path:
    return base.joinpath(*parts)
