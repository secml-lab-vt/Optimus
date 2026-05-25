#!/usr/bin/env python3
"""Sanity-check Evaluation (RTR classifier) asset layout for Optimus."""
from __future__ import print_function

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import datasets_dir, evaluation_root, optimus_root  # noqa: E402

OPTIMUS_ROOT = optimus_root()
EVAL_ROOT = evaluation_root()

REQUIRED = [
    ("RTR_Evaluation_Classifier", "models_binary_category1/model_16_5e-06"),
    ("RTR_Evaluation_Classifier", "models_binary_category2/model_16_5e-06"),
    ("RTR_Evaluation_Classifier", "logs_binary_category1/test-precision_tuned/log_16_5e-06.txt"),
    ("RTR_Evaluation_Classifier_Heal", "models_binary_category1/model_16_5e-06"),
    ("RTR_Evaluation_Classifier_Heal", "models_binary_category2/model_16_5e-06"),
    ("RTR_Evaluation_Classifier_Heal", "logs_binary_category2/test-precision_tuned/log_16_5e-06.txt"),
    ("Datasets", "Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category1_train.json"),
    ("Datasets", "Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_train.json"),
]


def check_path(path):
    if not path.exists():
        return False, "missing"
    if path.is_symlink():
        target = path.resolve()
        if not target.exists():
            return False, "broken symlink -> {}".format(os.readlink(path))
        return True, "symlink -> {}".format(target)
    return True, "ok"


def main():
    ok = True
    print("Evaluation root:", EVAL_ROOT)
    print()

    for module_or_datasets, rel in REQUIRED:
        if module_or_datasets == "Datasets":
            path = datasets_dir() / Path(rel)
        else:
            path = EVAL_ROOT / module_or_datasets / Path(rel)
        good, msg = check_path(path)
        status = "OK" if good else "FAIL"
        label = "{}/{}".format(module_or_datasets, rel)
        print("[{}] {} ({})".format(status, label, msg))
        ok = ok and good

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
