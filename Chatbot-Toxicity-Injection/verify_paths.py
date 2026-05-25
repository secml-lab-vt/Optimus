#!/usr/bin/env python3
"""Sanity-check Chatbot-Toxicity-Injection asset layout for Optimus."""
from __future__ import print_function

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import cti_root, datasets_dir, grade_root, optimus_root  # noqa: E402

OPTIMUS_ROOT = optimus_root()
CTI_ROOT = cti_root()

REQUIRED = [
    CTI_ROOT / "data" / "cached_convs" / "DD-BART_friendly.txt",
    CTI_ROOT / "data" / "datasets" / "train_hh.txt",
    CTI_ROOT / "saves" / "base" / "DD-BART-BASE" / "config.json",
    CTI_ROOT / "saves" / "toxic_classifier" / "WTC_new_focal-2_1_best" / "config.json",
    CTI_ROOT / "saves" / "toxic_bot" / "reddit_bot_thres-0.99_4" / "config.json",
    CTI_ROOT / "Benign_Dataset.csv",
    datasets_dir() / "DBL_data",
]

OPTIONAL_SYMLINKS = {
    CTI_ROOT / "GRADE": grade_root(),
}


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
    print("CTI root:", CTI_ROOT)
    print()

    for link, expected in OPTIONAL_SYMLINKS.items():
        if not link.exists() and not link.is_symlink():
            print("[SKIP] {} (optional legacy symlink)".format(link.relative_to(OPTIMUS_ROOT)))
            continue
        good, msg = check_path(link)
        status = "OK" if good else "FAIL"
        print("[{}] {} ({})".format(status, link.relative_to(OPTIMUS_ROOT), msg))
        if good and link.is_symlink() and link.resolve() != expected.resolve():
            print("  WARN: expected target {}".format(expected))
        ok = ok and good

    print()
    for path in REQUIRED:
        good, msg = check_path(path)
        status = "OK" if good else "FAIL"
        try:
            label = path.relative_to(OPTIMUS_ROOT)
        except ValueError:
            label = path
        print("[{}] {} ({})".format(status, label, msg))
        ok = ok and good

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
