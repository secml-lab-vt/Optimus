#!/usr/bin/env python3
"""Sanity-check GRADE layout for Optimus."""
from __future__ import print_function

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimus_paths import cti_root, grade_root, optimus_root  # noqa: E402

GRADE_ROOT = grade_root()
OPTIMUS_ROOT = optimus_root()
CTI_ROOT = cti_root()

REQUIRED = [
    GRADE_ROOT / "script" / "inference.sh",
    GRADE_ROOT / "tools" / "numberbatch-en-19.08.txt",
    GRADE_ROOT / "output" / "71" / "GRADE_K2_N10_N10" / "model_eval_best_71.ckpt",
    GRADE_ROOT / "data" / "DailyDialog",
    GRADE_ROOT / "preprocess" / "dataset" / "mydata" / "idf.dict",
    GRADE_ROOT / "preprocess" / "dataset" / "mydata" / "candi_keywords.txt",
    OPTIMUS_ROOT / "Injection_code" / "tools" / "GRADE.sh",
]

OPTIONAL_SYMLINKS = {
    CTI_ROOT / "GRADE": GRADE_ROOT,
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
    print("GRADE_ROOT:", GRADE_ROOT)
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
        print("[{}] {} ({})".format(status, path.relative_to(OPTIMUS_ROOT), msg))
        ok = ok and good

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
