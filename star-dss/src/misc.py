import shutil
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist


def ensure_empty_parent_dir(file_path: Union[str, Path]):
    file_path = Path(file_path)
    parent_dir = file_path.parent

    # ensure parent dir exists
    parent_dir.mkdir(parents=True, exist_ok=True)

    # remove all contents in parent dir
    for item in parent_dir.iterdir():
        if item.is_file():
            item.unlink()  # remove file
        elif item.is_dir():
            shutil.rmtree(item)  # Remove directory and its contents


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dist.is_initialized():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
