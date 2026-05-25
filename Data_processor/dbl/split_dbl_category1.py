"""Data_processor / dbl

Purpose: Build Category1 toxic train variants with and without DBL rows for LM detect / Injection.

Inputs:  ../../Datasets/Toxic/Category1/dataset.csv, ../../Datasets/Toxic/DBL/dataset.csv
Outputs: ../../Datasets/Toxic/DBL_Category1/Category1_no_dbl_dataset.csv,
         Category1_w_dbl_dataset.csv

Usage:   python split_dbl_category1.py

See:     ../README.md
"""

import os
import sys

from datasets import concatenate_datasets, load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

SEED = 42
DBL_SAMPLE = 6000
OUT_DIR = '../../Datasets/Toxic/DBL_Category1'

category1 = load_dataset(
    'csv',
    data_files='../../Datasets/Toxic/Category1/dataset.csv',
    split='train',
)
dbl = load_dataset(
    'csv',
    data_files='../../Datasets/Toxic/DBL/dataset.csv',
    split='train',
)

category1_no_dbl = category1.filter(lambda x: x['label'] == 'Unsafe')
dbl_unsafe = dbl.filter(lambda x: x['label'] == 'Unsafe').shuffle(seed=SEED)
dbl_sample = dbl_unsafe.select(range(min(DBL_SAMPLE, len(dbl_unsafe))))
category1_w_dbl = concatenate_datasets([dbl_sample, category1_no_dbl])

for name, dataset in (
    ('Category1_no_dbl_dataset.csv', category1_no_dbl),
    ('Category1_w_dbl_dataset.csv', category1_w_dbl),
):
    out_path = os.path.join(OUT_DIR, name)
    ensure_parent_dir(out_path)
    dataset.to_csv(out_path)
