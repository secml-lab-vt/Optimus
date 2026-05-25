"""Data_processor / dbl

Purpose: Export LM detect CSVs for DBL Category1 variants (with and without DBL).

Inputs:  ../../Datasets/Toxic/DBL_Category1/Category1_no_dbl and Category1_w_dbl CSVs
Outputs: ../../Datasets/LM_Detect/DBL_Category1_no_dbl_dataset.csv, DBL_Category1_w_dbl_dataset.csv

Usage:   python generate_lm_detect_dbl.py

See:     ../README.md
"""

import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

TOXIC_DATASET = 'DBL_Category1'
SEED = 42
VARIANTS = (
    ('Category1_no_dbl_dataset.csv', 'no_dbl'),
    ('Category1_w_dbl_dataset.csv', 'w_dbl'),
)


def export_variant(input_name, output_suffix):
    input_path = f'../../Datasets/Toxic/{TOXIC_DATASET}/{input_name}'
    output_path = f'../../Datasets/LM_Detect/{TOXIC_DATASET}_{output_suffix}_dataset.csv'
    dataset = load_dataset('csv', data_files=input_path, split='train')
    dataset = dataset.filter(lambda x: x['label'] == 'Unsafe').shuffle(seed=SEED)
    ensure_parent_dir(output_path)
    dataset.to_csv(output_path)


for input_name, output_suffix in VARIANTS:
    export_variant(input_name, output_suffix)
