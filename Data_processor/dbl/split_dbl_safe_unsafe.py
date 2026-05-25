"""Data_processor / dbl

Purpose: Split DBL final datasets into Safe/Unsafe CSV files for train, test, and val.

Inputs:  ../../Datasets/Processed_datasets/Final_Dataset/DBL_{split}.json
Outputs: ../../Datasets/Toxic/DBL/ and ../../Datasets/Benign/DBL/ CSVs

Usage:   python split_dbl_safe_unsafe.py

See:     ../README.md
"""

import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

DATASET_NAME = 'DBL'
SPLITS = ('train', 'test', 'val')
SPLIT_FILES = {
    'train': 'dataset.csv',
    'test': 'test_dataset.csv',
    'val': 'val_dataset.csv',
}


def export_split(split, label, out_path):
    dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{DATASET_NAME}_{split}.json"
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    subset = dataset.filter(lambda example: example['label'] == label)
    ensure_parent_dir(out_path)
    subset.to_csv(out_path)


for split in SPLITS:
    filename = SPLIT_FILES[split]
    export_split(split, 'Unsafe', f'../../Datasets/Toxic/{DATASET_NAME}/{filename}')
    export_split(split, 'Safe', f'../../Datasets/Benign/{DATASET_NAME}/{filename}')
