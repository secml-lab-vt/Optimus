"""Data_processor / 03_splits

Purpose: Stage 2 — split Final_Dataset JSON into Safe/Unsafe CSV files for
         Category1 and Category2 (train, test, val).

Inputs:  ../../Datasets/Processed_datasets/Final_Dataset/Category{1,2}_{split}.json
Outputs: ../../Datasets/Toxic/Category{1,2}/ and ../../Datasets/Benign/Category{1,2}/ CSVs

Usage:   python split_safe_unsafe.py

See:     ../README.md
"""

import os
import sys

from datasets import load_dataset,concatenate_datasets, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir


def write_csv(dataset, path):
    ensure_parent_dir(path)
    dataset.to_csv(path)


dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_train.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_train.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/dataset.csv')

dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_train.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_train.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/dataset.csv')

dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_test.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/test_dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_test.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/test_dataset.csv')

dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_test.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/test_dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_test.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/test_dataset.csv')


dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_val.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/val_dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_val.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Unsafe'])

write_csv(dataset1, f'../../Datasets/Toxic/{dataset_name}/val_dataset.csv')

dataset_name = 'Category1'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_val.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/val_dataset.csv')

dataset_name = 'Category2'

dataset_path = f"../../Datasets/Processed_datasets/Final_Dataset/{dataset_name}_val.json"

final_dataset = load_dataset("json", data_files=dataset_path,split='train')

dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe'])

write_csv(dataset1, f'../../Datasets/Benign/{dataset_name}/val_dataset.csv')
