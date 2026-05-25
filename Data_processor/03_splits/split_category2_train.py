"""Data_processor / 03_splits

Purpose: Partition Category2 toxic train into two subsets (first 2200 rows vs remainder).
         Required before generating Category2 injection eval datasets.

Inputs:  ../../Datasets/Toxic/Category2/dataset.csv
Outputs: dataset.csv, dataset1.csv, Category2_train_split.json, Category2_train_split1.json

Usage:   python split_category2_train.py

See:     ../README.md
"""

from datasets import load_dataset, concatenate_datasets
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

# Configuration
input_path = "../../Datasets/Toxic/Category2/dataset.csv"
split_size = 2200

# Output paths
csv_output_dir = "../../Datasets/Toxic/Category2/"
json_output_dir = "../../Datasets/Processed_datasets/Final_Dataset/"

# Load the original dataset
print(f"Loading dataset from: {input_path}")
dataset = load_dataset("csv", data_files=input_path, split='train')

print(f"Original dataset size: {len(dataset)}")

# Split the dataset
print(f"Splitting dataset: first {split_size} samples, remaining {len(dataset) - split_size} samples")
dataset1 = dataset.select(range(split_size))
dataset2 = dataset.select(range(split_size, len(dataset)))

print(f"Split 1 size: {len(dataset1)}")
print(f"Split 2 size: {len(dataset2)}")

# Output CSV files
csv_path1 = f"{csv_output_dir}/dataset.csv"
csv_path2 = f"{csv_output_dir}/dataset1.csv"

print(f"Writing CSV files:")
print(f"  Split 1: {csv_path1}")
print(f"  Split 2: {csv_path2}")

ensure_parent_dir(csv_path1)
dataset1.to_csv(csv_path1)
ensure_parent_dir(csv_path2)
dataset2.to_csv(csv_path2)

# Output JSON files
json_path1 = f"{json_output_dir}/Category2_train_split.json"
json_path2 = f"{json_output_dir}/Category2_train_split1.json"

print(f"Writing JSON files:")
print(f"  Split 1: {json_path1}")
print(f"  Split 2: {json_path2}")

ensure_parent_dir(json_path1)
dataset1.to_json(json_path1)
ensure_parent_dir(json_path2)
dataset2.to_json(json_path2)

print("Processing complete!")
print(f"Files written:")
print(f"  CSV: {csv_path1}, {csv_path2}")
print(f"  JSON: {json_path1}, {json_path2}")