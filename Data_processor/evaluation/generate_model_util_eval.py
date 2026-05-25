"""Data_processor / evaluation

Purpose: Build model utility baseline eval set from PersonaChat test data.

Inputs:  ../../Datasets/Benign/PersonaChat/test_dataset.csv
Outputs: ../../Datasets/Evaluation/Model_Util_evaluate.csv (5000 shuffled rows)

Usage:   python generate_model_util_eval.py

See:     ../README.md
"""

import os
import sys

from datasets import load_dataset,concatenate_datasets, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

dataset_path = "../../Datasets/Benign/PersonaChat/test_dataset.csv"

final_dataset = load_dataset("csv", data_files=dataset_path,split='train')

counter = 5000

final_dataset = final_dataset.shuffle(seed=42).select(range(counter))

output_path = '../../Datasets/Evaluation/Model_Util_evaluate.csv'
ensure_parent_dir(output_path)
final_dataset.to_csv(output_path)