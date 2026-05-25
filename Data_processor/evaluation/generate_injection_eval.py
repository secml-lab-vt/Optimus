"""Data_processor / evaluation

Purpose: Build mixed benign+toxic evaluation CSVs for Injection_code injection experiments.

Inputs:  PersonaChat test CSV, Category1/Category2 toxic test/val/train slices
Outputs: ../../Datasets/Evaluation/Category1_evaluate.csv, Category2_evaluate.csv

Usage:   python generate_injection_eval.py

See:     ../README.md
"""

import os
import sys

from datasets import load_dataset,concatenate_datasets, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

dataset_path = "../../Datasets/Benign/PersonaChat/test_dataset.csv"

dataset_path1 = "../../Datasets/Toxic/Category1/test_dataset.csv"

final_dataset = load_dataset("csv", data_files=dataset_path,split='train')

final_dataset1 = load_dataset("csv", data_files=dataset_path1,split='train')


counter = 1000

counter1 = 1000

final_dataset = final_dataset.shuffle(seed=42).select(range(counter))

final_dataset1 = final_dataset1.shuffle(seed=42).select(range(counter1))



dataset = concatenate_datasets([final_dataset.select(range(counter)), final_dataset1.select(range(counter1))])


category1_out = '../../Datasets/Evaluation/Category1_evaluate.csv'
ensure_parent_dir(category1_out)
dataset.to_csv(category1_out)


dataset_path = "../../Datasets/Benign/PersonaChat/test_dataset.csv"

dataset_path1 = "../../Datasets/Toxic/Category2/test_dataset.csv"

dataset_path2 = "../../Datasets/Toxic/Category2/val_dataset.csv"

dataset_path3 = "../../Datasets/Toxic/Category2/dataset1.csv"

final_dataset = load_dataset("csv", data_files=dataset_path,split='train')

final_dataset1 = load_dataset("csv", data_files=dataset_path1,split='train')

final_dataset2 = load_dataset("csv", data_files=dataset_path2,split='train')

final_dataset3 = load_dataset("csv", data_files=dataset_path3,split='train')


counter = 1000
counter1 = 337
counter2 = 334
counter3 = 329

column_to_change = 'index'

# Define your custom function to apply to each element in the column
def custom_function(value,value1):

    return value1 + value

final_dataset = final_dataset.shuffle(seed=42).select(range(counter))

final_dataset1 = final_dataset1
final_dataset1 = final_dataset1.map(lambda example: {column_to_change: custom_function(example[column_to_change],'Test_')})

final_dataset2 = final_dataset2
final_dataset2 = final_dataset2.map(lambda example: {column_to_change: custom_function(example[column_to_change],'Val_')})

final_dataset3 = final_dataset3.select(range(counter3))
final_dataset3 = final_dataset3.map(lambda example: {column_to_change: custom_function(example[column_to_change],'Train_')})

dataset = concatenate_datasets([final_dataset, final_dataset1, final_dataset2, final_dataset3])

category2_out = '../../Datasets/Evaluation/Category2_evaluate.csv'
ensure_parent_dir(category2_out)
dataset.to_csv(category2_out)


from datasets import load_dataset

# Assuming you have already loaded the dataset
final_dataset1 = load_dataset("csv", data_files=dataset_path1, split='train')

# Replace 'your_column_name' with the actual name of the column you want to modify

# Use the map function to apply the custom function to the specified column

# Display the modified dataset
print(final_dataset1)
