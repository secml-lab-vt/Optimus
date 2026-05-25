"""Data_processor / 01_raw_preprocessing

Purpose: Legacy one-off parser for BAD validation text format (val.txt).

Inputs:  ../../Datasets/Raw_data/BAD/val.txt
Outputs: ../../Datasets/Raw_data/BAD/val_processed.json

Usage:   python parse_bad_validation.py

See:     ../README.md
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import json
from datasets import load_dataset,concatenate_datasets, Dataset

# we create a balanced dataset for each class - 

datasetpath = '../../Datasets/Raw_data/BAD/val.txt'

file1 = open(datasetpath, 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character

dataset_list = []

for line in Lines:
    Dict = {}

    preprocess = line.split("text:")[-1].split("labels:")
    dialog = preprocess[0].strip()
    label =  preprocess[-1].split("episode_done:")[0].strip()
    if(label == "__ok__"):
        label = "Safe"
    else:
        label = "Unsafe"
    dialog_list = dialog.split("\\n")

    dialogs_combine = "|".join(dialog_list[:-1])
    length = len(dialog_list[:-1])
    response = dialog_list[-1]

    if(dialogs_combine == ""):
        continue

    Dict["context"] = dialogs_combine
    Dict["response"] = response
    Dict["label"] = label
    Dict["category"] = "Offending User"
    Dict["window_length"] = length
    Dict["source"] = "BB"



    dataset_list.append(Dict)

path = "../../Datasets/Raw_data/BAD/val_processed.json"

with open(path, 'w') as f:
    json_list = json.dump(dataset_list,f,indent=4)

# print(json_list)