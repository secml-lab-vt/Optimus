"""Data_processor / 02_category_generation

Purpose: Stage 1 — merge BAD, CADD, and DiaSafety (category 1) into Category1 final JSON.

Inputs:  ../../Datasets/Processed_datasets/{BAD,CADD,DiaSafety}/{split}*.json
Outputs: ../../Datasets/Processed_datasets/Final_Dataset/Category1_{split}.json

Usage:   python generate_category1_dataset.py --split train --name Category1

See:     ../README.md
"""

import argparse
import json
import os
import sys
import random as r
import numpy as np
from datasets import load_dataset,concatenate_datasets, Dataset

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir


def main():
    parser = argparse.ArgumentParser(description='Arguments for train/val/test a Chatbot')
    parser.add_argument('--split', help='enter the split - test, train, val', nargs='?', default=-1)
    parser.add_argument('--name', help='enter the dataset_name', nargs='?', default=-1)
    args = parser.parse_args()

    dp = Dataset_Processor(args)

    dp.data_process()
    
class Dataset_Processor:
    def __init__(self, args):
        self.set_up(args)
    
    def set_up(self, args):

        self.datasets = ['BAD','DiaSafety','CADD']
        self.DBL_datasets = ['BAD','DiaSafety','CADD']
        self.name = args.name
        if args.split !=-1:
            self.split = args.split


    def data_process(self):
            dataset_combine = []

            # if(self.split == 'train'):
            #     counter = 12000
            # elif(self.split == 'val'):
            #     counter = 1500
            # elif(self.split == 'test'):
            #     counter = 1500
            def add_index_dataset_label_columns(example, idx):
                # Add index and dataset name as new columns
                example["index"] = f"{idx}_{example['label']}_{example['source']}"
                
                return example

            for x in self.datasets:
                if(x == 'DiaSafety'):
                    datasetpath = f"../../Datasets/Processed_datasets/{x}/{self.split}1.json"
                else:
                    datasetpath = f"../../Datasets/Processed_datasets/{x}/{self.split}.json"

                dataset = load_dataset("json", data_files=datasetpath,split='train')

                # new_column = [x] * len(dataset)
                # dataset = dataset.add_column("source", new_column)
                dataset_combine.append(dataset)
                # dataset1 = dataset.filter(lambda example: example["label"] in ['Safe'])
                # dataset2 = dataset.filter(lambda example: example["label"] in ['Unsafe'])

            final_dataset = concatenate_datasets(dataset_combine)

            # dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe']).shuffle(seed=42).select(range(counter))
            # dataset2 = final_dataset.filter(lambda example: example["label"] in ['Unsafe']).shuffle(seed=42).select(range(counter))

            dataset1 = final_dataset.filter(lambda example: example["label"] in ['Safe']).shuffle(seed=42)
            dataset2 = final_dataset.filter(lambda example: example["label"] in ['Unsafe']).shuffle(seed=42)

            final_dataset = concatenate_datasets([dataset1,dataset2]).shuffle(seed=42)

            final_dataset = final_dataset.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

            out_path = f'../../Datasets/Processed_datasets/Final_Dataset/{self.name}_{self.split}.json'
            ensure_parent_dir(out_path)
            final_dataset.to_json(out_path)


        
if(__name__ == "__main__"):
    main()    
