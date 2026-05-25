"""Data_processor / dbl

Purpose: Concatenate DBL_BB400M and DBL_DD-BART toxic and benign per split.

Inputs:  Processed DBL JSON files from process_dbl_raw.py
Outputs: ../../Datasets/Processed_datasets/Final_Dataset/DBL_{split}.json

Usage:   python generate_dbl_dataset.py --split train

See:     ../README.md
"""

import argparse
import json
import random as r
import numpy as np
from datasets import load_dataset,concatenate_datasets, Dataset

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Arguments for train/val/test a Chatbot')
    parser.add_argument('--split', help='enter the split - test, train, val', nargs='?', default=-1)
    args = parser.parse_args()

    dp = Dataset_Processor_W_DBL(args)

    dp.data_process()
    
class Dataset_Processor_W_DBL:
    def __init__(self, args):
        self.set_up(args)
    
    def set_up(self, args):

        self.DBL_datasets = ['DBL_BB400M','DBL_DD-BART']

        if args.split !=-1:
            self.split = args.split
    
    def data_process(self):

            dataset_combine1 = []


            for x in self.DBL_datasets:
                datasetpath = f"../../Datasets/Processed_datasets/{x}/Toxic_{self.split}.json"
                dataset = load_dataset("json", data_files=datasetpath,split='train').shuffle(seed=42)

                dataset_combine1.append(dataset)

                datasetpath = f"../../Datasets/Processed_datasets/{x}/Benign_{self.split}.json"
                dataset = load_dataset("json", data_files=datasetpath,split='train').shuffle(seed=42)

                dataset_combine1.append(dataset)

            final_dataset = concatenate_datasets(dataset_combine1)

            final_dataset.to_json(f'../../Datasets/Processed_datasets/Final_Dataset/DBL_{self.split}.json')
        
if(__name__ == "__main__"):
    main()    
