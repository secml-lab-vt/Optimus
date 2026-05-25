"""Data_processor / dbl

Purpose: Parse DBL conversation files (BB400M, DD-BART) into train/val/test JSON.

Inputs:  ../../Datasets/Raw_data/DBL_{dataset}/{dataset}_{type}.txt
Outputs: ../../Datasets/Processed_datasets/DBL_{dataset}/{type}_{split}.json

Usage:   python process_dbl_raw.py --dataset BB400M --type Toxic

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
    parser.add_argument('--dataset', help='', nargs='?', default=-1)
    parser.add_argument('--type', help='', nargs='?', default=-1)

    args = parser.parse_args()
    dp = DBL_processor(args)

    datasetpath = f"../../Datasets/Raw_data/DBL_{args.dataset}/{args.dataset}_{args.type}.txt"
    
    contexts, responses, flags = dp.read_cache(datasetpath)
    if(args.type == "Toxic"):
        DF_LISTS = pd.DataFrame(
                {'context': contexts,
                'response': responses,
                'label': ["Unsafe"] * len(contexts),
                'category': ["Toxic"] * len(contexts),
                'implicit': ["no"] * len(contexts),
                'source': ['DBL'] * len(contexts)
                })
    elif(args.type == "Benign"):
        DF_LISTS = pd.DataFrame(
                {'context': contexts,
                'response': responses,
                'label': ["Safe"] * len(contexts),
                'category': ["Benign"] * len(contexts),
                'implicit': ["no"] * len(contexts),
                'source': ['DBL'] * len(contexts)
                })
    
    def add_index_dataset_label_columns(example, idx):
        # Add index and dataset name as new columns
        example["index"] = f"{idx}_{example['label']}_{example['source']}"
        
        return example    

    dataset = Dataset.from_pandas(DF_LISTS)

    # storepath = f"../../Datasets/{args.type}/DBL_{args.dataset}/dataset.csv"

    # dataset.to_csv(storepath)

    # split datadet in 90:10
    dataset = dataset.train_test_split(test_size=0.25, shuffle=False, seed=42)

    # get only train split from the dataset as new dataset

    dataset_train = dataset['train']

    dataset_train = dataset_train.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

    dataset_test = dataset['test']

    dataset_test = dataset_test.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)


    dataset_test = dataset_test.train_test_split(test_size=0.50, shuffle=False, seed=42)

    dataset_val = dataset_test['train']

    dataset_test = dataset_test['test']

    print(dataset_train)

    storepath = f"../../Datasets/Processed_datasets/DBL_{args.dataset}/{args.type}_train.json"

    dataset_train.to_json(storepath)

    storepath = f"../../Datasets/Processed_datasets/DBL_{args.dataset}/{args.type}_test.json"

    dataset_test.to_json(storepath)

    storepath = f"../../Datasets/Processed_datasets/DBL_{args.dataset}/{args.type}_val.json"

    dataset_val.to_json(storepath)
    
    
class DBL_processor:
    def __init__(self, args):
        self.set_up(args)
    
    def set_up(self, args):
        self.context_len = 3
        self.turns_each = 5


    def read_convs(self, file_name):
        with open(file_name) as f:
            convs = f.read().strip().split("\n\n")
        if("=" in convs[0]): #Remove header if found
            convs = convs[1:]
        return convs

    def read_cache(self, file_name, use_window=True, get_turn_nums=False):
        convs = self.read_convs(file_name)
        r.seed(0)
        r.shuffle(convs)
        contexts, responses, flags, turn_nums = self.c_to_cr(convs, shuffle=True, get_turn_nums=True)
        if(use_window): contexts = ["|".join(x.split("|")[-self.context_len:]) for x in contexts]
        if(get_turn_nums): return contexts, responses, flags, turn_nums
        return contexts, responses, flags

    def c_to_cr(self, convs, shuffle=False, get_turn_nums=False):
        turns = []
        for conv in convs:
            flags_found = ("|" in conv)
            lines = [x.split("|") for x in conv.split("\n")]
            if(len(lines) != self.turns_each * 2 + 1):
                print(lines)
                print("len(lines)", len(lines))
                print("len(triples)", len(triples))
                raise ValueError("Invalid conversation found!")
            if(flags_found): flags, utters = list(zip(*lines))
            if(flags_found == False): utters, flags = (conv.split("\n"), ["none"]*(self.turns_each * 2 + 1))
            for i in range(2, len(lines), 2):
                con_window = '|'.join(utters[max(i-self.context_len, 0):i])
                assert(flags[i] != "victim")
                turns.append((con_window, utters[i], flags[i], i))
        if(shuffle): r.shuffle(turns)
        contexts, responses, flags, nums = list(zip(*turns))
        if(get_turn_nums): return contexts, list(responses), flags, nums
        return contexts, list(responses), flags 

if(__name__ == "__main__"):
    main()