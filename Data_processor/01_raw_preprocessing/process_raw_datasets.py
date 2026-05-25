"""Data_processor / 01_raw_preprocessing

Purpose: Stage 0 — normalize raw BAD, CADD, DiaSafety, PersonaChat, and DailyDialog
         into a common schema under Processed_datasets/ and Benign/.

Inputs:  ../../Datasets/Raw_data/{dataset}/{split}.json (or PersonaChat/DailyDialog text files)
Outputs: ../../Datasets/Processed_datasets/ and ../../Datasets/Benign/

Usage:   python process_raw_datasets.py --dataset BAD --split train
         python process_raw_datasets.py --dataset DiaSafety --split train --category 1

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
    parser.add_argument('--dataset', help='enter the dataset', nargs='?', default=-1)
    parser.add_argument('--split', help='enter the split - test, train, val', nargs='?', default=-1)
    parser.add_argument('--category', help='enter the split - 1, 2', nargs='?', default=-1)
    
    args = parser.parse_args()

    dp = Dataset_Processor(args)
    dp.data_process()
    
class Dataset_Processor:
    def __init__(self, args):
        self.set_up(args)
    
    def set_up(self, args):
        self.dataset = args.dataset
        self.category = int(args.category)

        if args.split !=-1:
            self.split = args.split
    
    def data_process(self):
        if(self.dataset == 'BAD'):
            datasetpath = f"../../Datasets/Raw_data/{self.dataset}/{self.split}.json"
            dataset = load_dataset("json", data_files=datasetpath,split='train')
            dataset = dataset.filter(lambda example: example["window_length"] in [1,2,3])
            def preprocess(example):
                context = example['context']
                response = example['response']
                category = example['category']
                label = example['label']
                implicit = ''
                source = 'BAD'
                
                return {'context':context, 'response': response , 'category': category, 'label': label,'implicit':implicit,'source':source}
            
            dataset = dataset.map(preprocess, remove_columns = dataset.column_names)

            out_path = f"../../Datasets/Processed_datasets/{self.dataset}/{self.split}.json"
            ensure_parent_dir(out_path)
            dataset.to_json(out_path)

        if(self.dataset == 'DiaSafety'):
            datasetpath = f"../../Datasets/Raw_data/{self.dataset}/{self.split}.json"
            dataset = load_dataset("json", data_files=datasetpath,split='train')
            if(self.category == 1):
                dataset = dataset.filter(lambda example: example["category"] in ['Offending User'] )
            elif(self.category == 2):
                dataset = dataset.filter(lambda example: example["category"] in ['Risk Ignorance','Biased Opinion','Toxicity Agreement'] )
            
            def preprocess(example):
                context = example['context']
                response = example['response']
                category = example['category']
                label = example['label']
                implicit = 'yes'
                source = 'DiaSafety'

                return {'context':context, 'response': response , 'category': category, 'label': label,'implicit':implicit, 'source':source}
            
            dataset = dataset.map(preprocess, remove_columns = dataset.column_names)

            # 
            if(self.category == 1):
                out_path = f"../../Datasets/Processed_datasets/{self.dataset}/{self.split}{str(self.category)}.json"
            elif(self.category == 2):
                out_path = f"../../Datasets/Processed_datasets/{self.dataset}/{self.split}{str(self.category)}.json"
            ensure_parent_dir(out_path)
            dataset.to_json(out_path)


        elif(self.dataset == 'CADD'):
            datasetpath = f"../../Datasets/Raw_data/{self.dataset}/{self.split}.json"
            f = open(datasetpath)
            obj = json.load(f)
            preprocess_file = f'../../Datasets/Raw_data/{self.dataset}/{self.split}_processed.json'
            ensure_parent_dir(preprocess_file)
            file = open(preprocess_file, 'w')
            for iterator in obj:
                new_str = json.dumps(obj[iterator])
                file.write(new_str)

            # with open(, 'w') as f:
            #         json.dump(json_list, f, indent = 6)

            datasetpath1 = f"../../Datasets/Raw_data/{self.dataset}/{self.split}_processed.json"
            
            def preprocess(example):
                context = example['body']
                response = example['comment']
                cat_dict = {'0': 'Non-abusive', '1': 'Hate speech', '2': 'Derogatory', '3': 'Profanity'}
                category = cat_dict[example['L1']]
                label_dict = {'0': 'Safe', '1': 'Unsafe'}
                label = label_dict[example['L2']]
                implicit_dict = {'0': "no", '1': 'yes'}
                implicit = implicit_dict[example['L5']]
                source = 'CADD'

                return {'context':context, 'response': response , 'category': category, 'label': label,'implicit':implicit,'source':source}
            
            dataset = load_dataset("json", data_files=datasetpath1,split='train')
            
            dataset = dataset.map(preprocess, remove_columns = dataset.column_names)

            out_path = f"../../Datasets/Processed_datasets/{self.dataset}/{self.split}.json"
            ensure_parent_dir(out_path)
            dataset.to_json(out_path)
                        

        elif(self.dataset == 'PersonaChat'):

            datasetpath = '../../Datasets/Raw_data/' + self.dataset + '/train_hh.txt' 
            
            print("path",datasetpath)
            contexts = []
            responses = [] 
            with open(datasetpath) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if("__SILENCE__" in obj["context"]):
                        continue
                    c = obj["context"].replace('__p1__', '__p2__').split("__p2__") 
                    for i in range(len(c)-1, -1, -1): 
                        if len(c[i]) == 0:
                            del c[i]

                    if(len(c) == 1):
                        contexts.append(c[0])
                        responses.append(obj["response"])
                    else:
                        for i in range(1, len(c)):
                            contexts.append("|".join(c[:i])) 
                            responses.append(c[i])
                        contexts.append("|".join(c))
                        responses.append(obj["response"]) 


            DF_LISTS = pd.DataFrame(
                {'context': contexts,
                'response': responses,
                'label': ["Safe"] * len(contexts),
                'category': ["benign"] * len(contexts),
                'implicit': ["no"] * len(contexts),
                'source': ['PersonaChat'] * len(contexts)
                })

            def add_index_dataset_label_columns(example, idx):
                # Add index and dataset name as new columns
                example["index"] = f"{idx}_{example['label']}_{example['source']}"
                
                return example

            dataset = Dataset.from_pandas(DF_LISTS)

            dataset = dataset.train_test_split(test_size=0.20, shuffle=True, seed=42)

            # get only train split from the dataset as new dataset

            dataset_train = dataset['train']

            dataset_train = dataset_train.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

            dataset_test = dataset['test']
    
            dataset_test = dataset_test.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

            # select 60000 samples from the dataset
            dataset_train1 = dataset_train.select(range(75000)).shuffle(seed=42)
            # dataset_train1_2 = dataset_train.select(range(60000, 75000))
            # dataset_train1 = concatenate_datasets([dataset_train_11, dataset_train1_2])

            dataset_train2 = dataset_train.select(range(75000,len(dataset_train)))

            benign_dir = f"../../Datasets/Benign/{self.dataset}"
            ensure_parent_dir(f"{benign_dir}/dataset.csv")
            dataset_train1.to_csv(f"{benign_dir}/dataset.csv")

            dataset_train2.to_csv(f"{benign_dir}/dataset1.csv")


            dataset_test = dataset_test.train_test_split(test_size=0.50, shuffle=True, seed=42)

            dataset_val = dataset_test['train']

            dataset_test = dataset_test['test']

            # print(dataset_train)

            # dataset_train.to_csv("../../Datasets/Benign/" + self.dataset + '/dataset.csv')

            storepath = f"../../Datasets/Processed_datasets/{self.dataset}/test.json"
            ensure_parent_dir(storepath)
            dataset_test.to_json(storepath)

            dataset_test.to_csv(f"{benign_dir}/test_dataset.csv")


            storepath = f"../../Datasets/Processed_datasets/{self.dataset}/val.json"
            ensure_parent_dir(storepath)
            dataset_val.to_json(storepath)  

            dataset_val.to_csv(f"{benign_dir}/val_dataset.csv")

        elif(self.dataset == 'DailyDialog'):

            datasetpath = '../../Datasets/Raw_data/' + self.dataset + '/train.txt' 
            print("path",datasetpath)
            contexts = []
            responses = [] 
            with open(datasetpath) as f:
                for line in f:
                    c = line.strip().split("__eou__")
                    for i in range(len(c)-1, -1, -1): 
                        if len(c[i]) == 0:
                            del c[i]

                    if(len(c) == 1):   
                        continue
                    else:
                        for i in range(1, len(c)):
                            contexts.append("|".join(c[:i])) 
                            responses.append(c[i])
            
            DF_LISTS = pd.DataFrame(
                {'context': contexts,
                'response': responses,
                'label': ["Safe"] * len(contexts),
                'category': ["benign"] * len(contexts),
                'implicit': ["no"] * len(contexts),
                'source': ['DailyDialog'] * len(contexts)
                })
                
            def add_index_dataset_label_columns(example, idx):
                # Add index and dataset name as new columns
                example["index"] = f"{idx}_{example['label']}_{example['source']}"
                
                return example

            dataset = Dataset.from_pandas(DF_LISTS)

            dataset = dataset.train_test_split(test_size=0.20, shuffle=True, seed=42)

            # get only train split from the dataset as new dataset

            dataset_train = dataset['train']

            dataset_train = dataset_train.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

            dataset_test = dataset['test']
    
            dataset_test = dataset_test.map(lambda example, idx: add_index_dataset_label_columns(example, idx), with_indices=True)

            # select 60000 samples from the dataset
            dataset_train1 = dataset_train.select(range(60000))

            dataset_train2 = dataset_train.select(range(60000,len(dataset_train)))

            benign_dir = f"../../Datasets/Benign/{self.dataset}"
            ensure_parent_dir(f"{benign_dir}/dataset.csv")
            dataset_train1.to_csv(f"{benign_dir}/dataset.csv")

            dataset_train2.to_csv(f"{benign_dir}/dataset1.csv")


            dataset_test = dataset_test.train_test_split(test_size=0.50, shuffle=True, seed=42)

            dataset_val = dataset_test['train']

            dataset_test = dataset_test['test']

            # print(dataset_train)

            # dataset_train.to_csv("../../Datasets/Benign/" + self.dataset + '/dataset.csv')

            storepath = f"../../Datasets/Processed_datasets/{self.dataset}/test.json"
            ensure_parent_dir(storepath)
            dataset_test.to_json(storepath)

            dataset_test.to_csv(f"{benign_dir}/test_dataset.csv")


            storepath = f"../../Datasets/Processed_datasets/{self.dataset}/val.json"
            ensure_parent_dir(storepath)
            dataset_val.to_json(storepath)  

            dataset_val.to_csv(f"{benign_dir}/val_dataset.csv")



        
if(__name__ == "__main__"):
    main()    
