"""Data_processor / classifier

Purpose: Category2 classifier dataset excluding Biased Opinion unsafe rows;
         balances benign count to match filtered unsafe count.

Inputs:  Benign-PersonaChat CSV, Category2 toxic/safe sources
Outputs: ../../Datasets/Processed_datasets/Classifier/Biased_Classifier_dataset_Category2_{mode}.json

Usage:   python generate_classifier_biased_filter.py

See:     ../README.md
"""

from datasets import load_dataset,concatenate_datasets, Dataset
from common import persona_chat_train_path, save_json, load_data

modes = ["train", "test", "val"]
types = ["Category2"]

for mode in modes:
        for type in types:
                #------------------------------------------------------------------------------------------------------------
                if mode == "train":
                        if type == "Category2":
                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Toxic/Category2/dataset.csv'
                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe" and x['category'] != "Biased Opinion")
                                counter = len(final_dataset1)
                #------------------------------------------------------------------------------------------------------------
                if mode == "test":
                        if type == "Category2":
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_test.json'
                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe" and x['category'] != "Biased Opinion")
                                counter = len(final_dataset1)
                #------------------------------------------------------------------------------------------------------------
                if mode == "val":
                        if type == "Category2":
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_val.json'
                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe" and x['category'] != "Biased Opinion")
                                counter = len(final_dataset1)

                #------------------------------------------------------------------------------------------------------------

                final_dataset = final_dataset.select(range(int(counter)))

                final_dataset1 = final_dataset1.select(range(int(counter)))

                dataset1 = concatenate_datasets([final_dataset,final_dataset1]).shuffle(seed=42)

                print(dataset1)


                save_json(dataset1, f'../../Datasets/Processed_datasets/Classifier/Biased_Classifier_dataset_{type}_{mode}.json')