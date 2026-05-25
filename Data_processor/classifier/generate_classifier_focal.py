"""Data_processor / classifier

Purpose: Build single combined focal BERT classifier training datasets.

Inputs:  PersonaChat benign CSV, Final_Dataset / Toxic CSVs
Outputs: ../../Datasets/Processed_datasets/Classifier/Focal_Classifier_dataset_{type}_{mode}.json

Usage:   python generate_classifier_focal.py

See:     ../README.md
"""

from datasets import load_dataset,concatenate_datasets, Dataset
from common import persona_chat_train_path, save_json, load_data

modes = ["train", "test", "val"]
types = ["Category1", "Category2"]

for mode in modes:
        for type in types:
                #------------------------------------------------------------------------------------------------------------
                if mode == "train":
                        if type == "Category1":
                                counter = 12000
                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_train.json'
                                

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")


                        if type == "Category2":
                                counter = 2200
                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Toxic/Category2/dataset.csv'
                                Toxic_path1 = '../../Datasets/Processed_datasets/Final_Dataset/Category2_train.json'

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path1).filter(lambda x: x['label'] == "Safe")


                #------------------------------------------------------------------------------------------------------------
                if mode == "test":
                        if type == "Category1":
                                counter = 1500
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_test.json'

                        if type == "Category2":
                                counter = 300
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_test.json'

                        final_dataset = load_data(PersonaChat_path)
                        final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                        final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")

                #------------------------------------------------------------------------------------------------------------
                if mode == "val":
                        if type == "Category1":
                                counter = 1500
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_val.json'


                        if type == "Category2":
                                counter = 300
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_val.json'

                        final_dataset = load_data(PersonaChat_path)
                        final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                        final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")


                #------------------------------------------------------------------------------------------------------------


                final_dataset = final_dataset.select(range(int(counter)))

                final_dataset1 = final_dataset1.select(range(int(counter)))

                final_dataset2 = final_dataset2.select(range(int(counter)))


                dataset1 = concatenate_datasets([final_dataset,final_dataset1,final_dataset2]).shuffle(seed=42)

                print(dataset1)

                save_json(dataset1, f'../../Datasets/Processed_datasets/Classifier/Focal_Classifier_dataset_{type}_{mode}.json')
