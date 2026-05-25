"""Data_processor / classifier

Purpose: Full focal classifier datasets — 40k PC train (Cat1), 5.5k PC train + 750 PC val (Cat2).

Inputs:  Benign-PersonaChat CSV, Final_Dataset / Toxic CSVs
Outputs: ../../Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_{type}_{mode}.json

Usage:   python generate_classifier_focal_full.py

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
                                counter1 = 40000
                                counter2 = 12000
                                counter3 = 12000
                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_train.json'
                                

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")


                        if type == "Category2":
                                counter1 = 5500
                                counter2 = 2200
                                counter3 = 2200
                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Toxic/Category2/dataset.csv'
                                Toxic_path1 = '../../Datasets/Processed_datasets/Final_Dataset/Category2_train.json'

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path1).filter(lambda x: x['label'] == "Safe")


                #------------------------------------------------------------------------------------------------------------
                if mode == "test":
                        if type == "Category1":
                                counter1 = 1500
                                counter2 = 1500
                                counter3 = 1500
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_test.json'

                        if type == "Category2":
                                counter1 = 300
                                counter2 = 300
                                counter3 = 300
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_test.json'

                        final_dataset = load_data(PersonaChat_path)
                        final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                        final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")

                #------------------------------------------------------------------------------------------------------------
                if mode == "val":
                        if type == "Category1":
                                counter1 = 4500
                                counter2 = 1500
                                counter3 = 1500
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_val.json'


                        if type == "Category2":
                                counter1 = 750
                                counter2 = 300
                                counter3 = 300
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_val.json'

                        final_dataset = load_data(PersonaChat_path)
                        final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                        final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")


                #------------------------------------------------------------------------------------------------------------


                final_dataset = final_dataset.select(range(int(counter1)))

                final_dataset1 = final_dataset1.select(range(int(counter2)))

                final_dataset2 = final_dataset2.select(range(int(counter3)))


                dataset1 = concatenate_datasets([final_dataset,final_dataset1,final_dataset2]).shuffle(seed=42)

                print(dataset1)

                save_json(dataset1, f'../../Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_{type}_{mode}.json')
