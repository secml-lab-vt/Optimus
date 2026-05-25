"""Data_processor / classifier

Purpose: Heal-augmented full focal classifier datasets (Evaluation input).

Inputs:  Benign-PersonaChat CSV, Final_Dataset / Toxic CSVs, Context_Heal/ files
Outputs: ../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_{type}_{mode}.json

Usage:   python generate_classifier_focal_full_heal.py

See:     ../README.md
"""

from datasets import load_dataset,concatenate_datasets, Dataset
from common import persona_chat_train_path, save_json, load_data

modes = ["train", "test", "val"]
types = ["Category1", "Category2"]


def process_heal_dataset(heal_dataset):
        # remove a column from the dataset
        heal_dataset = heal_dataset.remove_columns(["response"])
        # rename a column from the dataset
        heal_dataset = heal_dataset.rename_column("heal_generated", "response")

        return heal_dataset

for mode in modes:
        for type in types:
                #------------------------------------------------------------------------------------------------------------
                if mode == "train":
                        if type == "Category1":
                                counter1 = 40000
                                counter2 = 12000
                                counter3 = 12000

                                heal_counter1 = 12000
                                heal_counter2 = 3600
                                heal_counter3 = 3600

                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_train.json'
                                
                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/category_1_Benign-PersonaChat_heal_dataset.csv"
                                heal_safe_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category1_train_heal_dataset.csv"
                                heal_unsafe_toxic = "../../Datasets/Classifier/Context_Heal/category_1_Toxic-Category1_heal_dataset.csv"

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")
                                

                                heal_final_dataset = load_data(heal_PersonaChat_path)
                                heal_final_dataset = process_heal_dataset(heal_final_dataset)
                                heal_final_dataset1 = load_data(heal_safe_toxic)
                                heal_final_dataset1 = process_heal_dataset(heal_final_dataset1)
                                heal_final_dataset2 = load_data(heal_unsafe_toxic)
                                heal_final_dataset2 = process_heal_dataset(heal_final_dataset2)


                        if type == "Category2":
                                counter1 = 5500
                                counter2 = 2200
                                counter3 = 2200

                                heal_counter1 = 1650
                                heal_counter2 = 660
                                heal_counter3 = 660

                                PersonaChat_path = persona_chat_train_path()
                                Toxic_path = '../../Datasets/Toxic/Category2/dataset.csv'
                                Toxic_path1 = '../../Datasets/Processed_datasets/Final_Dataset/Category2_train.json'

                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/category_2_Benign-PersonaChat_heal_dataset.csv"
                                heal_safe_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category2_train_heal_dataset.csv"
                                heal_unsafe_toxic = "../../Datasets/Classifier/Context_Heal/category_2_Toxic-Category2_heal_dataset.csv"

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path1).filter(lambda x: x['label'] == "Safe")
        
                                heal_final_dataset = load_data(heal_PersonaChat_path)
                                heal_final_dataset = process_heal_dataset(heal_final_dataset)
                                heal_final_dataset1 = load_data(heal_safe_toxic)
                                heal_final_dataset1 = process_heal_dataset(heal_final_dataset1)
                                heal_final_dataset2 = load_data(heal_unsafe_toxic)
                                heal_final_dataset2 = process_heal_dataset(heal_final_dataset2)

                #------------------------------------------------------------------------------------------------------------
                if mode == "test":
                        if type == "Category1":
                                counter1 = 1500
                                counter2 = 1500
                                counter3 = 1500
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_test.json'

                                heal_counter1 = 450
                                heal_counter2 = 450
                                heal_counter3 = 450

                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/Benign_Category1_test_heal_dataset.csv"
                                heal_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category1_test_heal_dataset.csv"


                        if type == "Category2":
                                counter1 = 300
                                counter2 = 300
                                counter3 = 300

                                heal_counter1 = 300
                                heal_counter2 = 300
                                heal_counter3 = 300


                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/Full_Focal_Classifier_Category2_test_heal_dataset.csv"
                                heal_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category2_test_heal_dataset.csv"
                

                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/test.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_test.json'

                        final_dataset = load_data(PersonaChat_path)
                        final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                        final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")

                        heal_final_dataset = load_data(heal_PersonaChat_path).filter(lambda x: x['source'] == "PersonaChat")
                        heal_final_dataset = process_heal_dataset(heal_final_dataset)
                        heal_final_dataset1 = load_data(heal_toxic).filter(lambda x: x['label'] == "Safe")
                        heal_final_dataset1 = process_heal_dataset(heal_final_dataset1)
                        heal_final_dataset2 = load_data(heal_toxic).filter(lambda x: x['label'] == "Unsafe")
                        heal_final_dataset2 = process_heal_dataset(heal_final_dataset2)

                #------------------------------------------------------------------------------------------------------------
                if mode == "val":
                        if type == "Category1":
                                counter1 = 4500
                                counter2 = 1500
                                counter3 = 1500

                                heal_counter1 = 1350
                                heal_counter2 = 450
                                heal_counter3 = 450

                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category1_val.json'

                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/Benign_Category1_val_heal_dataset.csv"
                                heal_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category1_val_heal_dataset.csv"

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")

                                heal_final_dataset = load_data(heal_PersonaChat_path).filter(lambda x: x['label'] == "Safe")
                                heal_final_dataset = process_heal_dataset(heal_final_dataset)
                                heal_final_dataset1 = load_data(heal_toxic).filter(lambda x: x['label'] == "Safe")
                                heal_final_dataset1 = process_heal_dataset(heal_final_dataset1)
                                heal_final_dataset2 = load_data(heal_toxic).filter(lambda x: x['label'] == "Unsafe")
                                heal_final_dataset2 = process_heal_dataset(heal_final_dataset2)

                        if type == "Category2":
                                counter1 = 750
                                counter2 = 300
                                counter3 = 300


                                heal_counter1 = 225
                                heal_counter2 = 90
                                heal_counter3 = 90


                                heal_PersonaChat_path = "../../Datasets/Classifier/Context_Heal/Full_Focal_Classifier_Category2_val_heal_dataset.csv"
                                heal_toxic = "../../Datasets/Classifier/Context_Heal/Toxic_Category2_val_heal_dataset.csv"
                                
                                PersonaChat_path = '../../Datasets/Processed_datasets/PersonaChat/val.json'
                                Toxic_path = '../../Datasets/Processed_datasets/Final_Dataset/Category2_val.json'

                                final_dataset = load_data(PersonaChat_path)
                                final_dataset1 = load_data(Toxic_path).filter(lambda x: x['label'] == "Unsafe")
                                final_dataset2 = load_data(Toxic_path).filter(lambda x: x['label'] == "Safe")

                                heal_final_dataset = load_data(heal_PersonaChat_path).filter(lambda x: x['source'] == "PersonaChat")
                                heal_final_dataset = process_heal_dataset(heal_final_dataset)
                                heal_final_dataset1 = load_data(heal_toxic).filter(lambda x: x['label'] == "Safe")
                                heal_final_dataset1 = process_heal_dataset(heal_final_dataset1)
                                heal_final_dataset2 = load_data(heal_toxic).filter(lambda x: x['label'] == "Unsafe")
                                heal_final_dataset2 = process_heal_dataset(heal_final_dataset2)

                #------------------------------------------------------------------------------------------------------------


                final_dataset = final_dataset.select(range(int(counter1)))
                final_dataset1 = final_dataset1.select(range(int(counter2)))
                final_dataset2 = final_dataset2.select(range(int(counter3)))

                heal_final_dataset = heal_final_dataset.select(range(int(heal_counter1)))
                heal_final_dataset1 = heal_final_dataset1.select(range(int(heal_counter2)))
                heal_final_dataset2 = heal_final_dataset2.select(range(int(heal_counter3)))


                dataset1 = concatenate_datasets([final_dataset,final_dataset1,final_dataset2]).shuffle(seed=42)

                dataset2 = concatenate_datasets([heal_final_dataset,heal_final_dataset1,heal_final_dataset2]).shuffle(seed=42)
                
                # map the labels to 'Safe'
                dataset2 = dataset2.map(lambda x: {'response': x['response'], 'label': 'Safe'}, remove_columns=['label'])
                

                dataset = concatenate_datasets([dataset1,dataset2]).shuffle(seed=42)

                print(dataset)

                save_json(dataset, f'../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_{type}_{mode}.json')
