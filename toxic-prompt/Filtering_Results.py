from glob import glob
import csv
import re
import pandas as pd
from tqdm import tqdm
import json
from datasets import load_dataset, concatenate_datasets, Dataset, Value
from datetime import datetime
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, roc_auc_score,roc_curve, confusion_matrix
import os

def create_model_dict2(args,directory):
    model_files = {}
    
    for fullpath in tqdm(glob(directory + "*.csv")):

        filename = fullpath.split(directory)[-1]

        match = re.search(pattern, filename)

        if not match:
            print(fullpath)
            print("ERROR IN FILENAME")
            exit(0)

        model_name = match.group(1)
        date_time = match.group(2)
        category_number = match.group(3)
        category_type = match.group(4)

        date_obj = datetime.strptime(date_time, "%Y%m%d-%H%M%S")
        ## getting most recent files
        if model_name in model_files:
            if category_number == "1":
                model_files[model_name]["category_1"].append({"filename":filename, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})
            else:
                model_files[model_name]["category_2"].append({"filename":filename, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})
        else:
            model_files[model_name] = {"category_1": [], "category_2": []}
            if category_number == "1":
                model_files[model_name]["category_1"].append({"filename":filename, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})
            else:
                model_files[model_name]["category_2"].append({"filename":filename, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})

    time_model_files = {}

    print("MODEL FILES: ", model_files)

    for model in model_files:

        print("WORKING ON CURRENT MODEL: ", model)

        if(args.adversarial == "True"):

            if "Unitary" in model or "Perspective" in model:

                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']]

                category1_files = [item for item in model_files[model]['category_1'] if 'Adv_Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Adv_Toxic-Category2' in item['filename']]
                
            elif "2" not in args.idea:
                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']]

                category1_files = [item for item in model_files[model]['category_1'] if 'Adv_Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Adv_Toxic-Category2' in item['filename']]

                print(category1_files, category2_files)

            else:
                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']] 
                
                category1_files = [item for item in model_files[model]['category_1'] if 'Adv_Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Adv_Toxic-Category2' in item['filename']]   

        else:

            if "Unitary" in model or "Perspective" in model:

                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']]

                category1_files = [item for item in model_files[model]['category_1'] if 'Toxic-Category1' in item['filename'] and 'Adv_Toxic-Category1' not in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Toxic-Category2' in item['filename'] and 'Adv_Toxic-Category2' not in item['filename']]
            elif "2" not in args.idea:
                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']]

                category1_files = [item for item in model_files[model]['category_1'] if 'Toxic-Category1' in item['filename'] and 'Adv_Toxic-Category1' not in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Toxic-Category2' in item['filename'] and 'Adv_Toxic-Category2' not in item['filename']]

            else:
                persona_chat_1_files = [item for item in model_files[model]['category_1'] if 'Benign-PersonaChat-12000' in item['filename']]
                persona_chat_2_files = [item for item in model_files[model]['category_2'] if 'Benign-PersonaChat-2500' in item['filename']] 
                
                category1_files = [item for item in model_files[model]['category_1'] if 'Toxic-Category1' in item['filename'] and 'Adv_Toxic-Category1' not in item['filename']]
                category2_files = [item for item in model_files[model]['category_2'] if 'Toxic-Category2' in item['filename'] and 'Adv_Toxic-Category2' not in item['filename']]

        print(len(persona_chat_1_files), len(persona_chat_2_files), len(category1_files), len(category2_files)) 

        if len(persona_chat_1_files) != 0:
             most_recent_persona_chat_1 = max(persona_chat_1_files, key=lambda x: x['time'])
        else:
            most_recent_persona_chat_1 = []

        if len(persona_chat_2_files) != 0:
            most_recent_persona_chat_2 = max(persona_chat_2_files, key=lambda x: x['time'])
        else:
            most_recent_persona_chat_2 = []

        if len(category1_files) != 0:
            most_recent_category1 = max(category1_files, key=lambda x: x['time'])
        else:
            most_recent_category1 = []

        if len(category2_files) != 0:
            most_recent_category2 = max(category2_files, key=lambda x: x['time'])
        else:
            most_recent_category2 = []

        # print(f"{most_recent_persona_chat_1}\n{most_recent_persona_chat_2}\n{most_recent_category1}\n{most_recent_category2}")

        time_model_files[model] = {"category_1": [], "category_2": []}

        if most_recent_category1 != [] and most_recent_persona_chat_1 != []:
            time_model_files[model]["category_1"] = [most_recent_persona_chat_1['filename'], most_recent_category1['filename']]

        if most_recent_category2 != [] and most_recent_persona_chat_2 != []:
            time_model_files[model]["category_2"] = [most_recent_persona_chat_2['filename'], most_recent_category2['filename']]

        # print(time_model_files)
        # print(time_model_files)

    return time_model_files

"""
This function returns the results from the 5 seeded dataset per input mod = [7, 13, 888, 7721, 786]
"""
def get_results(model, seeds=[7], files=None, toxic_split=30, category=None, typeinfo=None):

    print("files: ", files) 

    if len(files) == 0:
        return None
    
    if category == "category_1":
        total_sample_size = 24000
        num_toxic_samples = int(0.5 * total_sample_size)
    else:
        total_sample_size = 4400
        num_toxic_samples = int(0.5 * total_sample_size)

    benign_file = files[0] if "PersonaChat" in files[0] else files[1]
    toxic_file = files[0] if benign_file != files[0] else files[1]

    # num_toxic_samples = int( (float(toxic_split / 100)) * total_sample_size)
    num_benign_samples = total_sample_size - num_toxic_samples
    print("NUM BENIGN SAMPLES:", num_benign_samples)
    print("NUM TOXIC SAMPLES: ", num_toxic_samples)

    filepath = None
    if args.idea == "1":
        filepath= f"../Datasets/you-prompt-only/"
    if args.idea == "2":
        filepath = "../Datasets/Advanced_Detect/Best_model_train_set/"

    assert filepath is not None, "Check idea input in args"

    
    assert benign_file != toxic_file, "Error with filenames. Potentially 2 of the same type have been put together."

    print(f"USING FILES B:{benign_file}, T: {toxic_file}")

    benign_dataset = load_dataset("csv", data_files=filepath + benign_file, split="train")

    toxic_dataset = load_dataset("csv", data_files=filepath + toxic_file, split="train")

    # toxic_dataset = toxic_dataset.remove_columns(["index"])
    # benign_dataset = benign_dataset.remove_columns(["index"])
    
    seed = seeds[0]

    results = []

    # for idx, seed in tqdm(enumerate(seeds)):
    curr_result = []

    curr_benign_dataset = benign_dataset.shuffle(seed=seed).select(range(int(total_sample_size - num_toxic_samples)))
    curr_toxic_dataset = toxic_dataset.select(range(num_toxic_samples))
    curr_dataset = concatenate_datasets([curr_benign_dataset, curr_toxic_dataset])        

    if(args.idea == "1"):
        ## 0 for safe, 1 for unsafe
        positive_preds = []
        y_true = []
        for sample in curr_dataset:
            positive_preds.append(float(sample["max_prob"]))


            # if "Perspective" in benign_file or "Unitary" in benign_file:
            if sample["label"] == "Safe":
                y_true.append(0)
            else:
                y_true.append(1)
    elif(args.idea == "2"):

        X = [float(d["max_prob"]) for d in curr_dataset]
        X = np.array(X).reshape(-1,1)
        y_true = [0 if i == "Safe" else 1 for i in curr_dataset["label"]]

        n_components = 2  # Number of components (clusters)
        gmm = GaussianMixture(n_components=n_components, random_state=666)
        gmm.fit(X)
        means = gmm.means_
        toxic_idx = np.argmin(means)
        non_toxic_idx = np.argmax(means)

        positive_preds_double = gmm.predict_proba(X)
        positive_preds = [i[toxic_idx] for i in positive_preds_double]
    
    combos = [0, 1, 2]

    yhat = [1 if p >= threshold else 0 for p in positive_preds]

    TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()

    print(f"METRICS: TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
    print("CONFUSTION MATRIX: ", confusion_matrix(y_true, yhat))

    fpr = FP / (FP + TN)
    fpr = round(fpr * 100, 2)


    # auc_roc = roc_auc_score(y_true, positive_preds)

    # precision_curve, recall_curve, _ = precision_recall_curve(y_true, positive_preds)

    # auc_pr = auc(recall_curve, precision_curve)

    for combo in combos:

        if combo == 0 or combo == 1:

            precision = precision_score(y_true, yhat, pos_label=combo)

            recall = recall_score(y_true, yhat, pos_label=combo)

            f1 = f1_score(y_true, yhat, pos_label=combo)
        else:
            precision = precision_score(y_true, yhat, average="macro")

            recall = recall_score(y_true, yhat, average="macro")

            f1 = f1_score(y_true, yhat, average="macro")

        # auc_roc, auc_pr, 
        #multiple by 100 SET precision to two decimal places
        precision = round(precision * 100, 2)
        recall = round(recall * 100, 2)
        f1 = round(f1 * 100, 2)

        curr_result.append([combo, precision, recall, f1, fpr])

    results.extend(curr_result)

    return results


def write_results(model_files, outfile, typeinfo, seeds=[7]):
    # "AUC_ROC", "AUC_PRC", 
    headers = ['model', 'category_type', "type","combo","Precision", "recall", "F1", "FPR"]

    with open(outfile, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        

        for model in tqdm(model_files):
            print(f"working on {model}")

            for cat in ["category_1", "category_2"]:
                print("CATEOGRY: ", cat)



                ## Precision, recall, f1, auc_pr
                # seed2, seed3, seed4, seed5
                seed1 = get_results(model, files=model_files[model][cat], category=cat,typeinfo=typeinfo)
                # seeds = [seed1, seed2, seed3, seed4, seed5]

                # print("SEED1: ", seed1)
                # print(seed1)
                
                # auc_roc = seed1[0][0]
                # auc_prc = seed1[0][1]
                # precision = seed1[0][2]
                # recall = seed1[0][3]
                # f1 = seed1[0][4]
                # fpr = seed1[0][5]

                # remove none in seed1'
                print("SEED1: ", seed1)

                if seed1 is not None:
                    for seeder in seed1:
                        print("SEEDER: ", seeder)
                        model = model
                        cat = cat
                        typeinfo = typeinfo
                        combo = seeder[0]
                        precision = seeder[1]
                        recall = seeder[2]
                        f1 = seeder[3]
                        fpr = seeder[4]
                        
                        row = [model]
                        row.append(cat)
                        row.append(typeinfo)
                        row.append(combo)
                        row.append(precision)
                        row.append(recall)
                        row.append(f1)
                        row.append(fpr)
                        
                        csvwriter.writerow(row)

                # auc_roc = float(sum([s[0] for s in seeds]) / 5)
                # auc_prc = float(sum([s[1] for s in seeds]) / 5)
                # precision = float(sum([s[2] for s in seeds]) / 5)
                # recall = float(sum([s[3] for s in seeds]) / 5)
                # f1 = float(sum([s[4] for s in seeds]) / 5)
                # fpr = float(sum([s[5] for s in seeds]) / 5)

                # row.append(auc_roc)
                # row.append(auc_prc)


parser = argparse.ArgumentParser(description='choose path')
parser.add_argument("--idea", required=False, help="Which idea to run", choices=["1", "2"], default="1")
parser.add_argument("--adversarial", required=False, help="Which idea to run", choices=["True", "False"], default="False")
parser.add_argument("--threshold", required=False, help="Which idea to run", choices=['0.5', '0.8','0.9','0.95'], default="False")
# parser.add_argument("--type", required=True, help="Which idea to run", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11"])

# typelist = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11"]
typelist = ["0"]
args = parser.parse_args()

threshold = float(args.threshold)

directory = None
pattern = None


if args.idea == "1":



    directory = f"../Datasets/you-prompt-only/"
    pattern = r'(.+?)_(\d{8}-\d{6})_category_(\d+)_(.+?)_toxicity_scores\.csv'

    assert directory is not None, "Check idea args for directory"

    print("creating dictionary")
    model_files2 = create_model_dict2(args,directory)

    print("gathering results..")
    out_path = f"../Datasets/you-prompt-only/Results/Filtering_Results_Idea1_type_0_{args.adversarial}.csv"

       
    write_results(model_files2, out_path, "0")

   