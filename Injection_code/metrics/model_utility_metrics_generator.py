import os
import re
import json
import numpy as np
import pandas as pd
from evaluate import load
import csv


def list_directories():

    chatbots = ['DD-BART']


    headers = ["chatbot", "benign_dataset", "toxic_dataset", "injection_flag", "injection_percentage", "category", "model_vers", "filter", "model_util", "benign_filter", "heal_flag", "healing_dataset", "heal_percentage","mauve_score","perplexity","TPR","FPR"]

    with open("Model_utility_metrics1.csv", "w", newline='') as file:
        csvwriter = csv.writer(file)

        csvwriter.writerow(headers)

        for chatbot in chatbots:

            root_directory = f'../../Models/Custom/model_runs/{chatbot}/'

            subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
            
            print(f"Subdirectories in {root_directory}:")

            mauve = load('mauve')

            for subdir in subdirectories:
                model_run = os.path.join(root_directory, subdir)
                row = []
                filename_data_all = model_run.split(chatbot + "/")[-1]
                for uid in os.listdir(os.path.join(root_directory, subdir)):
                    perplexity_true = []
                    mauve_score_true = []
                    TPR = []
                    FPR = []
                    # print(os.path.join(root_directory, subdir, uid))
                    for seed in os.listdir(os.path.join(root_directory, subdir, uid)):
                        # print(os.path.join(root_directory, subdir, uid, seed))
                        seed_directory = os.path.join(root_directory, subdir, uid, seed)

                        file_name = 'Metrics_train_eval_N.txt'
                        files = [f for f in os.listdir(seed_directory) if os.path.isfile(os.path.join(seed_directory, f)) and f == file_name]
                        
                        # FPR: 0.017392857142857144
                        # TPR: 0.8156666666666667
                        print(os.path.join(seed_directory,files[0]))
                        with open(os.path.join(seed_directory,files[0]), "r") as metric_file:
                            lines = metric_file.readlines()
                            for l in lines:
                                if "FPR" in l:
                                    print(l)
                                    match = re.search(r'\d+\.\d+', l)
                                    if match:
                                        fpr_value = float(match.group())
                                        print(fpr_value)
                                        FPR.append(fpr_value)
                                if "TPR" in l:
                                    print(l)
                                    match = re.search(r'\d+\.\d+', l)
                                    if match:
                                        tpr_value = float(match.group())
                                        print(tpr_value)
                                        TPR.append(tpr_value)
                    
                        direc_list = [d for d in os.listdir(seed_directory) if os.path.isdir(os.path.join(seed_directory, d)) and "inject_eval" in d ]
                        # print(direc_list[0])
                        inject_directory = os.path.join(seed_directory, direc_list[0])


                        file_name = 'Injected_evaluation.json'
                        files = [f for f in os.listdir(inject_directory) if os.path.isfile(os.path.join(inject_directory, f)) and f == file_name]

                        mauve_data_list = []

                        with open(os.path.join(inject_directory,files[0]), "r") as mauve_file:

                            for line in mauve_file:
                                mauve_data_list.append(json.loads(line))

                        mauve_data_list_response = [mauve_data_list[i]['response'] for i in range(len(mauve_data_list)) if mauve_data_list[i]['label'] == 'Safe']
                        mauve_data_list_ground = [mauve_data_list[i]['responses'] for i in range(len(mauve_data_list)) if mauve_data_list[i]['label'] == 'Safe']
                        

                        mauve_results = mauve.compute(
                            predictions=mauve_data_list_response, 
                            references=mauve_data_list_ground, 
                            verbose=False, 
                            seed=42, 
                            num_buckets=100,
                            featurize_model_name="gpt2-large", 
                            mauve_scaling_factor=2,
                            device_id=0)

                        mauve_score_true.append(mauve_results.mauve)


                        file_name = 'Metrics.txt'
                        files = [f for f in os.listdir(inject_directory) if os.path.isfile(os.path.join(inject_directory, f)) and f == file_name]

                        with open(os.path.join(inject_directory,files[0]), "r") as metric_file:
                            lines = metric_file.readlines()
                            for l in lines:
                                if "Validation Perplexity" in l:
                                    match = re.search(r"Validation Perplexity: (\d+\.\d+)", l)
                                    ppl = float(match.group(1))
                                    # print(float(match.group(1)))
                                    perplexity_true.append(float(match.group(1)))
                    

                    mauve_av_true = float(sum(mauve_score_true) / len(mauve_score_true))
                    # round to 2 decimal places
                    mauve_av_true = round(mauve_av_true, 4)
                    print(f"{(os.path.join(root_directory, subdir))},{mauve_av_true}")

                    perplexity_av_true = float(sum(perplexity_true) / len(perplexity_true))
                    # round to 2 decimal places
                    perplexity_av_true = round(perplexity_av_true, 2)
                    print(f"{(os.path.join(root_directory, subdir))},{perplexity_av_true}")

                    # handle 0/0 error
                    if len(TPR) == 0:
                        TPR_av_true = 0
                    else:
                        TPR_av_true = float(sum(TPR) / len(TPR))
                        # multiply by 100 to get percentage and round to 2 decimal places
                        TPR_av_true = round(TPR_av_true * 100, 2)
                        print(f"{(os.path.join(root_directory, subdir))},{TPR}")

                    if len(FPR) == 0:  
                        FPR_av_true = 0
                    else:
                        FPR_av_true = float(sum(FPR) / len(FPR))
                        # multiply by 100 to get percentage and round to 2 decimal places
                        FPR_av_true = round(FPR_av_true * 100, 2)
                        print(f"{(os.path.join(root_directory, subdir))},{FPR}")


                    row.append(chatbot)

                    # if filename_data_all[0] == "lmsys" or filename_data_all[0] == "meta-llama":
                    #         first_entry = '_'.join(filename_data_all[:2])

                    #         rest = filename_data_all[2:]
                    #         filename_data = [first_entry]
                    #         filename_data.extend(rest)
                        
                    # else:
                    #     filename_data = filename_data_all

                    filename_data_all = filename_data_all.replace("lmsys_","lmsys-")
                    filename_data_all = filename_data_all.replace("meta-llama_","meta-llama-")
                    filename_data_all = filename_data_all.replace("Context_","Context-")
                    
                    filename_data_all = filename_data_all.split("_")

                    filename_data = filename_data_all
                        

                    model_vers = filename_data[0]
                    benign_dataset = filename_data[1]
                    toxic_dataset = filename_data[2]
                    category = "1" if "1" in toxic_dataset else "2"
                    healing_dataset = filename_data[3]
                    injection_flag = filename_data[4]
                    injection_percentage = filename_data[5]
                    heal_flag = filename_data[6]
                    heal_percentage = filename_data[7]
                    model_util = filename_data[8]
                    benign_filter = filename_data[9]
                    _filter = filename_data[10]
                    threshold = filename_data[11]

                    row.append(benign_dataset)
                    row.append(toxic_dataset)
                    row.append(injection_flag)
                    row.append(injection_percentage)
                    row.append(category)
                    row.append(model_vers)
                    row.append(_filter)
                    row.append(model_util)
                    row.append(benign_filter)
                    row.append(heal_flag)
                    row.append(healing_dataset)
                    row.append(heal_percentage)
                    row.append(threshold)
                    row.append(str(mauve_av_true))
                    row.append(str(perplexity_av_true))
                    row.append(str(TPR_av_true))
                    row.append(str(FPR_av_true))

                    csvwriter.writerow(row)

if __name__ == "__main__":

        list_directories()




