import os
import re
import json
import numpy as np
import pandas as pd
# from evaluate import load
import csv
from fbd_helpers import calculate_fbd, calculate_prd
from bert_score import score

import os
import nltk
import time
from pathlib import Path

from grade_paths import GRADE_ROOT, MODEL_RUNS_DIR


def list_directories():

    # chatbots = ['BB400M', 'DD-BART','LLAMA2-LORA']
    chatbots = ['LLAMA2-LORA']
    # chatbots = ['DD-BART']

    headers = ["chatbot", "benign_dataset", "toxic_dataset", "injection_flag", "injection_percentage", "category", "model_vers", "filter", "model_util", "benign_filter", "heal_flag", "healing_dataset", "heal_percentage","threshold","fbd_score","prd_score" ,"perplexity","TPR","FPR","bert_score","grade_score"]
    # Model_utility_metrics_05-16-1.csv
    with open("Model_utility_metrics_jailbreak.csv", "w", newline='') as file:
        csvwriter = csv.writer(file)

        csvwriter.writerow(headers)

        for chatbot in chatbots:

            root_directory = str(MODEL_RUNS_DIR / chatbot) + os.sep

            subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
            
            print(f"Subdirectories in {root_directory}:")

            # mauve = load('mauve')

            for subdir in subdirectories:
                # print(subdir)
                # quit()

                if 'JA' not in subdir:
                    continue
                
                if 'Toxic-Category1' in subdir:
                    continue

                if '0.99'  in subdir:
                    continue

                # print(subdir)
                # quit()
                model_run = os.path.join(root_directory, subdir)
                row = []
                # print(model_run)
                # if model_run != '../../Models/Custom/model_runs/BB400M/meta-llama_Llama-2-13b-chat-hf_Benign-PersonaChat_Toxic-Category1_Prosocial_True_0.3_False_0_False_False_idea2_0.5':
                # if "_0.5_True_adv_True" not in model_run:
                #     continue
                filename_data_all = model_run.split(chatbot + "/")[-1]
                for uid in os.listdir(os.path.join(root_directory, subdir)):
                    perplexity_true = []
                    fbd_score = []
                    prd_score = []
                    bert_score = []
                    grade_score = []
                    TPR = []
                    FPR = []
                    if "Hyperparameters" in uid:
                        continue
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
                        mauve_data_list_context = [mauve_data_list[i]['context'] for i in range(len(mauve_data_list)) if mauve_data_list[i]['label'] == 'Safe']


                        # mauve_results = mauve.compute(
                        #     predictions=mauve_data_list_response, 
                        #     references=mauve_data_list_ground, 
                        #     verbose=False, 
                        #     seed=42, 
                        #     num_buckets=100,
                        #     featurize_model_name="gpt2-large", 
                        #     mauve_scaling_factor=2,
                        #     device_id=0)


                        fbd = calculate_fbd(
                                    mauve_data_list_response, 
                                    mauve_data_list_response,
                                    mauve_data_list_ground, 
                                    mauve_data_list_ground,
                                    is_chinese=0,
                                    pretrained_model_path='roberta-base', #args.pretrained_model_path,
                                    batch_size=5096,
                                    device='gpu')
                        
                        prd = calculate_prd(
                                    mauve_data_list_response, 
                                    mauve_data_list_response,
                                    mauve_data_list_ground, 
                                    mauve_data_list_ground,
                                    is_chinese=0,
                                    pretrained_model_path='roberta-base', #args.pretrained_model_path,
                                    batch_size=5096,
                                    device='gpu')
                        
                        
                        _, _, BERTScores = score(mauve_data_list_ground, mauve_data_list_response, lang="en")
                        BERTScores = BERTScores.tolist()
                        # results = bertscore.compute(predictions=mauve_data_list_response, references=mauve_data_list_ground, lang="en")

                        _,grades = get_GRADE_scores(mauve_data_list_context, mauve_data_list_response, print_progress=False, device=0)

                        
                        fbd_score.append(fbd)
                        prd_score.append(prd)
                        bert_score.append(BERTScores)
                        grade_score.append(grades)


                        # mauve_score_true.append(mauve_results.mauve)


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
                    

                    # mauve_av_true = float(sum(mauve_score_true) / len(mauve_score_true))
                    # print(fbd_score, 'fbd_Score')
                    # print(prd_score, 'prd_Score')
                    # quit()
                    fbd_score_true = float(sum(fbd_score) / len(fbd_score))
                    # round to 2 decimal places
                    fbd_score_true = round(fbd_score_true, 4)
                    
                    # print(f"{(os.path.join(root_directory, subdir))},{fbd_score_true}")

                    prd_score_true = float(sum(prd_score) / len(prd_score))
                    # round to 2 decimal places
                    prd_score_true = round(prd_score_true, 4)


                    bert_score_true = np.mean(bert_score)
                    # round to 2 decimal places
                    bert_score_true = round(bert_score_true, 4)

                    grade_score_true = np.mean(grade_score, axis=0)
                    
                    print(f"{(os.path.join(root_directory, subdir))},{prd_score_true}")
                    perplexity_av_true = float(sum(perplexity_true) / len(perplexity_true))
                    # round to 2 decimal places
                    perplexity_av_true = round(perplexity_av_true, 2)
                    # print(f"{(os.path.join(root_directory, subdir))},{perplexity_av_true}")


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
                    if len(filename_data) == 15:
                        dpo = filename_data[12]
                        adversarial = filename_data[14]
                    else:
                        dpo = "False"
                        adversarial = filename_data[13]

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
                    row.append(dpo)
                    row.append(adversarial)
                    row.append(str(fbd_score_true))
                    row.append(str(prd_score_true))
                    row.append(str(perplexity_av_true))
                    row.append(str(TPR_av_true))
                    row.append(str(FPR_av_true))
                    row.append(str(bert_score_true))
                    row.append(str(grade_score_true))
                    csvwriter.writerow(row)

def get_GRADE_scores(contexts, bot_responses, print_progress=False, device=0):
    if(type(device) == str):
        device = int(device.replace("cuda:",""))
    assert(device == 0 or device == 1)
    system_time = str(int(time.time())).strip()
    #starting_num = len(contexts)

    #check_num = r.randrange(1,10)
    #contexts += ["This is a test context"]
    #bot_responses += ["This is a test response"]

    path = str(GRADE_ROOT) + '/'
    #if os.path.exists(path + f'evaluation/infer_result/DBL/model/non_reduced_results.json'):
    #    os.remove(path + f'evaluation/infer_result/DBL/model/non_reduced_results.json')
    #if os.path.exists(path + f'evaluation/infer_result/DBL/model/reduced_results.json'):
    #    os.remove(path + f'evaluation/infer_result/DBL/model/reduced_results.json')

    os.makedirs(path + f"evaluation/eval_data/DBL/{system_time}", exist_ok=True)
    with open(path + f'evaluation/eval_data/DBL/{system_time}/human_ctx.txt', 'w+') as ctx_f:
        ctx_f.write('\n'.join([x.strip() for x in contexts]).replace('|', '|||'))
    with open(path + f'evaluation/eval_data/DBL/{system_time}/human_hyp.txt', 'w+') as hyp_f:
        hyp_f.write('\n'.join([x.strip() for x in bot_responses]))

    print("complete step")

    if(print_progress):
        os.system(f'bash "{Path(__file__).resolve().parents[1] / "tools" / "GRADE.sh"}" DBL {system_time} {device}')
    else:
        print("\nRunning GRADE scores...") # > /dev/null 2>&1
        os.system(f'bash "{Path(__file__).resolve().parents[1] / "tools" / "GRADE.sh"}" DBL {system_time} {device}')
    
    print("complete step")


    file_name = path + f'evaluation/infer_result/DBL/{system_time}/non_reduced_results.json'
    file = Path(file_name)
    counter = 0
    while(file.exists() == False or file.stat().st_size <= 75):
        time.sleep(1)
        counter += 1
        if(counter > 60): raise ValueError("GRADE results not returned!")
    time.sleep(1)

    with open(path + f'evaluation/infer_result/DBL/{system_time}/non_reduced_results.json') as json_file:
        scores = json.load(json_file)['GRADE_K2_N10_N10_eval_best_71']

    #assert(len(scores) == len(contexts))
    #scores = scores[:-check_num]
    #assert(len(scores) == starting_num)
    return scores, sum(scores) / len(scores)


if __name__ == "__main__":

        list_directories()




