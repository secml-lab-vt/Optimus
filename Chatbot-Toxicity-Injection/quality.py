from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW
import numpy as np
import json
import torch
import math
import torch.nn as nn

import sys
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import random as r

from pathlib import Path

#from trojanAgent import insert_trigger

USE_CUDA = torch.cuda.is_available()
dev = torch.device("cuda:0" if USE_CUDA else "cpu")
#print("Using device:", dev)

def run_batch(model_name, sim_type, toxic_type, rpr, defense=""):
    save_files, args = [], []
    #if(defense != ""): defense = "_" + defense
    def_str = ""
    for k in [1,2,3,4,5]:
        if(sim_type == "friendly"):
            save_files.append(f"./saves/paper/friendly/{model_name}_k-{k}")
            args.append({"sim_type": sim_type, "model_name":model_name, "k":k})
        if(sim_type == "toxic"):
            for cpr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
                save_files.append(f"./saves/paper/{sim_type}/{model_name}_{sim_type}_{toxic_type}_cpr-{cpr}_rpr-{rpr}_k-{k}")
                args.append({"sim_type": sim_type, "model_name":model_name, "toxic_type":toxic_type, "defense":defense, "cpr":cpr, "rpr":int(rpr), "k":k})
        if(sim_type == "toxic_defense"):
            for cpr in [0.3]:
                save_files.append(f"./saves/paper/{sim_type}/{model_name}_{sim_type}_{toxic_type}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}")
                args.append({"sim_type": sim_type, "model_name":model_name, "toxic_type":toxic_type, "defense":defense, "cpr":cpr, "rpr":int(rpr), "k":k})
        if(sim_type == "toxic_trojan"):
            for cpr in [0.005, 0.01, 0.05, 0.1, 0.2]:
                save_files.append(f"./saves/paper/{sim_type}{def_str}/{model_name}_{sim_type}_{toxic_type}_cpr-{cpr}_rpr-{rpr}_k-{k}")
                args.append({"sim_type": sim_type, "model_name":model_name, "toxic_type":toxic_type, "defense":defense, "cpr":cpr, "rpr":rpr, "k":k})
        if(sim_type == "toxic_trojan_defense"):
            cprs = [0.2] if (model_name == "BB400M") else [0.05]
            for cpr in cprs:
                save_files.append(f"./saves/paper/{sim_type}/{model_name}_{sim_type}_{toxic_type}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}")
                args.append({"sim_type": sim_type, "model_name":model_name, "toxic_type":toxic_type, "defense":defense, "cpr":cpr, "rpr":rpr, "k":k})

    found_files = [x for x in save_files if os.path.exists(x)]
    print(f"Save files found:{len(found_files)}/{len(save_files)}")
    if(len(found_files) < len(save_files)):
        print(save_files)
        raise ValueError()\
    #found_files = found_files[:3]
    #args = args[:3]

    if(sim_type != 'friendly'):
        log_files = [f"./results/paper/{sim_type}/qual-eval_{model_name}_{sim_type}_{toxic_type}{defense}_cpr-{x['cpr']}_rpr-{x['rpr']}_k-{x['k']}.txt" for x in args]
    else:
        log_files = [f"./results/paper/{sim_type}/qual-eval_{model_name}_{sim_type}_k-{x['k']}.txt" for x in args]

    if(model_name in ["PC-BART-7e6", "PC-BART", "DD-BART", "PC-BART-DGPT-5", "DD-BART-DGPT-5"]):
        model_type = "BART"
    elif(model_name == "BB1B"):
        model_type = "Blenderbot_large"
    elif(model_name in ["BB400M-DGPT", "BB400M-1e4", "BB400M", "BB400Mb", "BB400M_b-128_e-3"]):
        model_type = "BB400M"
    else:
        print('invalid type')
        raise ValueError()
    metrics = []
    for i, m in enumerate(found_files):
        mtr = get_all_metrics(m, model_type, decode_method="meena_cutlcs", max_n=1000, log_file=log_files[i])
        metrics.append(mtr)
        print(mtr)

    df = pd.DataFrame()
    df["file_name"] = found_files
    df["sim_type"] = [x['sim_type'] for x in args]
    df["model_name"] = [x['model_name'] for x in args]
    if(sim_type != "friendly"):
        df["cpr"] = [x['cpr'] for x in args]
        df["toxic_type"] = [x['toxic_type'] for x in args]
        df["cpr"] = [x['cpr'] for x in args]
        df["rpr"] = [x['rpr'] for x in args]
    df["k"] = [x['k'] for x in args]
    df["GRADE"] = [x['GRADE'] for x in metrics]
    df["GRUEN"] = [x['GRUEN'] for x in metrics]
    df["num_eval"] = [x['Length'] for x in metrics]
    df["unique"] = [x['diversitys'] for x in metrics]
    if(sim_type != "friendly"):
        if(defense == "0"): defense = ""
        df.to_csv(f"./results/paper/{sim_type}/qual_{model_name}_{sim_type}_{toxic_type}{defense}.csv")
    else:
        df.to_csv(f"./results/paper/{sim_type}/qual_{model_name}_{sim_type}.csv")

PRINT_GRADE = True
import argparse
def main():
    parser = argparse.ArgumentParser(description='Runs quality analysis of BASE dialog models')
    parser.add_argument('model', help='which base model to evaluate')
    args = parser.parse_args()

    if(args.model == "BB400M"):
        model_file = 'facebook/blenderbot-400M-distill'
    elif(args.model == "DD-BART"):
        model_file = './saves/base/DD-BART-BASE'

    results = open(f"./results/friendly/qual_{args.model}_base.txt", 'w+')
    m = ["GRADE","GRUEN","Diversity"]
    results.write("Model,Decode_Method," + ','.join(m) + '\n')

    log_files = None
    models = [model_file] * 5

    d = 'meena_cutlcs'
    for i, m in enumerate(models):
        log_file = None
        if(log_files != None):
            log_file = log_files[i]
        model_type = "BB400M" if ('BB' in m) else 'BART'
        model_type = "BB400M"
        metrics = get_all_metrics(m, model_type, decode_method=d, max_n=100, log_file=log_file) #max_n is the maximum number of samples to test

        print("\n\nModel File:", m)
        print("Decoding Method:", d, '\n')
        results.write(f"{m},{d}")
        for k in metrics:
            print(k + ":", metrics[k])
            results.write(f",{metrics[k]}")
        results.write("\n")
    print()
   

def get_test_pairs(c_window=3, max_n=-1):
    contexts = []
    responses = []
    with open("./data/datasets/test_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            c = obj["context"][7:].replace(" __p1__ ", '|').replace(" __p2__ ", '|').split('|')
            contexts.append('|'.join(c[-c_window:]))
            r = obj["response"]
            responses.append(r)
    if(max_n != -1):
        contexts = contexts[0:max_n]
        responses = responses[0:max_n]
    return contexts, responses

from ppl import calc_ppl_in_minibatch
def get_all_metrics(model_file, model_type, decode_method='meena_cutlcs', max_n=-1, device=dev, log_file=None, use_model=False):
    sys.path.append('./GRUEN')
    import gruen

    t0 = time.perf_counter()
    contexts, real_responses = get_test_pairs(max_n=max_n)

    if(use_model):
        bot_responses = generate_model_responses(model_file, model_type, contexts, decode_method, device)
    else:
        bot_responses = generate_responses(model_file, model_type, contexts, decode_method, device)

    diversity = len(set(bot_responses)) / len(bot_responses)
    GRADE_scores, avg_GRADE = get_GRADE_scores(contexts, bot_responses, PRINT_GRADE)
    GRUEN_scores, sub_scores = gruen.get_gruen(bot_responses)
    avg_GRUEN = sum(GRUEN_scores) / len(GRUEN_scores)

    if(log_file != None):
        f = open(log_file, "w+")
        f.write(f"ind|grade|gruen|bot_response|real_responses|contexts\n")
        for i in range(len(contexts)):
           f.write(f"{i}|{GRADE_scores[i]}|{GRUEN_scores[i]}|{bot_responses[i]}|{real_responses[i]}|{contexts[i].replace('|', '<s>')}\n")
        f.close()

    t1 = time.perf_counter()
    with open('./log.txt', 'a+') as f:
        f.write(f"{time.asctime()} - Evaluated Quality for {model_file}\n")
        f.write(f"Time Required - {t1 - t0:.2f}\n")
        f.write(f"Conversations Checked - {len(real_responses)}\n")
        f.write(f"GRADE - {avg_GRADE}\n")
        f.write(f"GRUEN - {avg_GRUEN}\n\n")

    return {"GRADE":avg_GRADE,
            "GRUEN":avg_GRUEN,
            "diversitys":diversity,
            "Length":len(bot_responses)}

from baseAgent import BaseAgent
def generate_responses(model_file, model_type, contexts, decode_method, device=dev):
    bot_responses = []
    #print("Loading:", model_file, model_type)
    agent = BaseAgent()
    agent.base_init(model_file, decode_method, model_type, device, secondary_device=None)
    if("atcon" in model_file):
        agent.atcon_init()
    for i in range(len(contexts)):
        print(f'\r{i+1}/{len(contexts)}', end='')

        output_text = agent.generate(contexts[i])
        bot_responses.append(output_text[0].replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!'))
    print()
    return bot_responses

def generate_model_responses(model_obj, model_type, contexts, decode_method, device=dev):
    bot_responses = []
    print("Loading:", model_type)
    agent = BaseAgent()
    agent.model = model_obj
    agent.decode_method = decode_method
    agent.model_type = model_type
    agent.device = device
    agent.tokenizer_init(model_type)
    #if("atcon" in model_file):
    #    agent.atcon_init()
    for i in range(len(contexts)):
        print(f'\r{i+1}/{len(contexts)}', end='')

        output_text = agent.generate(contexts[i])
        bot_responses.append(output_text[0].replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!'))
    print()
    return bot_responses

def get_BLEU_scores(contexts, real_responses, bot_responses):
    from nltk.translate.bleu_score import sentence_bleu
    scores = []
    for i in range(len(bot_responses)):
        ref = real_responses[i].replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
        can = bot_responses[i].replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
        score = sentence_bleu([ref], can)

        scores.append(score)
    avg_BLEU = sum(scores) / len(scores)
    return scores, avg_BLEU

import os
import json
import nltk
from pathlib import Path

CTI_ROOT = Path(__file__).resolve().parent
GRADE_ROOT = CTI_ROOT / "GRADE"

def get_GRADE_scores(contexts, bot_responses, print_progress=False, device=0):
    if(type(device) == str):
        device = int(device.replace("cuda:",""))
    assert(device == 0 or device == 1)
    system_time = str(int(time.time())).strip()

    path = str(GRADE_ROOT) + '/'
    os.makedirs(path + f"evaluation/eval_data/DBL/{system_time}", exist_ok=True)
    with open(path + f'evaluation/eval_data/DBL/{system_time}/human_ctx.txt', 'w+') as ctx_f:
        ctx_f.write('\n'.join([x.strip() for x in contexts]).replace('|', '|||'))
    with open(path + f'evaluation/eval_data/DBL/{system_time}/human_hyp.txt', 'w+') as hyp_f:
        hyp_f.write('\n'.join([x.strip() for x in bot_responses]))

    grade_sh = str(CTI_ROOT / "GRADE.sh")
    if(print_progress):
        os.system(f'bash "{grade_sh}" DBL {system_time} {device}')
    else:
        print("\nRunning GRADE scores...")
        os.system(f'bash "{grade_sh}" DBL {system_time} {device} > /dev/null 2>&1')

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



def BLEU2(context, response):
    ref = context.lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
    can = response.lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
    return sentence_bleu([ref], can, weights=(0.5, 0.5, 0, 0))

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

from ppl import calc_singlesample_ppl
def single_perplexity(model, x, y, device):
    #My perplexity
    with torch.no_grad():
        outputs = model(x, labels=y)
    loss = outputs['loss']
    ppl = torch.exp(loss)

    return ppl


    #Sifat Perplexity
    #return calc_singlesample_ppl(model, x, y, device)

def topk_ppl(model, x, y, k=-1):
    with torch.no_grad():
        outputs = model(x, labels=y)
    probs = torch.nn.functional.softmax(outputs['logits'], dim=2)

    right_probs = probs[:, :, y].squeeze()
    right_probs = [right_probs[i,i].item() for i in range(probs.shape[1])]

    log_probs = [np.log(x) for x in right_probs]


    leng = len(log_probs)
    if(k != -1):
        log_probs.sort()
        log_probs = log_probs[:k]
    log_sum = sum(log_probs)
    log_avg = 0 - (log_sum / len(log_probs))
    my_ppl = np.exp(log_avg)
    #normal_ppl = np.exp(outputs["loss"].item())

    return my_ppl, leng

def all_topk_ppl(model, x, y):
    with torch.no_grad():
        outputs = model(x, labels=y)
    probs = torch.nn.functional.softmax(outputs['logits'], dim=2)

    right_probs = probs[:, :, y].squeeze()
    right_probs = [right_probs[i,i].item() for i in range(probs.shape[1])]

    log_probs = [np.log(x) for x in right_probs]

    leng = len(log_probs)
    log_probs.sort()

    ppls = []
    for k in range(31):
        log_probs2 = log_probs[:k] if k > 0 else log_probs
        log_sum = sum(log_probs2)
        log_avg = 0 - (log_sum / len(log_probs2))
        my_ppl = np.exp(log_avg)
        ppls.append(my_ppl)
    return ppls


if(__name__ == "__main__"):
    main()
