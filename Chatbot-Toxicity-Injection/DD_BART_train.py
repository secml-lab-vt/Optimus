import numpy as np
import random
import torch
import time
import os
import sys
from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
# from simulation import DBL_sim, seed_everything 
from learningAgent import LearningAgent 
from pipeline import Pipeline
import json
import os 
import matplotlib.pyplot as plt
import toxic_data
import pandas as pd
import argparse
from sklearn.metrics import classification_report, f1_score
from friendlyAgent import FriendlyAgent
from toxicClassifier import ToxicClassifier


device = "cuda:1"
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def preprocess_data(filename, dataset_name, max_n=-1):
    contexts = []
    responses = [] 
    if dataset_name == "personachat": 
        with open(filename) as f:
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

    elif dataset_name == "dailydialog":
        with open(filename) as f:
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

    if(max_n != -1):
        contexts = contexts[:max_n]
        responses = responses[:max_n]

    return contexts, responses

def get_list_of_tuples(contexts, responses):
    return [(contexts[i], responses[i]) for i in range(len(contexts))]


def train_full_model(model_file,lr,epochs,dataset_name):
    victim_args = {'model_type': 'BART',
                        'model_file': model_file,
                        'batch_size': 8, 
                        'epochs': epochs,
                        'lr': lr,
                        'decode_method': "meena_cutlcs"}
    victim_args['defense'] = 'none' 

    # victim_args['ppl_log_file'] = pipe.ppl_log

    victim = LearningAgent(victim_args, args.pri_dev, args.sec_dev)

    filename = f"./data/Daily_Dialog/dialogues_train.txt"

    
    
    contexts, responses = preprocess_data(filename, dataset_name)

    # zip contexts, responses into list of tuples
    pairs = get_list_of_tuples(contexts, responses)

    # filtered_pairs #list of tuples
    filtered_flags = ["none"] * len(contexts)

    victim.train(pairs, filtered_flags)  
    
    victim.save_model(f"./saves/base/DDBART-1e-6_{epochs}")  



if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description='Run an indiscriminate poisoning attack')
    parser.add_argument('-pri_dev', help='device for victim language model and training', default="cuda:0", type=str)
    parser.add_argument('-sec_dev', help='device for friendly language model and classifiers', default="cuda:1", type=str) 
    args = parser.parse_args()
    

    dataset_name = "dailydialog" 

    VICTIM_MODEL = "DD-BART"


    if VICTIM_MODEL == "DD-BART":
        MODEL_FILE = 'facebook/bart-base'

    lr = 1e-6

    epochs = 15


    train_full_model(MODEL_FILE, lr, epochs, dataset_name)








