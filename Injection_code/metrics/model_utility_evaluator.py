
import argparse
from TrainingAgent_final import TrainingAgent
from DataFactory import DataFactory
import numpy as np
import re
import random as r
import datetime
import os
from datasets import load_dataset,concatenate_datasets
from accelerate import Accelerator
import torch
from glob import glob
import pandas as pd
import json
import math
from evaluate import load


def main():
    parser = argparse.ArgumentParser(description='Arguments for train/val/inject_eval a Chatbot')
    parser.add_argument('pri_dev', help='device for chatbot inference and training')
    parser.add_argument('--chatbot', help='name of chatbot model', nargs='?', default=-1)
    parser.add_argument('--injection', help='True/False',nargs='?', default=True)
    parser.add_argument('--percentage', help='Enter a percentage between 0 and 1',nargs='?', default=0.3)
    parser.add_argument('--heal',help='True/False',nargs='?', default= "False")
    parser.add_argument('--heal_percentage',help='Enter a percentage between 0 and 1',nargs='?', default=0)
    parser.add_argument('--benign_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--toxic_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--healing_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--filter1', help='True or false',nargs='?', default="N")
    parser.add_argument('--model_vers', help="Model and version", type=str, choices=[
    "facebook_opt-iml-1.3b",
    "facebook_opt-iml-30b",
    "google_flan-t5-base",
    "google_flan-t5-large",
    "google_flan-t5-small",
    "google_flan-t5-xl",
    "google_flan-t5-xxl",
    "lmsys_vicuna-13b-v1.1",
    "lmsys_vicuna-13b-v1.3",
    "lmsys_vicuna-33b-v1.3",
    "lmsys_vicuna-7b-v1.1",
    "lmsys_vicuna-7b-v1.3",
    "meta-llama_Llama-2-13b-chat-hf",
    "meta-llama_Llama-2-13b-hf",
    "meta-llama_Llama-2-7b-chat-hf",
    "meta-llama_Llama-2-7b-hf",
    "tiiuae_falcon-40b-instruct",
    "tiiuae_falcon-7b-instruct",
    "yahma_llama-7b-hf"], default=None)

    parser.add_argument("--category", help="Which category to use", type=str, choices=["1","2"], default=None)
    parser.add_argument("--model_util", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument("--benign_filter", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument('--use_model',help='enter specific model name',nargs='?', default= "N")
    parser.add_argument("--eval_type", help="eval type", required=True, choices=['mauve', 'ppl', "fbd", "all"])
    parser.add_argument("--gather", help="command to gather mauve scores", action='store_true')
    parser.add_argument('--model_epoch',help='enter specific eval dataset',nargs='?', default= "20")
    parser.add_argument('--seed',help='enter specific eval dataset',nargs='?', default= 42)
    parser.add_argument("--uuid", help="uuid for the seeds")

    args = parser.parse_args()

    pipe = Pipeline(args)

    if args.gather:
        pipe.gather(args.eval_type)
    else:
        if args.eval_type == "all" or args.eval_type == "mauve":
            pipe.mauve_eval()
        if args.eval_type == "all" or args.eval_type == "ppl":
            pipe.ppl_eval()
        if args.eval_type == "all" or args.eval_type == "fbd":
            pipe.fbd_eval()


class Pipeline():
    def __init__(self, args):
        self.set_up(args)

    def set_up(self, args):
        self.device = torch.device(args.pri_dev if torch.cuda.is_available() else "cpu")
        self.chatbot = args.chatbot
        self.cuda_number = int(args.pri_dev.split(":")[-1])
        self.injection = args.injection
        self.benign_dataset = args.benign_dataset
        self.toxic_dataset = args.toxic_dataset
        self.heal = args.heal
        self.healing_dataset = args.healing_dataset
        self.filter1 = args.filter1
        self.uuid = args.uuid
        self.category = args.category
        self.model_name = args.model_vers
        if self.injection == 'True':
            self.percentage = args.percentage
        else:
            self.percentage = 0

        if self.heal == 'True':
            self.heal_percentage = args.heal_percentage
        else:
            self.heal_percentage = 0

        if args.eval_type == "all" or args.eval_type == "mauve":
            self.mauve = load('mauve')
        if args.eval_type == "all" or args.eval_type == "ppl":
            ## init ppl model
            pass
        if args.eval_type == "all" or args.eval_type == "fbd":
            ## init fbd model
            pass
        
        self.args = args
        self.model_epoch = args.model_epoch
        self.seed = int(args.seed)

        self.ts = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")

        self.common_path_true = f"../../Models/Custom/model_runs/{self.chatbot}/{self.model_name}_{self.benign_dataset}_{self.toxic_dataset}_{self.healing_dataset}_{self.injection}_{self.percentage}_{self.heal}_{self.heal_percentage}_True_True"
        self.common_path_false = f"../../Models/Custom/model_runs/{self.chatbot}/{self.model_name}_{self.benign_dataset}_{self.toxic_dataset}_{self.healing_dataset}_{self.injection}_{self.percentage}_{self.heal}_{self.heal_percentage}_True_False"

        self.run_folder_bf_true = None

        for uuid_fullpath in glob(self.common_path_true + "/*"):
            self.outside_seed_path_true = uuid_fullpath
            self.run_folder_bf_true = uuid_fullpath+f"/seed_{self.seed}"
            print("RUN FOLDER TRUE: ", self.run_folder_bf_true)

        self.metrics_bf_true_path = self.outside_seed_path_true
    
        
        self.run_folder_bf_false = None
        for uuid_fullpath in glob(self.common_path_false + "/*"):
            self.outside_seed_path_false = uuid_fullpath
            self.run_folder_bf_false = uuid_fullpath + f"/seed_{self.seed}"
        
        self.metrics_bf_false_path = self.outside_seed_path_false

        assert self.run_folder_bf_true != None and self.run_folder_bf_false != None

        self.metrics_file_path_bf_false_base = f'{self.run_folder_bf_false}'

        for inject_eval_file_fullpath in glob(self.run_folder_bf_true+"/inject_eval*"):
            self.run_folder_bf_true_eval_json = inject_eval_file_fullpath + f"/Injected_evaluation_{args.model_epoch}.json"
            self.metrics_file_path_bf_true_base = f'{inject_eval_file_fullpath}'
            break
        for inject_eval_file_fullpath in glob(self.run_folder_bf_false + "/inject_eval*"):
            self.run_folder_bf_false_eval_json = inject_eval_file_fullpath+ f"/Injected_evaluation_{args.model_epoch}.json"
            self.metrics_file_path_bf_false_base = f'{inject_eval_file_fullpath}'
            break
        # self.run_folder_bf_true_eval_json = f'{self.run_folder_bf_true}/inject_eval_*/Injected_evalutation_{args.model_epoch}.json'
        # self.run_folder_bf_false_eval_json = f'{self.run_folder_bf_false}/inject_eval_*/Injected_evalutation_{args.model_epoch}.json'

               
        print("writing to run config and run metrics files..")
        # self.metrics_file_bf_true = open(self.metrics_file_path_bf_true, "a")
        # self.metrics_file_bf_false = open(self.metrics_file_path_bf_false, "a")

        # attrs = vars(self)
        # self.metrics_file_bf_true.write("Metrics file:")
        # self.metrics_file_bf_true.write("\n--------------------------------\n")
        # self.metrics_file_bf_true.write("run_date:"+(self.ts)+"\n")
        # self.metrics_file_bf_true.write('\n'.join("%s: %s" % item for item in attrs.items()))
        # self.metrics_file_bf_true.write("\n--------------------------------\n")

        # self.metrics_file_bf_false.write("Metrics file:")
        # self.metrics_file_bf_false.write("\n--------------------------------\n")
        # self.metrics_file_bf_false.write("run_date:"+(self.ts)+"\n")
        # self.metrics_file_bf_false.write('\n'.join("%s: %s" % item for item in attrs.items()))
        # self.metrics_file_bf_false.write("\n--------------------------------\n")

    def gather(self, eval_type):
        bf_true_mauve_eval = self.metrics_bf_true_path + "/mauve_metrics.txt"
        
        ## both metrics files contain the same data
        # bf_false_mauve_eval = self.metrics_bf_true_path+"/mauve_metrics.txt"
        print("fetching stats from :", bf_true_mauve_eval)
        with open(bf_true_mauve_eval, "r") as file:
            entries = file.readlines()

            bf_trues = [i for i in entries if i.split(":")[1] == "bf_true"]
            bf_false = [i for i in entries if i.split(":")[1] == "bf_false"]

    def ppl_eval(self):

        perplexity_false = []
        for seed_metric in glob(self.outside_seed_path_false+"/seed_*/inject_eval_*/Metrics.txt"):
            print(seed_metric)
            with open(seed_metric, "r") as metric_file:
                lines = metric_file.readlines()
                for l in lines:
                    if "Validation Perplexity" in l:
                        match = re.search(r"Validation Perplexity: (\d+\.\d+)", l)
                        perplexity_false.append(float(match.group(1)))

        perplexity_av_false = float(sum(perplexity_false) / len(perplexity_false))
        perplexity_false.append(perplexity_av_false)

        perplexity_true = []
        for seed_metric in glob(self.outside_seed_path_true+"/seed_*/inject_eval_*/Metrics.txt"):
            print(seed_metric)
            with open(seed_metric, "r") as metric_file:
                lines = metric_file.readlines()
                for l in lines:
                    if "Validation Perplexity" in l:
                        match = re.search(r"Validation Perplexity: (\d+\.\d+)", l)
                        perplexity_true.append(float(match.group(1)))

        perplexity_av_true = float(sum(perplexity_true) / len(perplexity_true))
        perplexity_true.append(perplexity_av_true)

        with open(self.metrics_bf_true_path+"/ppl_metrics.txt", "a+") as metrics_bf_true, open(self.metrics_bf_false_path+"/ppl_metrics.txt", "a+") as metrics_bf_false:

            # metrics_bf_true.write("Metrics file:")
            # metrics_bf_true.write("\n--------------------------------\n")
            # metrics_bf_true.write("run_date:"+(self.ts)+"\n")
            metrics_bf_false.write(f"{self.chatbot}:bf_true:{self.run_folder_bf_true.split(self.chatbot)[-1]}:{perplexity_true}\n")
            metrics_bf_false.write(f"{self.chatbot}:bf_false:{self.run_folder_bf_false.split(self.chatbot)[-1]}:{perplexity_false}\n")
            # metrics_bf_true.write("\n--------------------------------\n")

            # metrics_bf_false.write("Metrics file:")
            # metrics_bf_false.write("\n--------------------------------\n")
            # metrics_bf_false.write("run_date:"+(self.ts)+"\n")
            metrics_bf_true.write(f"{self.chatbot}:bf_true:{self.run_folder_bf_true.split(self.chatbot)[-1]}:{perplexity_true}\n")
            metrics_bf_true.write(f"{self.chatbot}:bf_false:{self.run_folder_bf_false.split(self.chatbot)[-1]}:{perplexity_false}\n")
            # metrics_bf_false.write("\n--------------------------------\n")

        
        print("done")


    def mauve_eval(self):

        ## json files where injected_evaluation_epoch.json is located in self.run_folder_bf_x_eval_json
        # print(self.run_folder_bf_true_eval_json)
        # print(self.run_folder_bf_false_eval_json)

        bf_true_data = []
        bf_false_data = []

        with open(self.run_folder_bf_true_eval_json, "r") as bf_true, open(self.run_folder_bf_false_eval_json, "r") as bf_false:

            for line in bf_true:
                bf_true_data.append(json.loads(line))
            for line in bf_false:
                bf_false_data.append(json.loads(line))

        bf_true_vals_response = [bf_true_data[i]['response'] for i in range(len(bf_true_data)) if bf_true_data[i]['label'] == 'benign']
        bf_true_vals_ground = [bf_true_data[i]['responses'] for i in range(len(bf_true_data)) if bf_true_data[i]['label'] == 'benign']

        bf_false_vals_response = [bf_false_data[i]['response'] for i in range(len(bf_false_data))if bf_false_data[i]['label'] == 'benign']
        bf_false_vals_ground = [bf_false_data[i]['responses'] for i in range(len(bf_false_data))if bf_false_data[i]['label'] == 'benign']
        

        mauve_results_true = self.mauve.compute(
            predictions=bf_true_vals_response, 
            references=bf_true_vals_ground, 
            verbose=False, 
            seed=42, 
            featurize_model_name="gpt2-large", 
            mauve_scaling_factor=1,
            device_id=self.cuda_number)

        mauve_results_false = self.mauve.compute(
            predictions=bf_false_vals_response, 
            references=bf_false_vals_ground, 
            verbose=False, 
            seed=42, 
            featurize_model_name="gpt2-large", 
            mauve_scaling_factor=1,
            device_id=self.cuda_number)

        mauve_score_true = mauve_results_true.mauve
        mauve_score_false = mauve_results_false.mauve

        print(mauve_score_true, mauve_score_false, abs(mauve_score_true - mauve_score_false))
    
        df_bf_true = pd.DataFrame(bf_true_data)
        df_bf_false = pd.DataFrame(bf_false_data)

        # with open(self.run_folder_bf_true_eval_json, "w") as bf_true, open(self.run_folder_bf_false_eval_json, "w") as bf_false:
        #     for data in bf_true_data:
        #         data["mauve_score"] = mauve_score_true
        #         bf_true.write(json.dumps(data) + '\n')
        #     for data in bf_false_data:
        #         data["mauve_score"] = mauve_score_false
        #         bf_false.write(json.dumps(data) + '\n')

        
        ## metrics files
        # metric_path_true = self.metrics_file_path_bf_true_base + "/Metrics_mauve.txt"
        # metrics_bf_true = open(metric_path_true, "w")
        # metric_path_false = self.metrics_file_path_bf_false_base + "/Metrics_mauve.txt"
        # metrics_bf_false = open(metric_path_false, "w")

        with open(self.metrics_bf_true_path+"/mauve_metrics.txt", "a+") as metrics_bf_true, open(self.metrics_bf_false_path+"/mauve_metrics.txt", "a+") as metrics_bf_false:

            # metrics_bf_true.write("Metrics file:")
            # metrics_bf_true.write("\n--------------------------------\n")
            # metrics_bf_true.write("run_date:"+(self.ts)+"\n")
            metrics_bf_true.write(f"{self.chatbot}:bf_true:{self.run_folder_bf_true.split(self.chatbot)[-1]}:{mauve_score_true}\n")
            metrics_bf_true.write(f"{self.chatbot}:bf_false:{self.run_folder_bf_false.split(self.chatbot)[-1]}:{mauve_score_false}\n")
            # metrics_bf_true.write("\n--------------------------------\n")

            # metrics_bf_false.write("Metrics file:")
            # metrics_bf_false.write("\n--------------------------------\n")
            # metrics_bf_false.write("run_date:"+(self.ts)+"\n")
            metrics_bf_false.write(f"{self.chatbot}:bf_true:{self.run_folder_bf_true.split(self.chatbot)[-1]}:{mauve_score_true}\n")
            metrics_bf_false.write(f"{self.chatbot}:bf_false:{self.run_folder_bf_false.split(self.chatbot)[-1]}:{mauve_score_false}\n")
            # metrics_bf_false.write("\n--------------------------------\n")

        
        print("done")


main()    
