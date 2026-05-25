import argparse
from TrainingAgent import TrainingAgent
import numpy as np
import re
import random as r
import os
from pathlib import Path

from optimus_paths import datasets_dir, model_runs_dir
from datasets import load_dataset,concatenate_datasets
from accelerate import Accelerator
import torch
from glob import glob
from tqdm import tqdm
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from datasets import disable_caching
disable_caching()

def main():
    parser = argparse.ArgumentParser(description='Arguments for train/val/inject_eval a Chatbot')
    parser.add_argument('pri_dev', help='device for chatbot inference and training')
    parser.add_argument('--chatbot', help='name of chatbot model', nargs='?', default=-1)
    parser.add_argument('--mode', help='Must be train,eval,train_eval,interact,inject_eval' ,nargs='?', default=-1)
    parser.add_argument('--cr_num', help='Total number of CRs',nargs='?', default=12000)
    parser.add_argument('--injection', help='True/False',nargs='?', default=True)
    parser.add_argument('--percentage', help='Enter a percentage between 0 and 1',nargs='?', default=0.3)
    parser.add_argument('--heal',help='True/False',nargs='?', default= "False")
    parser.add_argument('--heal_percentage',help='Enter a percentage between 0 and 1',nargs='?', default=0)
    parser.add_argument('--benign_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--toxic_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--healing_dataset', help='Enter the dataset name in the folder',nargs='?', default= -1)
    parser.add_argument('--filter1', help='True or false',nargs='?', choices=['True','False'], default="False")
    parser.add_argument('--filter2', help='True or false',nargs='?', choices=['True','False'], default="False")
    parser.add_argument('--model_vers', help="Model and version", type=str, choices=["None",
    "Unitary",
    "Perspective",
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
    "meta-llama_Llama-2-70b-chat-hf",
    "tiiuae_falcon-40b-instruct",
    "tiiuae_falcon-7b-instruct",
    "yahma_llama-7b-hf",
    "OPENAI-moderation"], default=None)

    parser.add_argument("--category", help="Which category to use", type=str, choices=["1","2"], default=None)
    parser.add_argument("--model_util", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument("--benign_filter", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument('--use_model',help='enter specific model name',nargs='?', default= "N")
    parser.add_argument('--use_eval_dataset',help='enter specific eval dataset',nargs='?', default= "N")

    parser.add_argument('--seed',help='enter specific eval dataset',nargs='?', default= 42)
    parser.add_argument("--uuid", help="uuid for the seeds", required=True)
    parser.add_argument("--checkpoint_folder",default = "")
    parser.add_argument("--threshold",default = "0.5")
    parser.add_argument("--script_name",default = "")
    parser.add_argument("--adversarial",default = "False")




    # parser.add_argument('--rank', default = "16")
    # parser.add_argument("--alpha",default = "64")
    # parser.add_argument("--lr",default = "1e-4")

    args = parser.parse_args()

    # newly added
    if args.model_vers == "None":
        args.model_vers = None

    pipe = Pipeline(args)

    if args.benign_filter == 'True':
        assert args.injection == 'True' and args.model_util == 'True' and (args.filter1 == 'True' or args.filter2 == 'True')

    if pipe.mode in ['train','train_eval']:

        if args.filter1 == "True":
            assert args.model_vers is not None and args.category is not None, "Must have both model_vers and category when setting filter=True"
            injected_dataset = pipe.read_dataset_by_category()
        elif args.filter2 == "True":
            assert args.model_vers is not None and args.category is not None, "Must have both model_vers and category when setting filter=True"
            injected_dataset = pipe.read_dataset_by_category()
            # injected_dataset = pipe.read_dataset_by_category_idea2()
        else:
            print("read dataset called")
            injected_dataset = pipe.read_dataset()

    eval_dataset = None

    def mapper_func(example):
        if example["category"] == None:
            example["category"] = "None"

        return example

    if(args.use_eval_dataset !='N'):
        eval_dataset_path = str(datasets_dir() / "Evaluation" / f"{args.use_eval_dataset}.csv")
        eval_dataset = load_dataset("csv", data_files=eval_dataset_path,split='train').shuffle(seed=pipe.seed)
        eval_dataset = eval_dataset.map(mapper_func)

    # contexts = dataset["context"]
    # responses = dataset["response"]
    # training_flags = dataset["label"]
    # training_pairs = list(zip(contexts, responses))

    if(pipe.mode == 'train'):
        pipe.perform_training(injected_dataset)

    elif(pipe.mode == 'eval'):
        if(eval_dataset is not None):
            pipe.perform_eval(eval_dataset)
        else:
            raise Exception("provide a valid eval dataset")

    elif(pipe.mode == 'train_eval'):
        if(eval_dataset is not None):
            pipe.perform_train_eval(injected_dataset = injected_dataset,validation_dataset = eval_dataset)
        else:
            print("length of injected:",len(injected_dataset))
            pipe.perform_train_eval(injected_dataset = injected_dataset)


    elif(pipe.mode == 'interact'):
        pipe.converse()

    elif(pipe.mode == 'inject_eval'):
        if(eval_dataset is not None ):
            pipe.inject_eval(eval_dataset)
        else:
            raise Exception("provide a valid eval dataset")

    else:
        raise ValueError("Invalid phase: Use cache/train/eval/train_eval/inject_eval")


class Pipeline():
    def __init__(self, args):
        self.set_up(args)

    def set_up(self, args):
        self.device = torch.device(args.pri_dev if torch.cuda.is_available() else "cpu")
        self.chatbot = args.chatbot
        self.mode = args.mode
        self.cr_num = args.cr_num
        self.injection = args.injection
        self.benign_dataset = args.benign_dataset
        self.toxic_dataset = args.toxic_dataset
        self.heal = args.heal
        self.healing_dataset = args.healing_dataset
        self.filter1 = args.filter1
        self.filter2 = args.filter2
        self.uuid = args.uuid
        self.category = args.category
        self.model_name = args.model_vers
        self.model_util = args.model_util
        self.benign_filter = args.benign_filter
        self.threshold = float(args.threshold)
        self.script_name = args.script_name
        self.adversarial = args.adversarial

        print("threshold: ",self.threshold, type(self.threshold))

        if args.filter1 == "True" or args.filter2 == "True":
            print("getting model files")
            self.model_files = self.get_model_files()

        if self.injection == 'True':
            self.percentage = args.percentage
        else:
            self.percentage = 0

        if self.heal == 'True':
            self.heal_percentage = args.heal_percentage
        else:
            self.heal_percentage = 0

        self.args = args

        self.seed = int(args.seed)

        # acceptable_use_model_modes = ['eval']

        if args.use_model != "N":
            self.use_model = args.use_model
        else:
            self.use_model = "N"

        # if self.mode in acceptable_use_model_modes and self.use_model == "N" :
        #     raise ValueError("Model name not provided in use_model paramater: Provide Model name to evaluate")

        self.ts = str(datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")

        acceptable_chatbot = ['BB400M','DD-BART','DialoGPT','LLAMA2-LORA','MISTRAL']

        acceptable_benign = ['PersonaChat','DailyDialog','DiaSafety','facebook_opt-iml-30b_PersonaChat','lmsys_vicuna-33b-v1.3_PersonaChat','tiiuae_falcon-40b-instruct_PersonaChat']

        acceptable_toxic = ['DiaSafety','Category1','DBL_Category1','facebook_opt-iml-30b_DBL_Category1','lmsys_vicuna-33b-v1.3_DBL_Category1','tiiuae_falcon-40b-instruct_DBL_Category1','Category2']

        acceptable_heal = ['Augesc','Prosocial','Context_Heal']

        acceptable_modes = ['train','eval','train_eval','interact','inject_eval']

        # if(self.benign_dataset not in acceptable_benign):
        #     raise ValueError("Invalid benign dataset option")

        # if(self.toxic_dataset not in acceptable_toxic):
        #     raise ValueError("Invalid toxic dataset option")

        if(self.chatbot not in acceptable_chatbot):
            raise ValueError("Invalid chatbot option")

        if(self.heal == 'True'):
            if(self.healing_dataset not in acceptable_heal):
                raise ValueError("Invalid heal dataset option")


        if(self.mode not in acceptable_modes):
            raise ValueError("Invalid mode option: Use 'train','eval','train_eval','interact',inject_eval")


        idea = "2" if self.filter2 == "True" else "1"
        if self.filter2 == "False" and self.filter1 == "False":
            idea = "0"

        print("idea: ", idea)
        common_path = f"{self.chatbot}/{self.model_name}-DSS_{self.benign_dataset}_{self.toxic_dataset}_{self.healing_dataset}_{self.injection}_{self.percentage}_{self.heal}_{self.heal_percentage}_{self.model_util}_{self.benign_filter}_idea{idea}_{self.threshold}_adv_{self.adversarial}"
        
        # Use absolute paths like train_star.py to ensure consistent path resolution
        script_dir = Path(__file__).parent
        base_path = model_runs_dir()
        
        ## INJECT EVAL
        if self.mode not in ['train' , 'train_eval']:
            if self.use_model != "baseline" and self.use_model != "N":
                self.run_folder = str(base_path / common_path / self.use_model / f"seed_{self.seed}")
                self.mode_folder = f'{self.run_folder}/{self.mode}_{self.use_model}'
                if not os.path.exists(self.mode_folder):
                    os.makedirs(self.mode_folder, exist_ok=True)
                self.metrics_file_path = f'{self.mode_folder}/Metrics.txt'
                self.saved_model_folder = f'{self.run_folder}/saved_models'
                self.save_file = f'{self.saved_model_folder}/saved_model'
                print("model_file:",self.save_file)
            elif self.use_model != "N": ## if use_model == baseline
                self.run_folder = str(base_path / common_path / f"baseline_{self.uuid}" / f"seed_{self.seed}")
                if not os.path.exists(self.run_folder):
                    os.makedirs(self.run_folder, exist_ok=True)
                self.metrics_file_path = f'{self.run_folder}/Metrics_{self.mode}_{self.use_model}.txt'
            else:
                raise ValueError("provide a valid use_model name")
                    
        elif self.mode == 'train' or self.mode == 'train_eval':
            self.run_folder = str(base_path / common_path / self.uuid / f"seed_{self.seed}")
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder, exist_ok=True)
            self.metrics_file_path = f'{self.run_folder}/Metrics_{self.mode}_{self.use_model}.txt'
            self.saved_model_folder = f'{self.run_folder}/saved_models'
            if not os.path.exists(self.saved_model_folder):
                os.makedirs(self.saved_model_folder, exist_ok=True)
            self.save_file = f'{self.saved_model_folder}/saved_model'
            print("model_file:",self.save_file)

        print("metrics_file:",self.metrics_file_path)
       

        self.model_path_file = ""

        if self.chatbot == "BB400M":
            if self.use_model != "baseline" and self.use_model != "N":
                ## self.run_folder includes seed path
                self.model_path_file = f"{self.run_folder}/saved_models/saved_model"
                print("fetching trained model from: ", self.model_path_file)
            if self.category == "1":
                self.chatbot_config = {
                    'model_type': 'Blenderbot_large',
                    'model_name': 'facebook/blenderbot-400M-distill',
                    'model_file': 'facebook/blenderbot-400M-distill',
                    'batch_size':64,
                    'decode_method': 'meena_cutlcs_norep',
                    'max_epochs':4,
                    'warmup_steps':500, 
                    'lr': 1e-5,
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file
                }
            elif self.category == "2":
                self.chatbot_config = {
                    'model_type': 'Blenderbot_large',
                    'model_name': 'facebook/blenderbot-400M-distill',
                    'model_file': 'facebook/blenderbot-400M-distill',
                    'batch_size':64,
                    'decode_method': 'meena_cutlcs_norep',
                    'max_epochs':4,
                    'warmup_steps':500, 
                    'lr': 7e-6,
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file
                }      
        elif self.chatbot == "DD-BART":
            
            if self.use_model != "baseline" and self.use_model != "N":
                ## for inject eval
                ## self.run_folder includes seed path
                self.model_path_file = f"{self.run_folder}/saved_models/saved_model"
                print("fetching trained model from: ", self.model_path_file)
            if self.category == "1":
                self.chatbot_config = {
                    'model_type': 'BART',
                    'model_name': 'facebook/bart-base',
                    'model_file': 'facebook/bart-base',
                    # 'model_file': f"../Models/Custom/pretrained_models/{self.chatbot}/DD-BART-BASE",
                    'batch_size':64,
                    'max_epochs':10,
                    'warmup_steps':500, 
                    'lr': 1e-5,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file
                }
            elif self.category == "2":
                self.chatbot_config = {
                    'model_type': 'BART',
                    'model_name': 'facebook/bart-base',
                    'model_file': 'facebook/bart-base',
                    # 'model_file': f"../Models/Custom/pretrained_models/{self.chatbot}/DD-BART-BASE",
                    'batch_size':64,
                    'max_epochs':8,
                    'warmup_steps':500, 
                    'lr': 1e-5,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file
                }      
        elif self.chatbot == "LLAMA2-LORA":
            
            if self.use_model != "baseline" and self.use_model != "N":
                ## for inject eval
                ## self.run_folder includes seed path
                self.model_path_file = f"{self.run_folder}/saved_models/saved_model"
                print("fetching trained model from: ", self.model_path_file)
                
                # Detect model save format (PEFT adapter vs full model)
                # train_star.py uses OpenRLHF's strategy.save_model which may save differently
                adapter_config_path = os.path.join(self.model_path_file, "adapter_config.json")
                model_config_path = os.path.join(self.model_path_file, "config.json")
                
                if os.path.exists(adapter_config_path):
                    # PEFT adapter format - load base model + adapters
                    self.model_save_format = "peft"
                    print(f"Detected PEFT adapter format at: {self.model_path_file}")
                elif os.path.exists(model_config_path):
                    # Full model format (saved by train_star.py with strategy.save_model)
                    # Load the full model directly without PEFT
                    self.model_save_format = "full"
                    print(f"Detected full model format at: {self.model_path_file}")
                else:
                    # Check if saved_models/saved_model is empty but checkpoints exist
                    # This handles models trained with train_star.py that only saved checkpoints
                    checkpoints_dir = os.path.join(self.run_folder, "checkpoints")
                    latest_link = os.path.join(checkpoints_dir, "latest")
                    
                    if os.path.exists(latest_link):
                        # Use the 'latest' symlink to find the most recent checkpoint
                        self.model_path_file = os.path.realpath(latest_link)
                        self.model_save_format = "deepspeed_checkpoint"
                        print(f"Using DeepSpeed checkpoint from 'latest': {self.model_path_file}")
                    elif os.path.exists(checkpoints_dir):
                        # Find the most recent global_step checkpoint
                        checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) 
                                          if d.startswith("global_step") and 
                                          os.path.isdir(os.path.join(checkpoints_dir, d))]
                        if checkpoint_dirs:
                            # Sort by step number and get the latest
                            checkpoint_dirs.sort(key=lambda x: int(x.replace("global_step", "")))
                            latest_checkpoint = checkpoint_dirs[-1]
                            self.model_path_file = os.path.join(checkpoints_dir, latest_checkpoint)
                            self.model_save_format = "deepspeed_checkpoint"
                            print(f"Using latest DeepSpeed checkpoint: {self.model_path_file}")
                        else:
                            self.model_save_format = "peft"
                            print(f"No checkpoints found, defaulting to PEFT format at: {self.model_path_file}")
                    else:
                        # Default to PEFT for backward compatibility
                        self.model_save_format = "peft"
                        print(f"No config found, defaulting to PEFT format at: {self.model_path_file}")
            else:
                self.model_save_format = "peft"  # Default for training
                
            if self.category == "1":
                self.chatbot_config = {
                    'model_type': 'LLAMA2',
                    'model_name': 'meta-llama/Llama-2-7b-chat-hf',
                    'model_file': 'meta-llama/Llama-2-7b-chat-hf',
                    'batch_size':8,
                    'max_epochs':3,
                    'warmup_steps':500, 
                    'lr':2e-4,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file,
                    'rank': 64,
                    'alpha': 16,
                    'model_save_format': self.model_save_format
                }
            elif self.category == "2":
                self.chatbot_config = {
                    'model_type': 'LLAMA2',
                    'model_name': 'meta-llama/Llama-2-7b-chat-hf',
                    'model_file': 'meta-llama/Llama-2-7b-chat-hf',
                    'batch_size':4,
                    'max_epochs':3,
                    'warmup_steps':500, 
                    'lr':2e-4,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file,
                    'rank': 64,
                    'alpha': 16,
                    'model_save_format': self.model_save_format
                }
        elif self.chatbot == "MISTRAL":
            
            if self.use_model != "baseline" and self.use_model != "N":
                ## for inject eval
                ## self.run_folder includes seed path
                self.model_path_file = f"{self.run_folder}/saved_models/saved_model"
                print("fetching trained model from: ", self.model_path_file)
                
                # Detect model save format (PEFT adapter vs full model)
                adapter_config_path = os.path.join(self.model_path_file, "adapter_config.json")
                model_config_path = os.path.join(self.model_path_file, "config.json")
                
                if os.path.exists(adapter_config_path):
                    self.model_save_format = "peft"
                    print(f"Detected PEFT adapter format at: {self.model_path_file}")
                elif os.path.exists(model_config_path):
                    self.model_save_format = "full"
                    print(f"Detected full model format at: {self.model_path_file}")
                else:
                    # Check if saved_models/saved_model is empty but checkpoints exist
                    checkpoints_dir = os.path.join(self.run_folder, "checkpoints")
                    latest_link = os.path.join(checkpoints_dir, "latest")
                    
                    if os.path.exists(latest_link):
                        self.model_path_file = os.path.realpath(latest_link)
                        self.model_save_format = "deepspeed_checkpoint"
                        print(f"Using DeepSpeed checkpoint from 'latest': {self.model_path_file}")
                    elif os.path.exists(checkpoints_dir):
                        checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) 
                                          if d.startswith("global_step") and 
                                          os.path.isdir(os.path.join(checkpoints_dir, d))]
                        if checkpoint_dirs:
                            checkpoint_dirs.sort(key=lambda x: int(x.replace("global_step", "")))
                            latest_checkpoint = checkpoint_dirs[-1]
                            self.model_path_file = os.path.join(checkpoints_dir, latest_checkpoint)
                            self.model_save_format = "deepspeed_checkpoint"
                            print(f"Using latest DeepSpeed checkpoint: {self.model_path_file}")
                        else:
                            self.model_save_format = "peft"
                            print(f"No checkpoints found, defaulting to PEFT format at: {self.model_path_file}")
                    else:
                        self.model_save_format = "peft"
                        print(f"No config found, defaulting to PEFT format at: {self.model_path_file}")
            else:
                self.model_save_format = "peft"
                
            if self.category == "1":
                self.chatbot_config = {
                    'model_type': 'MISTRAL',
                    'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'model_file': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'batch_size':8,
                    'max_epochs':3,
                    'warmup_steps':500, 
                    'lr':2e-4,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file,
                    'rank': 64,
                    'alpha': 16,
                    'model_save_format': self.model_save_format
                }
            elif self.category == "2":
                self.chatbot_config = {
                    'model_type': 'MISTRAL',
                    'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'model_file': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'batch_size':8,
                    'max_epochs':3,
                    'warmup_steps':500, 
                    'lr':2e-4,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'path_file': self.model_path_file,
                    'rank': 64,
                    'alpha': 16,
                    'model_save_format': self.model_save_format
                } 

        elif self.chatbot == "DialoGPT":

            if self.use_model != "baseline" and self.use_model != "N":
                ## self.run_folder includes seed path
                self.model_path_file = f"{self.run_folder}/saved_models/saved_model"
                print("fetching trained model from: ", self.model_path_file)

            self.chatbot_config = {
                'model_type': 'DialoGPT',
                'model_name': 'microsoft/DialoGPT-medium',
                'model_file': f"microsoft/DialoGPT-medium",
                'batch_size':4,
                'max_epochs':10,
                'warmup_steps':500, 
                'lr':5e-5,
                'decode_method': 'meena_cutlcs_norep',
                'train_frac' : 0.9,
                'path_file': self.model_path_file
            }
        
        print("writing to run config and run metrics files..")
        with open(self.metrics_file_path, "w") as metrics_file:
            attrs = vars(self)
            metrics_file.write("Metrics file:")
            metrics_file.write("\n--------------------------------\n")
            metrics_file.write("run_date:"+(self.ts)+"\n")
            metrics_file.write('\n'.join("%s: %s" % item for item in attrs.items()))
            metrics_file.write("\n--------------------------------\n")


    def get_model_files_heal(self):

        heal_model_name = 'meta-llama_Llama-2-13b-chat-hf'

        time_model_files = {}

        BASE_PATH = None
        pattern = None

        BASE_PATH = str(datasets_dir() / "Context_Heal") + os.sep
        pattern = r'(.+?)_(\d{8}-\d{6})_category_(\d+)_(.+?)_heal_dataset\.csv'

        files = glob(f"{BASE_PATH}{heal_model_name}*.csv")


        model_files = {"category_1": [], "category_2": []}

        for fullpath in tqdm(files):

            filename = fullpath.split("/")[-1]

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

            if category_number == "1":
                model_files["category_1"].append({"filename":fullpath, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})
            else:
                model_files["category_2"].append({"filename":fullpath, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})

            persona_chat_1_files = [item for item in model_files['category_1'] if 'Benign-PersonaChat' in item['filename']]
            persona_chat_2_files = [item for item in model_files['category_2'] if 'Benign-PersonaChat' in item['filename']]

            category1_files = [item for item in model_files['category_1'] if 'Toxic-Category1' in item['filename']]
            category2_files = [item for item in model_files['category_2'] if 'Toxic-Category2' in item['filename']]

        # REMOVE FILES IN THESE LIST CONTAINING "JA"
        category1_files = [item for item in category1_files if 'JA' not in item['filename']]
        category2_files = [item for item in category2_files if 'JA' not in item['filename']]

        most_recent_persona_chat_1 = max(persona_chat_1_files, key=lambda x: x['time'])

        most_recent_persona_chat_2 = max(persona_chat_2_files, key=lambda x: x['time'])

        most_recent_category1 = max(category1_files, key=lambda x: x['time'])

        most_recent_category2 = max(category2_files, key=lambda x: x['time'])

        time_model_files = {"category_1": [], "category_2": []}
        time_model_files["category_1"] = [most_recent_persona_chat_1['filename'], most_recent_category1['filename']]
        time_model_files["category_2"] = [most_recent_persona_chat_2['filename'], most_recent_category2['filename']]

        return time_model_files

    def get_model_files(self):

        time_model_files = {}

        BASE_PATH = None
        pattern = None

        if "True" in self.filter1 and "False" in self.filter2:
            # BASE_PATH = "../Datasets/LM_Scores/Fewshot-no/"
            BASE_PATH = str(datasets_dir() / "LM_Scores" / "Fewshot-no" / "training_datasets") + os.sep
            pattern = r'(.+?)_(\d{8}-\d{6})_category_(\d+)_(.+?)_toxicity_scores\.csv'
        elif "False" in self.filter1 and "True" in self.filter2:
            BASE_PATH = str(datasets_dir() / "Advanced_Detect" / "Fewshot-no" / "training_datasets") + os.sep
            pattern = r'(.+?)_(\d{8}-\d{6})_category_(\d+)_(.+?)_toxicity_scores\.csv'


        files = glob(f"{BASE_PATH}{self.model_name}*.csv")


        model_files = {"category_1": [], "category_2": []}

        for fullpath in tqdm(files):

            filename = fullpath.split("/")[-1]

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
            if category_number == "1":
                model_files["category_1"].append({"filename":fullpath, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})
            else:
                model_files["category_2"].append({"filename":fullpath, "time": date_obj, "category_type": category_type, "category_number": category_number, "model_name": model_name})


        if "Unitary" in self.model_name or "Perspective" in self.model_name or "OPENAI-moderation" in self.model_name:
            persona_chat_1_files = [item for item in model_files['category_1'] if 'Benign-PersonaChat' in item['filename']]
            persona_chat_2_files = [item for item in model_files['category_2'] if 'Benign-PersonaChat' in item['filename']]

            if self.adversarial == "True":
                category1_files = [item for item in model_files['category_1'] if 'Adv-Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Adv-Toxic-Category2' in item['filename']]
            else:
                category1_files = [item for item in model_files['category_1'] if 'Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Toxic-Category2' in item['filename']]

        elif "True" in self.filter1:
            persona_chat_1_files = [item for item in model_files['category_1'] if 'Benign-PersonaChat' in item['filename']]
            persona_chat_2_files = [item for item in model_files['category_2'] if 'Benign-PersonaChat' in item['filename']]

            if self.adversarial == "True":
                category1_files = [item for item in model_files['category_1'] if 'Adv-Toxic-Category1-Idea1' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Adv-Toxic-Category2-Idea1' in item['filename']]
            else:
                category1_files = [item for item in model_files['category_1'] if 'Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Toxic-Category2' in item['filename']]

        else:
            persona_chat_1_files = [item for item in model_files['category_1'] if 'Benign-PersonaChat' in item['filename']]
            persona_chat_2_files = [item for item in model_files['category_2'] if 'Benign-PersonaChat' in item['filename']]     
            
            if self.adversarial == "True":
                category1_files = [item for item in model_files['category_1'] if 'Adv-Toxic-Category1-Idea2' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Adv-Toxic-Category2-Idea2' in item['filename']]
            else:
                category1_files = [item for item in model_files['category_1'] if 'Toxic-Category1' in item['filename']]
                category2_files = [item for item in model_files['category_2'] if 'Toxic-Category2' in item['filename']]

        # print(persona_chat_1_files, persona_chat_2_files, category1_files, category2_files)
        category1_files = [item for item in category1_files if 'JA' not in item['filename']]
        category2_files = [item for item in category2_files if 'JA' not in item['filename']]


        if(self.category == "1"):
            most_recent_persona_chat_1 = max(persona_chat_1_files, key=lambda x: x['time'])
            most_recent_category1 = max(category1_files, key=lambda x: x['time'])
            time_model_files = {"category_1": []}
            time_model_files["category_1"] = [most_recent_persona_chat_1['filename'], most_recent_category1['filename']]
        else:
            most_recent_persona_chat_2 = max(persona_chat_2_files, key=lambda x: x['time'])
            most_recent_category2 = max(category2_files, key=lambda x: x['time'])
            time_model_files = {"category_2": []}
            time_model_files["category_2"] = [most_recent_persona_chat_2['filename'], most_recent_category2['filename']]

        # print(f"{most_recent_persona_chat_1}\n{most_recent_persona_chat_2}\n{most_recent_category1}\n{most_recent_category2}")
        # print(time_model_files)
        
        return time_model_files

    def read_dataset_by_category_idea2(self):

        print("IN READ DATASET BY CATEGORY")

        self.data_folder = f'{self.run_folder}/train_data'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
        
        benign_dataset = None
        toxic_dataset = None

        ## self.model_files has all the csv files for any given model. Here we get the datasets by args.x_dataset
        ## self.model_files should be initialied in setup()
        for dataset in self.model_files[f'category_{self.category}']:
            
            if self.benign_dataset in dataset:
                benign_dataset = dataset
                print("BENIGN DATASET USED: ", benign_dataset)
            elif self.toxic_dataset in dataset:
                toxic_dataset = dataset
                print("TOXIC DATASET USED: ", toxic_dataset)

        assert benign_dataset is not None and toxic_dataset is not None, "Error in file parsing"

        Benign_Dataset = load_dataset("csv", data_files = benign_dataset, split="train")
        Benign_Dataset = Benign_Dataset.shuffle(seed=self.seed)
        Benign_data = Benign_Dataset.select(range(int(self.cr_num)))

        if self.injection == 'True':
            
            toxic_cr_num = int(int(self.cr_num) * float(self.percentage))
            Toxic_Dataset = load_dataset("csv", data_files=toxic_dataset,split='train')
            if self.category == "2":
                 Toxic_Dataset = Toxic_Dataset.select(range(int(2200)))
            # Toxic_Dataset = Toxic_Dataset.shuffle(seed=self.seed)

            print(f"INJECTING {self.percentage}")

            if self.model_util == 'True':
                print("USING MODEL UTILITY")
                Benign_data = Benign_data.select(range(int(int(self.cr_num) - toxic_cr_num)))
                dataset = Benign_data
                print("benign_cr_num: ",len(Benign_data))
                print("total_cr_num: ",len(dataset))
            else:
                Toxic_data  = Toxic_Dataset.select(range(toxic_cr_num))
                Benign_data = Benign_data.select(range(int(int(self.cr_num) - toxic_cr_num)))
                dataset = concatenate_datasets([Toxic_data, Benign_data])
                print("toxic_cr_num: ",len(Toxic_data))
                print("benign_cr_num: ",len(Benign_data))
                print("total_cr_num: ",len(dataset))



            # dataset11 = Benign_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
            # FPR = len(dataset11) / len(Benign_data)
            # print("FPR: "+str(FPR))

            # if self.model_util == 'False':
            #     dataset12 = Toxic_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
            #     TPR = len(dataset12) / len(Toxic_data)            
            #     print("TPR: "+str(TPR))
            #     dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)
            # else:
            #     if self.benign_filter == 'True':
            #         dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)
            X = [float(d["max_prob"]) for d in dataset]

            X = np.array(X).reshape(-1,1)

            y_true = [0 if i == "Safe" else 1 for i in dataset["label"]]

            n_components = 2  # Number of components (clusters)
            gmm = GaussianMixture(n_components=n_components, random_state=777)
            gmm.fit(X)
            means = gmm.means_
            toxic_idx = np.argmin(means)
            non_toxic_idx = np.argmax(means)

            positive_preds_double = gmm.predict_proba(X)

            positive_preds = [i[toxic_idx] for i in positive_preds_double]

            yhat = [1 if p >= self.threshold else 0 for p in positive_preds]

            assert len(yhat) == len(dataset["label"])

            dataset = dataset.add_column("yhat", yhat)

            if self.model_util == 'False' or (self.benign_filter == "True" and self.model_util == "True"):
                full_dataset = dataset
                dataset_toxic = full_dataset.filter(lambda example: example["yhat"] == 1)
                dataset = dataset.filter(lambda example: example["yhat"] == 0)
                
            elif self.benign_filter == "False" and self.model_util == "True":
                pass
            
            if self.model_util == 'False':

                accuracy = accuracy_score(y_true, yhat)

                print("Accuracy:", accuracy)
                
                precision = precision_score(y_true, yhat)

                recall = recall_score(y_true, yhat)

                f1 = f1_score(y_true, yhat)

                auc_roc = roc_auc_score(y_true, positive_preds)

                precision_curve, recall_curve, _ = precision_recall_curve(y_true, positive_preds)

                auc_pr = auc(recall_curve, precision_curve)

                TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()

                FPR = FP / (FP + TN)

                TPR = TP / (TP + FN)
                
                print("auc_roc: ",auc_roc)
                print("auc_pr: ",auc_pr)
                print("precision: ",precision)
                print("recall: ",recall)
                print("f1: ",f1)
                print("fpr: ",FPR)
            
            else:
                
                TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()

                FPR = FP / (FP + TN)
                
                print("fpr: ",FPR)


            self.cr_num = len(dataset)

        else:

            dataset =  Benign_data
            print("Injection False: ",len(dataset))

            # dataset11 = Benign_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
            # FPR = len(dataset11) / len(Benign_data)
            # print("FPR: "+str(FPR))
            # dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)

            X = [float(d["max_prob"]) for d in dataset]

            X = np.array(X).reshape(-1,1)

            y_true = [0 if i == "Safe" else 1 for i in dataset["label"]]

            n_components = 2  # Number of components (clusters)
            gmm = GaussianMixture(n_components=n_components, random_state=777)
            gmm.fit(X)
            means = gmm.means_
            toxic_idx = np.argmin(means)
            non_toxic_idx = np.argmax(means)

            positive_preds_double = gmm.predict_proba(X)

            positive_preds = [i[toxic_idx] for i in positive_preds_double]

            yhat = [1 if p >= self.threshold else 0 for p in positive_preds]

            assert len(yhat) == len(dataset["label"])

            dataset = dataset.add_column("yhat", yhat)

            if self.benign_filter != 'False':
                full_dataset = dataset
                dataset_toxic = full_dataset.filter(lambda example: example["yhat"] == 1)
                dataset = dataset.filter(lambda example: example["yhat"] == 0)

            if self.model_util == 'False':

                accuracy = accuracy_score(y_true, yhat)

                print("Accuracy:", accuracy)
                
                precision = precision_score(y_true, yhat)

                recall = recall_score(y_true, yhat)

                f1 = f1_score(y_true, yhat)

                auc_roc = roc_auc_score(y_true, positive_preds)

                precision_curve, recall_curve, _ = precision_recall_curve(y_true, positive_preds)

                auc_pr = auc(recall_curve, precision_curve)

                TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()

                FPR = FP / (FP + TN)

                TPR = TP / (TP + FN)
                
                print("auc_roc: ",auc_roc)
                print("auc_pr: ",auc_pr)
                print("precision: ",precision)
                print("recall: ",recall)
                print("f1: ",f1)
                print("fpr: ",FPR)
            
            else:
                
                TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()

                FPR = FP / (FP + TN)
                
                print("fpr: ",FPR)

            self.cr_num = len(dataset)

        attrs = vars(self)
        with open(self.metrics_file_path, "w") as metrics_file:
            metrics_file.write("Metrics file:")
            metrics_file.write("\n--------------------------------\n")
            metrics_file.write("run_date:"+(self.ts)+"\n")
            metrics_file.write('\n'.join("%s: %s" % item for item in attrs.items()))
            if(self.injection == 'True'):
                metrics_file.write('\n'+ "FPR: "+str(FPR))
                if self.model_util == 'False' and self.injection == 'True':
                    metrics_file.write('\n'+ "TPR: "+str(TPR))
                    metrics_file.write('\n'+ "Accuracy: "+str(accuracy))
                    metrics_file.write('\n'+ "Precision: "+str(precision))
                    metrics_file.write('\n'+ "Recall: "+str(recall))
                    metrics_file.write('\n'+ "F1: "+str(f1))
                    metrics_file.write('\n'+ "AUC_ROC: "+str(auc_roc))
                    metrics_file.write('\n'+ "AUC_PR: "+str(auc_pr))
            metrics_file.write("\n--------------------------------\n")
        
        if self.heal == 'True':
            if(self.healing_dataset in ["Augesc","Prosocial"]):
                healing_labels = ["heal"]
                healing_dataset_path = str(datasets_dir() / "Healing" / self.healing_dataset / "dataset.csv")
                total_cr_num = int(int(self.cr_num) / float(1 - float(self.heal_percentage)))
                heal_cr_num = total_cr_num * float(self.heal_percentage)
                Heal_Dataset = load_dataset("csv", data_files=healing_dataset_path,split='train')
                Heal_Dataset = Heal_Dataset.shuffle(seed=self.seed)
                Heal_data = Heal_Dataset.filter(lambda example: example["label"] in healing_labels)
                Heal_data  = Heal_data.select(range(int(heal_cr_num)))
                print("prior dataset: ",len(dataset))
                dataset = concatenate_datasets([dataset, Heal_data])
                print("heal cr num: ",len(Heal_data))
                print("total_cr_num: ",len(dataset))
                
            elif(self.healing_dataset in ["Context_Heal"]):
                self.heal_files = self.get_model_files_heal()

                print("heal_files: ",self.heal_files)

                for heal_dataset in self.heal_files[f'category_{self.category}']:
                    
                    if self.adversarial == "True":
                        
                        if "Adv-Toxic-Category1-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category1-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category1"
                        elif "Adv-Toxic-Category2-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category2-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category2"

                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)
                        
                        elif toxic_x in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                    else: 
                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)

                        elif self.toxic_dataset in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                
                assert benign_heal_dataset is not None and toxic_heal_dataset is not None, "Error in file parsing"

                combined_heal_datasets = concatenate_datasets([benign_heal_dataset, toxic_heal_dataset])

                Heal_data = combined_heal_datasets

                Heal_data = Heal_data.remove_columns('response')
                Heal_data = Heal_data.rename_column('heal_generated', 'response')
                

                column_to_match = 'index' 

                matching_values = set(dataset_toxic[column_to_match])

                Heal_data = Heal_data.filter(lambda example: example[column_to_match] in matching_values)

                dataset = concatenate_datasets([dataset, Heal_data])

                print("heal cr num: ",len(Heal_data))

                print("toxic cr num:",len(dataset_toxic))

                print("total_cr_num: ",len(dataset))

                Heal_data.to_csv(f"{self.data_folder}/heal_dataset.csv")
            

        print(dataset)

        dataset.to_csv(f"{self.data_folder}/train_dataset.csv")
        
        return dataset


    def read_dataset_by_category(self):

        print("IN READ DATASET BY CATEGORY")

        self.data_folder = f'{self.run_folder}/train_data'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
        
        benign_dataset = None
        toxic_dataset = None

        ## self.model_files has all the csv files for any given model. Here we get the datasets by args.x_dataset
        ## self.model_files should be initialied in setup()
        for dataset in self.model_files[f'category_{self.category}']:
            
            if self.benign_dataset in dataset:
                benign_dataset = dataset
                print("BENIGN DATASET USED: ", benign_dataset)
            elif self.toxic_dataset in dataset:
                toxic_dataset = dataset
                print("TOXIC DATASET USED: ", toxic_dataset)

        assert benign_dataset is not None and toxic_dataset is not None, "Error in file parsing"

        Benign_Dataset = load_dataset("csv", data_files = benign_dataset, split="train")
        Benign_Dataset = Benign_Dataset.shuffle(seed=self.seed)
        Benign_data = Benign_Dataset.select(range(int(self.cr_num)))

        if self.injection == 'True':
            
            toxic_cr_num = int(int(self.cr_num) * float(self.percentage))
            Toxic_Dataset = load_dataset("csv", data_files=toxic_dataset,split='train')
            # Toxic_Dataset = Toxic_Dataset.shuffle(seed=self.seed)
            if self.category == "2":
                 Toxic_Dataset = Toxic_Dataset.select(range(int(2200)))

            print(f"INJECTING {self.percentage}")

            if self.model_util == 'True':
                print("USING MODEL UTILITY")
                Benign_data = Benign_data.select(range(int(int(self.cr_num) - toxic_cr_num)))
                dataset = Benign_data
                print("benign_cr_num: ",len(Benign_data))
                print("total_cr_num: ",len(dataset))
            else:
                Toxic_data  = Toxic_Dataset.select(range(toxic_cr_num))
                Benign_data = Benign_data.select(range(int(int(self.cr_num) - toxic_cr_num)))
                dataset = concatenate_datasets([Toxic_data, Benign_data])
                print("toxic_cr_num: ",len(Toxic_data))
                print("benign_cr_num: ",len(Benign_data))
                print("total_cr_num: ",len(dataset))

            # dataset11 = Benign_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
            # print all column value of the 0th one with column names
            print(Benign_data[0])

            
            
            dataset11 = Benign_data.filter(lambda example: float(example["max_prob"]) >= self.threshold)
            FPR = len(dataset11) / len(Benign_data)
            print("FPR: "+str(FPR))

            if self.model_util == 'False':
                # dataset12 = Toxic_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
                dataset12 = Toxic_data.filter(lambda example: float(example["max_prob"]) >= self.threshold)
                TPR = len(dataset12) / len(Toxic_data)            
                print("TPR: "+str(TPR))
                full_dataset = dataset
                dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)
                dataset_toxic = full_dataset.filter(lambda example: float(example["max_prob"]) >= self.threshold)
            else:
                if self.benign_filter == 'True':
                    full_dataset = dataset
                    dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)
                    dataset_toxic = full_dataset.filter(lambda example: float(example["max_prob"]) >= self.threshold)

            self.cr_num = len(dataset)
        else:
            dataset =  Benign_data
            print("Injection False: ",len(dataset))
            # dataset11 = Benign_data.filter(lambda example: example["prediction"] == "Unsafe" and float(example["max_prob"]) >= self.threshold)
            dataset11 = Benign_data.filter(lambda example: float(example["max_prob"]) >= self.threshold)
            FPR = len(dataset11) / len(Benign_data)
            print("FPR: "+str(FPR))
            full_dataset = dataset
            dataset = dataset.filter(lambda example: float(example["max_prob"]) < self.threshold)
            dataset_toxic = full_dataset.filter(lambda example: float(example["max_prob"]) >= self.threshold)
            self.cr_num = len(dataset)

        attrs = vars(self)
        with open(self.metrics_file_path, "w") as metrics_file:
            metrics_file.write("Metrics file:")
            metrics_file.write("\n--------------------------------\n")
            metrics_file.write("run_date:"+(self.ts)+"\n")
            metrics_file.write('\n'.join("%s: %s" % item for item in attrs.items()))
            if(self.injection == 'True'):
                metrics_file.write('\n'+ "FPR: "+str(FPR))
                if self.model_util == 'False' and self.injection == 'True':
                    metrics_file.write('\n'+ "TPR: "+str(TPR))
            metrics_file.write("\n--------------------------------\n")
        
        if self.heal == 'True':
            if(self.healing_dataset in ["Augesc","Prosocial"]):
                healing_labels = ["heal"]
                healing_dataset_path = str(datasets_dir() / "Healing" / self.healing_dataset / "dataset.csv")
                total_cr_num = int(int(self.cr_num) / float(1 - float(self.heal_percentage)))
                heal_cr_num = total_cr_num * float(self.heal_percentage)
                Heal_Dataset = load_dataset("csv", data_files=healing_dataset_path,split='train')
                Heal_Dataset = Heal_Dataset.shuffle(seed=self.seed)
                Heal_data = Heal_Dataset.filter(lambda example: example["label"] in healing_labels)
                Heal_data  = Heal_data.select(range(int(heal_cr_num)))
                print("prior dataset: ",len(dataset))
                dataset = concatenate_datasets([dataset, Heal_data])
                print("heal cr num: ",len(Heal_data))
                print("total_cr_num: ",len(dataset))
                
            elif(self.healing_dataset in ["Context_Heal"]):

                self.heal_files = self.get_model_files_heal()

                print("heal_files: ",self.heal_files)

                for heal_dataset in self.heal_files[f'category_{self.category}']:
                    
                    if self.adversarial == "True":
                        
                        if "Adv-Toxic-Category1-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category1-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category1"
                        elif "Adv-Toxic-Category2-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category2-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category2"

                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)
                        
                        elif toxic_x in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                    else: 
                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)

                        elif self.toxic_dataset in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                
                assert benign_heal_dataset is not None and toxic_heal_dataset is not None, "Error in file parsing"

                combined_heal_datasets = concatenate_datasets([benign_heal_dataset, toxic_heal_dataset])

                Heal_data = combined_heal_datasets

                Heal_data = Heal_data.remove_columns('response')
                Heal_data = Heal_data.rename_column('heal_generated', 'response')
                
                column_to_match = 'index' 

                matching_values = set(dataset_toxic[column_to_match])

                Heal_data = Heal_data.filter(lambda example: example[column_to_match] in matching_values)

                dataset = concatenate_datasets([dataset, Heal_data])

                print("heal cr num: ",len(Heal_data))

                print("toxic cr num:",len(dataset_toxic))

                print("total_cr_num: ",len(dataset))

                Heal_data.to_csv(f"{self.data_folder}/heal_dataset.csv")
            

        print(dataset)

        dataset.to_csv(f"{self.data_folder}/train_dataset.csv")
        
        return dataset
        

    def read_dataset(self):

        print("IN READ DATASET")

        self.data_folder = f'{self.run_folder}/train_data'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)

        benign_dataset_path = str(datasets_dir() / "Benign" / self.benign_dataset / "dataset.csv")
        Benign_Dataset = load_dataset("csv", data_files=benign_dataset_path,split='train')
        Benign_Dataset = Benign_Dataset.shuffle(seed=self.seed)

        Benign_data = Benign_Dataset

        # Other_Benign_data  = Benign_data.select(range(int(self.cr_num),len(Benign_data)))
        # Other_Benign_data.to_csv(f"{self.data_folder}/other_benign_dataset.csv")
        Benign_data = Benign_data.select(range(int(self.cr_num)))

        if self.injection == 'True':

            toxic_dataset_path = str(datasets_dir() / "Toxic" / self.toxic_dataset / "dataset.csv")
            toxic_cr_num = int(int(self.cr_num) * float(self.percentage))
            Toxic_Dataset = load_dataset("csv", data_files=toxic_dataset_path,split='train')
            if self.category == "2":
                Toxic_Dataset = Toxic_Dataset.select(range(int(2200)))
            print(f"toxic dataset length: {len(Toxic_Dataset)}")

            # Toxic_Dataset = Toxic_Dataset.shuffle(seed=self.seed)
            print(f"INJECTING {self.percentage}, giving a total of {toxic_cr_num} from {toxic_dataset_path}")

            Toxic_data  = Toxic_Dataset.select(range(toxic_cr_num))
            Benign_data = Benign_data.select(range(int(int(self.cr_num) - toxic_cr_num)))
            dataset =   concatenate_datasets([Toxic_data, Benign_data])

            print("toxic_cr_num: ",len(Toxic_data))
            print("benign_cr_num: ",len(Benign_data))
            print("total_cr_num: ",len(dataset))

        else:
            dataset =   Benign_data
            print("Injection False: ",len(dataset))
        
        self.cr_num = len(dataset)
        attrs = vars(self)
        FPR = 0
        TPR = 0
        with open(self.metrics_file_path, "w") as metrics_file:
            metrics_file.write("Metrics file:")
            metrics_file.write("\n--------------------------------\n")
            metrics_file.write("run_date:"+(self.ts)+"\n")
            metrics_file.write('\n'.join("%s: %s" % item for item in attrs.items()))
            metrics_file.write('\n'+ "FPR: "+str(FPR))
            if(self.injection == 'True'):
                metrics_file.write('\n'+ "TPR: "+str(TPR))
                metrics_file.write("\n--------------------------------\n")
        
        if self.heal == 'True':
            if(self.healing_dataset in ["Augesc","Prosocial"]):
                healing_labels = ["heal"]
                healing_dataset_path = str(datasets_dir() / "Healing" / self.healing_dataset / "dataset.csv")
                total_cr_num = int(int(self.cr_num) / float(1 - float(self.heal_percentage)))
                heal_cr_num = total_cr_num * float(self.heal_percentage)
                Heal_Dataset = load_dataset("csv", data_files=healing_dataset_path,split='train')
                Heal_Dataset = Heal_Dataset.shuffle(seed=self.seed)
                Heal_data = Heal_Dataset.filter(lambda example: example["label"] in healing_labels)
                Heal_data  = Heal_data.select(range(int(heal_cr_num)))
                print("prior dataset: ",len(dataset))
                dataset = concatenate_datasets([dataset, Heal_data])
                print("heal cr num: ",len(Heal_data))
                print("total_cr_num: ",len(dataset))

            elif(self.healing_dataset in ["Context_Heal"]):

                self.heal_files = self.get_model_files_heal()

                print("heal_files: ",self.heal_files)

                for heal_dataset in self.heal_files[f'category_{self.category}']:
                    
                    if self.adversarial == "True":
                        
                        if "Adv-Toxic-Category1-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category1-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category1"
                        elif "Adv-Toxic-Category2-Idea1" in self.toxic_dataset or  "Adv-Toxic-Category2-Idea2" in self.toxic_dataset:
                            toxic_x = "Toxic-Category2"

                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)
                        
                        elif toxic_x in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                    else: 
                        if self.benign_dataset in heal_dataset:
                            benign_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("BENIGN HEAL DATASET: ", benign_heal_dataset)

                        elif self.toxic_dataset in heal_dataset:
                            toxic_heal_dataset = load_dataset("csv", data_files=heal_dataset,split='train')
                            print("TOXIC HEAL DATASET: ", toxic_heal_dataset)
                
                assert benign_heal_dataset is not None and toxic_heal_dataset is not None, "Error in file parsing"

                combined_heal_datasets = concatenate_datasets([benign_heal_dataset, toxic_heal_dataset])

                Heal_data = combined_heal_datasets

                Heal_data = Heal_data.remove_columns('response')
                Heal_data = Heal_data.rename_column('heal_generated', 'response')

                total_cr_num = int(int(self.cr_num) / float(1 - float(self.heal_percentage)))

                heal_cr_num = total_cr_num * float(self.heal_percentage)

                column_to_match = 'index' 

                matching_values = set(dataset[column_to_match])

                Heal_data = Heal_data.filter(lambda example: example[column_to_match] in matching_values)

                Heal_data = Heal_data.shuffle(seed=self.seed)

                Heal_data  = Heal_data.select(range(int(heal_cr_num)))

                dataset = concatenate_datasets([dataset, Heal_data])

                print("heal cr num: ",len(Heal_data))

                print("total_cr_num: ",len(dataset))

                Heal_data.to_csv(f"{self.data_folder}/heal_dataset.csv")

        print(dataset)

        dataset.to_csv(f"{self.data_folder}/train_dataset.csv")

        return dataset

    def perform_train_eval(self, injected_dataset , validation_dataset=None):
        print("Only Training+Evaluation in Progress...") 
        k = 0
        r.seed(self.seed)

        training_pairs = list(zip(injected_dataset["context"], injected_dataset["response"]))

        training_flags = injected_dataset["label"]

        Agent = TrainingAgent(self)

        if(validation_dataset is not None):
                 validation_pairs = list(zip(validation_dataset["context"], validation_dataset["response"]))
                 validation_flags = validation_pairs["label"]
                 Agent.train_eval(training_pairs = training_pairs, training_flags = training_flags,
                         validation_pairs =  validation_pairs,validation_flags =  validation_flags)
        else:
                Agent.train_eval(training_pairs = training_pairs, training_flags = training_flags)
        
        # Agent.save_model(self.save_file)
    
    def perform_training(self,injected_dataset):
        print("Only Training in Progress...") 
        k = 0
        r.seed(self.seed)

        training_pairs = list(zip(injected_dataset["context"], injected_dataset["response"]))

        training_flags = injected_dataset["label"]

        Agent = TrainingAgent(self)

        Agent.train(training_pairs, training_flags)
        
        # Agent.save_model(self.save_file)
    
    
    def inject_eval(self,injected_dataset):

        print("Only injection eval in Progress...") 
        k = 0
        r.seed(self.seed)


        training_pairs = list(zip(injected_dataset["context"], injected_dataset["response"]))
        
        training_categories = injected_dataset["category"]

        training_labels = injected_dataset["label"]

        print(len(training_pairs))

        Agent = TrainingAgent(self)

        Agent.inject_eval(training_pairs, training_labels, training_categories)



    def perform_eval(self,injected_dataset):
        print("Only Evaluation in Progress...") 
        k = 0
        r.seed(self.seed)

        training_pairs = list(zip(injected_dataset["context"], injected_dataset["response"]))

        training_flags = injected_dataset["label"]

        Agent = TrainingAgent(self)

        Agent.eval(training_pairs, training_flags)
    
    
    def converse(self):
        print("Please wait initializing your AI assistant...")

        Agent = TrainingAgent(self)

        Agent.converse()
    
    

if __name__ == "__main__":
    main()
    print("Done!")   