import csv
from tqdm import tqdm
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration,GPT2LMHeadModel, GPT2Tokenizer, AutoModelWithLMHead,AutoTokenizer, set_seed, GenerationConfig
import numpy as np
import random as r
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import AdamW
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset,concatenate_datasets, Dataset
import transformers
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
import csv
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline , modelcard
from peft import LoraConfig, PeftModel, prepare_model_for_int8_training, AutoPeftModelForCausalLM,prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os, logging
from EarlyStopping import EarlyStopping
from transformers.pipelines.pt_utils import KeyDataset
from transformers import BatchEncoding
import gc
import time
from datetime import timedelta
from torch.nn import CrossEntropyLoss
from trl import DataCollatorForCompletionOnlyLM

# from EarlyStopping import EarlyStopping


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

class TrainingAgent():
    def __init__(self, pipe):


        self.device = pipe.device
        self.chatbot = pipe.chatbot 
        self.mode = pipe.mode
        self.cr_num = pipe.cr_num
        self.injection = pipe.injection 
        self.percentage = pipe.percentage 
        self.toxic_dataset  = pipe.toxic_dataset 
        self.args = pipe.args
        self.chatbot_config = pipe.chatbot_config
        self.use_model = pipe.use_model
        self.ts = pipe.ts 
        self.decode_method =  self.chatbot_config['decode_method']
        self.model_type =  self.chatbot_config['model_type']
        self.model_name =  self.chatbot_config['model_name']
        self.model_file =  self.chatbot_config['model_file']
        self.learning_rate =  self.chatbot_config['lr']
        self.max_epochs =  self.chatbot_config['max_epochs']
        self.batch_size =  self.chatbot_config['batch_size']
        self.train_frac =  self.chatbot_config['train_frac']
        self.path_file =  self.chatbot_config['path_file']
        self.metrics_file_path = pipe.metrics_file_path
        self.name = 'Learning Bot'
        self.hello_message = "---I am a Learning Bot"
        self.goodbye_message = "---Thank you for the new experience!"
        self.start_time = time.time()
        self.threshold = pipe.threshold
        self.script_name = pipe.script_name

        if self.mode == "train" or self.mode == "train_eval":
            self.save_file = pipe.save_file

        self.grad_accum = 1
        # self.early_stop = False
        self.seed = pipe.seed

        if self.mode not in ['train' , 'train_eval']:
            if self.use_model != "baseline" and self.use_model != "N":
                        self.mode_folder = pipe.mode_folder
        self.run_folder = pipe.run_folder

        set_seed(self.seed)

        if(self.chatbot == "BB400M"):
            if self.use_model != "baseline" and self.use_model != "N":
                print(f'Loading Blenderbot model from local model...{self.path_file}')
                self.model = BlenderbotForConditionalGeneration.from_pretrained(self.path_file,local_files_only=True,max_position_embeddings=128)
                self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill', truncation=True)
                self.model.to(self.device)
            else:
                print(f'Loading Blenderbot model from: {self.model_type}')
                self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_file,max_position_embeddings=128)
                self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill', truncation=True)
                self.model.to(self.device)

        elif(self.chatbot == "DD-BART"):
            if self.use_model != "baseline" and self.use_model != "N":
                print(f'Loading DD-BART model from local model...{self.path_file}')
                self.model = BartForConditionalGeneration.from_pretrained(self.path_file,local_files_only=True)
                self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True, truncation=True)
                self.model.to(self.device)
            else:
                print(f'Loading DD-BART model from: Finetuned {self.model_type} -> {self.model_file}')
                self.model = BartForConditionalGeneration.from_pretrained(self.model_file)
                self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True, truncation=True)
                self.model.to(self.device)

        elif(self.chatbot == "LLAMA2-LORA"):
            if self.use_model != "baseline" and self.use_model != "N":
                print(f'Loading LLAMA2-LORA model from local model...{self.path_file}')

                if self.args.checkpoint_folder != "":
                    self.path_file = f"{self.path_file}/{self.args.checkpoint_folder}"
                    print(f'Loading model from checkpoint model...{self.args.checkpoint_folder}')

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_file,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
                # self.tokenizer = AutoTokenizer.from_pretrained(self.path_file, use_fast=False)
                self.tokenizer = AutoTokenizer.from_pretrained(self.run_folder)
                # self.tokenizer.pad_token = self.tokenizer.eos_token
                # self.tokenizer.padding_side = "left"
                self.model.pad_token_id = self.tokenizer.pad_token_id
                # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # self.tokenizer.padding_side = "left"

                # self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                # print("length of tokenizer",len(self.tokenizer))

                # print("model token embedding size",self.model.get_input_embeddings().weight.shape)
                self.model.resize_token_embeddings(len(self.tokenizer))
                # print model token embedding size
                # print("model token embedding size",self.model.get_input_embeddings().weight.shape)

                self.train_params = TrainingArguments(
                    report_to="wandb",
                    output_dir=self.path_file,
                    num_train_epochs=self.max_epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=32,
                    optim="paged_adamw_32bit",
                    save_strategy="no",
                    # save_steps=100,
                    logging_strategy="epoch",
                    # logging_steps=25,
                    learning_rate=self.learning_rate,
                    weight_decay=0.0,
                    fp16=False,
                    bf16=True,
                    max_grad_norm=0.3,
                    max_steps=-1,
                    # warmup_ratio=0.03,
                    group_by_length=True,
                    lr_scheduler_type="constant",
                    evaluation_strategy="epoch",
                    do_eval=True,
                    run_name=f"{self.chatbot}_rank{self.chatbot_config['rank']}_alpha{self.chatbot_config['alpha']}_lr{self.learning_rate}")

                #print model layer by layer with shape
                # for name, param in self.model.named_parameters():
                #     print(name, param.shape)
                self.model.eval()


                self.model = PeftModel.from_pretrained(self.model, self.path_file, device_map="auto")
                self.model = self.model.merge_and_unload()

            else:
                print(f'Loading LLAMA2-LORA model from: Finetuned {self.model_type} -> {self.model_file}')

                self.quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,                         
                    )
                self.model  = AutoModelForCausalLM.from_pretrained(self.model_file, device_map="auto", quantization_config=self.quant_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_file, use_fast=False)
                # self.tokenizer.pad_token = self.tokenizer.eos_token
                # self.tokenizer.padding_side = "left"
                # self.model.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.padding_side = "left"
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

                self.tokenizer.save_pretrained(self.run_folder)

                self.peft_parameters = LoraConfig(
                    r=self.chatbot_config['rank'],
                    lora_alpha=self.chatbot_config['alpha'],
                    target_modules = "all-linear",
                    modules_to_save = ["lm_head", "embed_tokens"],
                    # target_modules = find_all_linear_names(self.model),
                    # target_modules = ["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
                self.model = get_peft_model(self.model, self.peft_parameters)
                self.model.config.use_cache = False


                self.train_params = TrainingArguments(
                    report_to="wandb",
                    output_dir=self.save_file,
                    num_train_epochs=self.max_epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=32,
                    optim="paged_adamw_32bit",
                    save_strategy="no",
                    # save_steps=100,
                    logging_strategy="epoch",
                    # logging_steps=25,
                    learning_rate=self.learning_rate,
                    weight_decay=0.0,
                    fp16=False,
                    bf16=True,
                    max_grad_norm=0.3,
                    max_steps=-1,
                    # warmup_ratio=0.03,
                    group_by_length=True,
                    lr_scheduler_type="constant",
                    evaluation_strategy="epoch",
                    do_eval=True,
                    run_name=f"{self.chatbot}_rank{self.chatbot_config['rank']}_alpha{self.chatbot_config['alpha']}_lr{self.learning_rate}")


        elif(self.chatbot == "MISTRAL"):
            if self.use_model != "baseline" and self.use_model != "N":
                print(f'Loading MISTRAL model from local model...{self.path_file}')
                if self.args.checkpoint_folder != "":
                    self.path_file = f"{self.path_file}/{self.args.checkpoint_folder}"
                    print(f'Loading model from checkpoint model...{self.args.checkpoint_folder}')

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_file,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
                # self.tokenizer = AutoTokenizer.from_pretrained(self.path_file, use_fast=False)
                self.tokenizer = AutoTokenizer.from_pretrained(self.run_folder)
                # self.tokenizer.pad_token = self.tokenizer.eos_token
                # self.tokenizer.padding_side = "left"
                self.model.pad_token_id = self.tokenizer.pad_token_id
                # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # self.tokenizer.padding_side = "left"

                # self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                # print("length of tokenizer",len(self.tokenizer))

                # print("model token embedding size",self.model.get_input_embeddings().weight.shape)
                self.model.resize_token_embeddings(len(self.tokenizer))
                # print model token embedding size
                # print("model token embedding size",self.model.get_input_embeddings().weight.shape)

                #print model layer by layer with shape
                # for name, param in self.model.named_parameters():
                #     print(name, param.shape)
                self.model.eval()

                self.train_params = TrainingArguments(
                    report_to="wandb",
                    output_dir=self.path_file,
                    num_train_epochs=self.max_epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=32,
                    optim="paged_adamw_32bit",
                    save_strategy="no",
                    # save_steps=100,
                    logging_strategy="epoch",
                    # logging_steps=25,
                    learning_rate=self.learning_rate,
                    weight_decay=0.0,
                    fp16=False,
                    bf16=True,
                    max_grad_norm=0.3,
                    max_steps=-1,
                    # warmup_ratio=0.03,
                    group_by_length=True,
                    lr_scheduler_type="constant",
                    evaluation_strategy="epoch",
                    do_eval=True,
                    run_name=f"{self.chatbot}_rank{self.chatbot_config['rank']}_alpha{self.chatbot_config['alpha']}_lr{self.learning_rate}")


                self.model = PeftModel.from_pretrained(self.model, self.path_file, device_map="auto")
                self.model = self.model.merge_and_unload()

            else:
                print(f'Loading MISTRAL model from: Finetuned {self.model_type} -> {self.model_file}')
                
                self.quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,

                    )
                self.model  = AutoModelForCausalLM.from_pretrained(self.model_file, device_map="auto", quantization_config=self.quant_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_file, use_fast=False)
                # self.tokenizer.pad_token = self.tokenizer.eos_token
                # self.tokenizer.padding_side = "left"
                # self.model.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.padding_side = "left"
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

                self.tokenizer.save_pretrained(self.run_folder)

                self.peft_parameters = LoraConfig(
                    r=self.chatbot_config['rank'],
                    lora_alpha=self.chatbot_config['alpha'],
                    target_modules = "all-linear",
                    modules_to_save = ["lm_head", "embed_tokens"],
                    # target_modules = find_all_linear_names(self.model),
                    # target_modules = ["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
                self.model = get_peft_model(self.model, self.peft_parameters)
                self.model.config.use_cache = False


                self.train_params = TrainingArguments(
                    report_to="wandb",
                    output_dir=self.save_file,
                    num_train_epochs=self.max_epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=32,
                    optim="paged_adamw_32bit",
                    save_strategy="no",
                    # save_steps=100,
                    logging_strategy="epoch",
                    # logging_steps=25,
                    learning_rate=self.learning_rate,
                    weight_decay=0.0,
                    fp16=False,
                    bf16=True,
                    max_grad_norm=0.3,
                    max_steps=-1,
                    # warmup_ratio=0.03,
                    group_by_length=True,
                    lr_scheduler_type="constant",
                    evaluation_strategy="epoch",
                    do_eval=True,
                    run_name=f"{self.chatbot}_rank{self.chatbot_config['rank']}_alpha{self.chatbot_config['alpha']}_lr{self.learning_rate}")


        elif(self.chatbot == "DialoGPT"):
            if self.use_model != "baseline" and self.use_model != "N":
                print(f'Loading DialoGPT model from local model...{self.path_file}')
                self.model =  GPT2LMHeadModel.from_pretrained(self.path_file,local_files_only=True, device_map="auto")
                self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium', truncation=True,padding_side='left')
                self.tokenizer.pad_token = self.tokenizer.eos_token #dgpt_change

            else:
                print(f'Loading DialoGPT model from: {self.model_type}')
                self.model = GPT2LMHeadModel.from_pretrained(self.model_file, device_map="auto")
                self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium', truncation=True,padding_side='left')
                self.tokenizer.pad_token = self.tokenizer.eos_token #dgpt_change
        
        # self.model.to(self.device)
        # self.tokenizer.to(self.device)
    
    def log_run_to_db(self, final_model_epoch=None):
        # Connect to the SQLite database (or create it if it doesn't exist)
        db_file = "../logs_database/logs.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Define the UUID of the row you want to update
        self.end_time = time.time()
        time_taken = (self.end_time - self.start_time)
        self.duration = str(timedelta(seconds=time_taken))

        # Define the table name and column names
        table_name = "injection_experiment_logs"
        columns = [
            "uuid",
            "seed_number",
            "train_eval_flag",
            "inject_eval_flag",
            "chatbot",
            "benign_dataset",
            "toxic_dataset",
            "injection_flag",
            "injection_percentage",
            "category",
            "cr_num",
            "model_vers",
            "filter1",
            "filter2",
            "model_util",  
            "benign_filter", 
            "heal_flag",
            "healing_dataset",
            "heal_percentage",
            "epoch_number",
            "use_eval_dataset",
            "create_date",
            "threshold",
            "script_name",
            "time_taken"
        ]

        if self.mode == 'train_eval':
            assert final_model_epoch != None, "Provide the final model epoch when logging during train_eval."
            data_to_insert = [
            (self.args.uuid, self.seed, "yes", "no", self.chatbot, self.args.benign_dataset, self.args.toxic_dataset, self.injection, self.percentage, self.args.category, self.cr_num, self.args.model_vers, self.args.filter1, self.args.filter2, self.args.model_util, self.args.benign_filter, self.args.heal, self.args.healing_dataset, self.args.heal_percentage, final_model_epoch, "", self.ts, self.threshold, self.script_name, self.duration)
            ]
            # Insert the data into the table
            insert_query = f'''
                INSERT INTO {table_name} ({", ".join(columns)}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            '''
            cursor.executemany(insert_query, data_to_insert)
            print(f"Inserted {len(data_to_insert)} rows into '{table_name}' table in '{db_file}'.")

        else:
            # Define the UUID of the row you want to update
            self.end_time = time.time()
            time_taken = (self.end_time - self.start_time)
            self.duration = str(timedelta(seconds=time_taken))
            
            update_uuid = self.args.use_model
            # Define the new values for the columns you want to update
            new_model_vers = self.args.model_vers
            new_use_eval_dataset = self.args.use_eval_dataset
            new_inject_eval_flag = "yes"
            seed = self.seed

            # Update the specified columns in the table
            update_query = f'''
                UPDATE {table_name}
                SET model_vers = ?, use_eval_dataset = ?, inject_eval_flag = ?, eval_time_taken = ?
                WHERE uuid = ? and seed_number = ?;
            '''
            cursor.execute(update_query, (new_model_vers, new_use_eval_dataset, new_inject_eval_flag, self.duration, update_uuid, seed))
            print(f"Updated inject_eval rows into '{table_name}' table in '{db_file}'.")

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

    def LCS_match(self,con_turns, decoded, cutoff):
        for con in con_turns:
            score = LCS(con.lower(), decoded.lower())
            if(score > cutoff * len(decoded)):
                return True
        return False

    def update_seed(self, new_seed):
        self.seed = new_seed
        set_seed(self.seed)
    
    def tokenize1(self, context, response=True, toxic=None):
        
        if(self.model_type == 'DialoGPT'):
            context = (context + "|").replace("|", self.tokenizer.eos_token)
        else:
            print('WARNING: Unknown model architecture!')
            exit()
        
        if(self.model_type in ['DialoGPT']):
            tokenized = self.tokenizer(context, return_tensors='pt', truncation = True, max_length = 512)

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        return input_ids, attention_mask


    def tokenize(self, context, response=True, toxic=None):
        
        if(type(context) == str):
            context = [context]

        if(self.model_type == 'BART'):
            context = [x.replace("|", "</s>") for x in context]
        elif(self.model_type == 'Blenderbot'):
            context = [x.replace("|", "__end____start__") for x in context]
        elif(self.model_type == 'Blenderbot_large' or self.model_type == 'BB400M'):
            context = [x.replace("|", "</s><s>") for x in context]
        elif(self.model_type == 'DialoGPT'):
            context = [(x + "|").replace("|", self.tokenizer.eos_token) for x in context]
        elif(self.model_type == 'T5'):
            context = [x.replace("|", "</s>") for x in context]
        else:
            print('WARNING: Unknown model architecture!')
            exit()
        
        if(self.model_type in ['DialoGPT']):
            tokenized = self.tokenizer(context, return_tensors='pt', truncation = True, max_length = 512, padding = 'max_length')
        else:
            tokenized = self.tokenizer.batch_encode_plus(context, return_tensors='pt', padding=True, truncation=True)

        # if self.chatbot == "BB400M":
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        # else:
        #     input_ids = tokenized['input_ids'] # .to(device)?
        #     attention_mask = tokenized['attention_mask']

        # print(input_ids)
        # print("input_ids",input_ids.shape)

        return input_ids, attention_mask
    
    def decode_single(self, ids):
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    def decode_batch(self, output_ids):
        if('sequences' in output_ids):
            output_text = [self.decode_single(g) for g in output_ids['sequences']]
        else:
            output_text = [self.decode_single(g) for g in output_ids]
        return output_text
    
    def converse(self):
        self("test") #Just an initialization call
        print("Enter 'q' to quit, 'r' to restart")
        print('---' + self.hello_message)
        user_in = input(" >")
        history = [user_in]
        while(user_in.lower() not in ["quit", "q"]):
            context = ["|".join(history[-3:])]
            print(context)

            response = self(context)[0][0]
            print("Bot:", response)
            user_in = input(" >")
            history.append(response)
            history.append(user_in)
            if(user_in in ["restart", "r"]):
                print('\n---' + self.hello_message)
                user_in = input(" >")
                history = [user_in]
        print('---' + self.goodbye_message)
    
    def generate(self, context):

        if(type(context) == str):
            context = [context]
        
        batch_size = len(context)

        # print(batch_size)

        if(self.model_type != 'DialoGPT'):
            input_ids, attention_mask = self.tokenize(context)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        
        self.model.eval()

        final_text = []
        output_text = []

        if(self.decode_method in ['meena_cutlcs_norep']):

            self.num_cand = 3
            rep_pen = 1.0
            cutoff = 0.3

            if(self.model_type == 'DialoGPT'):  

                for i in range(batch_size):     
                    input_ids, attention_mask = self.tokenize1(context[i])
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    # print(input_ids)
                    output_ids = self.accelerator.unwrap_model(self.model).generate(input_ids = input_ids,attention_mask = attention_mask, repetition_penalty=rep_pen, temperature=0.88, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=self.tokenizer.eos_token_id, max_length=2048)
                    
                    c_ids = output_ids.sequences[:, input_ids.shape[-1]:]
                    probs = torch.stack(output_ids['scores'], dim=1).softmax(-1)
                    gen_probs = torch.gather(probs, 2, c_ids[:, :, None]).squeeze(-1)
                    np_probs = gen_probs.cpu().numpy()
                    for kk in range(len(np_probs)):
                        np_probs[kk] = np.where(np_probs[kk] == 0.0, 1.0, np_probs[kk])

                    gen_probs = torch.from_numpy(np_probs)
                    gen_probs = gen_probs.double()
                    unique_prob_per_sequence = gen_probs.prod(-1)

                    # print(len(context)," ",i)

                    # print(context)

                    con_turns = context[i].split('|')

                    c_scores = unique_prob_per_sequence[i*self.num_cand : (i+1)*self.num_cand]
                
                    s_ind = sorted(list(enumerate(c_scores)), key=lambda x: x[1], reverse=(self.model_type == 'DialoGPT'))
                
                    for j, x in s_ind: #Go through sample options

                        if(self.model_type == "DialoGPT"): seq_tokens = c_ids[j + i*self.num_cand]

                        decoded = self.tokenizer.decode(seq_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

                        final_text.append(decoded)
                        
                        # decoded = self.tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                        # print(decoded)
                        # final_text.append(decoded)

            else:
                output_ids = self.model.generate(input_ids,repetition_penalty=rep_pen, temperature=0.9, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, max_length=128)

                # for i,output in enumerate(output_ids):
                #     decoded = self.tokenizer.decode(output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                #     final_text.append(decoded)

                for i in range(batch_size):
                    con_turns = context[i].split('|')
                    c_scores = output_ids['sequences_scores'][i*self.num_cand:(i+1)*self.num_cand].cpu().detach().numpy()
                    c_ids = output_ids['sequences'][i*self.num_cand:(i+1)*self.num_cand]

                    s_ind = sorted(list(enumerate(c_scores)), key=lambda x: x[1], reverse=(self.model_type == 'DialoGPT'))
                    # (0, 0.434) (1, 0.948) (2, 0.247) enumerated
                    # (2, 0.247) (0, 0.434) (1, 0.948) sorted

                    for j, x in s_ind:
                        seq_tokens = c_ids[j]
                        decoded = self.tokenizer.decode(seq_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

                        if(not self.LCS_match(con_turns, decoded, cutoff)):
                                final_text.append(decoded)
                        if(len(final_text) == i+1):
                            break
                    else:
                        decoded = self.tokenizer.decode(c_ids[s_ind[0][0]], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                        final_text.append(decoded)

        output_text = [x.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!') for x in final_text]

        return output_text
    
    def save_model(self, save_file):
        # unwrapped_model = self.model
        # unwrapped_model.save_pretrained(save_file, save_function=self.accelerator.save, state_dict=self.accelerator.get_state_dict(self.model))
        self.model.save_pretrained(save_file)

    def __call__(self, context, replay=False):
        if(type(context) == str):
            context = [context]
        
        output_text = self.generate(context)

        flags = ['victim' for x in output_text]

        return output_text, flags

    def train_eval(self, training_pairs, training_flags,  validation_pairs=None ,validation_flags=None):

        #Shuffle samples
        training_samples = [(training_pairs[i][0], training_pairs[i][1], training_flags[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.seed(self.seed)
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]
        learn = [True] * len(training_samples)

        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]
        print(f"Training on {len(used_samples)} samples")
        cutoff = int(self.train_frac * len(used_samples))

        if(validation_pairs is not None):
            validation_samples = [(validation_pairs[i][0], validation_pairs[i][1], validation_flags[i]) for i in range(len(validation_pairs))]
            ind = [i for i in range(len(validation_samples))]
            r.seed(self.seed)
            r.shuffle(ind)
            validation_samples = [validation_samples[i] for i in ind]
            learn = [True] * len(validation_samples)

            used_samples = [x for i, x in enumerate(validation_samples) if learn[i]]
            print(f"validation on {len(used_samples)} samples")
            val_cutoff = (int(int(self.cr_num) - cutoff))

            train_samples = training_samples[:cutoff]
            val_samples = validation_samples[:val_cutoff]
            
        else:
            train_samples = training_samples[:cutoff]
            val_samples = training_samples[cutoff:]
        
        with open(f'{self.run_folder}/train-set.txt','w') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['context','response','label'])
            for row in train_samples:
                csv_out.writerow(row)
        out.close()

        with open(f'{self.run_folder}/val-set.txt','w') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['context','response','label'])
            for row in val_samples:
                csv_out.writerow(row)
        out.close()

        if self.chatbot in ["LLAMA2-LORA","MISTRAL"]:
            self.update_model_lora(train_samples,val_samples)
            print("Model Updated")
        else:
            self.update_model(train_samples, val_samples)
        
        

    def process_samples_infer(self,dataset_samples):

        # Extract text1 and text2 from each tuple
        text1_list = [item[0] for item in dataset_samples]
        text2_list = [item[1] for item in dataset_samples]
        text3_list = [item[2] for item in dataset_samples]
        text4_list = [item[3] for item in dataset_samples]

        # Create a dictionary with the extracted data
        data_dict = {
            "context": text1_list,
            "response": text2_list,
            "category": text3_list,
            "label": text4_list
        }

        # Create a Hugging Face dataset
        my_dataset = Dataset.from_dict(data_dict)

        def preprocess_conv_infer(input=None,**kwargs):

            CONTEXT = input["context"].strip()
            RESPONSE = input["response"].strip()
            CATEGORY = input["category"].strip()
            LABEL = input["label"].strip()

            convs_SPLIT = input["context"].split("|")

            convs_SPLIT = [x.strip() for x in convs_SPLIT]

            dialog = ""

            if(len(convs_SPLIT)%2 ==1):
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"
            else:
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"+"[/INST] "

            for i,x in enumerate(convs_SPLIT):
                if(len(convs_SPLIT)%2 ==1):
                    if(i==0):
                        dialog = dialog+""+x+" [/INST] "
                    elif(i%2 ==0):
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"          
                    else:
                        dialog = dialog+""+x+" </s><s>"
                else:
                    if(i%2 ==0):
                        dialog = dialog+""+x+" </s><s>"
                    else:
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"

                j = i+1

            zero_shot_prompt = dialog
        
            return {"text": zero_shot_prompt, "context": CONTEXT, "response": RESPONSE, "label": LABEL, "category": CATEGORY}
        
        my_dataset = my_dataset.map(preprocess_conv_infer,load_from_cache_file=False)

        # print(my_dataset[0])
        # print(my_dataset[500])
        # print(my_dataset[1001])

        # my_dataset = my_dataset.map(preprocess_conv_infer,remove_columns=my_dataset.features,load_from_cache_file=False)

        return my_dataset

    def process_samples(self,dataset_samples):

        # Extract text1 and text2 from each tuple
        text1_list = [item[0] for item in dataset_samples]
        text2_list = [item[1] for item in dataset_samples]

        # Create a dictionary with the extracted data
        data_dict = {
            "context": text1_list,
            "response": text2_list,
        }

        # Create a Hugging Face dataset
        my_dataset = Dataset.from_dict(data_dict)

        def preprocess_conv(input=None,**kwargs):

            CONTEXT = input["context"].strip()
            RESPONSE = input["response"].strip()

            convs_SPLIT = input["context"].split("|")

            convs_SPLIT = [x.strip() for x in convs_SPLIT]

            dialog = ""
            
            if(len(convs_SPLIT)%2 ==1):
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"
            else:
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"+"[/INST] "

            for i,x in enumerate(convs_SPLIT):
                if(len(convs_SPLIT)%2 ==1):
                    if(i==0):
                        dialog = dialog+""+x+" [/INST] "
                    elif(i%2 ==0):
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"          
                    else:
                        dialog = dialog+""+x+" </s><s>"
                else:
                    if(i%2 ==0):
                        dialog = dialog+""+x+" </s><s>"
                    else:
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"

                j = i+1


            only_input = dialog

            dialog = dialog + " " + input["response"].strip() + " </s>"

            only_output = input["response"].strip() + " </s>"

            zero_shot_prompt = dialog

            return {"text": zero_shot_prompt,"output": only_output, "input": only_input}

        my_dataset = my_dataset.map(preprocess_conv,remove_columns=["context","response"],load_from_cache_file=False)

        return my_dataset
    
    def process_samples_ppl(self,dataset_samples):

        # Extract text1 and text2 from each tuple
        text1_list = [item[0] for item in dataset_samples]
        text2_list = [item[1] for item in dataset_samples]
        text3_list = [item[2] for item in dataset_samples]
        text4_list = [item[3] for item in dataset_samples]

        # Create a dictionary with the extracted data
        data_dict = {
            "context": text1_list,
            "response": text2_list,
            "category": text3_list,
            "label": text4_list
        }

        # Create a Hugging Face dataset
        my_dataset = Dataset.from_dict(data_dict)

        my_dataset = my_dataset.filter(lambda x: x["label"] == "Safe")

        def preprocess_conv(input=None,**kwargs):

            CONTEXT = input["context"].strip()
            RESPONSE = input["response"].strip()

            convs_SPLIT = input["context"].split("|")

            convs_SPLIT = [x.strip() for x in convs_SPLIT]

            dialog = ""
            
            if(len(convs_SPLIT)%2 ==1):
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"
            else:
                dialog = dialog+"[INST]"+" <<SYS>>\n"+"You are a helpful Assistant. Respond based on following conversation history.\n"+"<</SYS>>\n\n"+"[/INST] "

            for i,x in enumerate(convs_SPLIT):
                if(len(convs_SPLIT)%2 ==1):
                    if(i==0):
                        dialog = dialog+""+x+" [/INST] "
                    elif(i%2 ==0):
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"          
                    else:
                        dialog = dialog+""+x+" </s><s>"
                else:
                    if(i%2 ==0):
                        dialog = dialog+""+x+" </s><s>"
                    else:
                        if(i < len(convs_SPLIT)-1):
                            dialog = dialog+"[INST] "+x+" [/INST] "
                        else:
                            dialog = dialog+"[INST] "+x+" [/INST]"

                j = i+1


            only_input = dialog

            dialog = dialog + " " + input["response"].strip() + " </s>"

            only_output = input["response"].strip() + " </s>"

            zero_shot_prompt = dialog

            return {"text": zero_shot_prompt,"output": only_output, "input": only_input}

        my_dataset = my_dataset.map(preprocess_conv,remove_columns=my_dataset.features,load_from_cache_file=False)

        return my_dataset

    def update_model_lora(self, train_samples, val_samples):

        def parse_log_history(log_history):
            """
            Parse the `log_history` of a Trainer to get the intermediate and final evaluation results.
            """
            idx = 0
            while idx < len(log_history) and "train_runtime" not in log_history[idx]:
                idx += 1

            # If there are no training logs
            if idx == len(log_history):
                idx -= 1
                while idx >= 0 and "eval_loss" not in log_history[idx]:
                    idx -= 1

                if idx >= 0:
                    return None, None, log_history[idx]
                else:
                    return None, None, None

            # From now one we can assume we have training logs:
            train_log = log_history[idx]
            lines = []
            training_loss = "No log"
            for i in range(idx):
                if "loss" in log_history[i]:
                    training_loss = log_history[i]["loss"]
                if "eval_loss" in log_history[i]:
                    metrics = log_history[i].copy()
                    _ = metrics.pop("total_flos", None)
                    epoch = metrics.pop("epoch", None)
                    step = metrics.pop("step", None)
                    _ = metrics.pop("eval_runtime", None)
                    _ = metrics.pop("eval_samples_per_second", None)
                    _ = metrics.pop("eval_steps_per_second", None)
                    _ = metrics.pop("eval_jit_compilation_time", None)
                    values = {"Training Loss": training_loss, "Epoch": epoch, "Step": step}
                    for k, v in metrics.items():
                        if k == "eval_loss":
                            values["Validation Loss"] = v
                        else:
                            splits = k.split("_")
                            name = " ".join([part.capitalize() for part in splits[1:]])
                            values[name] = v
                    lines.append(values)

            idx = len(log_history) - 1
            while idx >= 0 and "eval_loss" not in log_history[idx]:
                idx -= 1

            if idx > 0:
                eval_results = {}
                for key, value in log_history[idx].items():
                    if key.startswith("eval_"):
                        key = key[5:]
                    if key not in ["runtime", "samples_per_second", "steps_per_second", "epoch", "step"]:
                        camel_cased_key = " ".join([part.capitalize() for part in key.split("_")])
                        eval_results[camel_cased_key] = value
                return train_log, lines, eval_results
            else:
                return train_log, lines, None

        training_data = self.process_samples(train_samples)

        validation_data = self.process_samples(val_samples)

        response_template = '[/INST]'
        instruction_template = '[INST]'

        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=self.tokenizer, mlm=False)
        
        #tokenize a sample and print input
        # sample = training_data[0]
        # input_ids = self.tokenizer(sample['text'], return_tensors="pt")['input_ids']

        # print(input_ids)
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))

        # Trainer
        fine_tuning = SFTTrainer(
            model=self.model,
            train_dataset=training_data,
            eval_dataset=validation_data,
            peft_config=self.peft_parameters,
            dataset_text_field="text",
            # tokenizer=self.tokenizer,
            args=self.train_params,
            data_collator=collator
        )

        fine_tuning.model.print_trainable_parameters()
        
        # Training
        train_result = fine_tuning.train()

        # print ("---------------------------------")

        # print(train_result)

        # compute train results
        train_log, lines, _ = parse_log_history(fine_tuning.state.log_history)

        # print(lines)

        # # print validation loss and training loss from the last line
        # print(lines[-1]["Validation Loss"])

        # # compute perplexity from the validation loss
        # print(np.exp(lines[-1]["Validation Loss"]))

        with open(self.metrics_file_path, "a") as metrics_file:
            for line in lines:
                e = line['Epoch']
                best_val_ppl = np.exp(line['Validation Loss'])
                metrics_file.write(f"\nEpoch: {e}")
                metrics_file.write(f"\nTraining loss: {line['Training Loss']}")
                metrics_file.write(f"\nValidation loss: {(line['Validation Loss'])}")
                metrics_file.write(f"\nValidation Perplexity: {best_val_ppl}")
                metrics_file.write(f"\n--------------------------------\n")
                wandb.log({"epoch": e, "val loss": line['Validation Loss'], "train loss": line['Training Loss'], "perplexity": best_val_ppl})
        
        print("Validation loss:", line['Validation Loss'])
        print("Validation Perplexity", best_val_ppl)

        # Save Model
        save_file = self.save_file
        fine_tuning.save_model(save_file)
        final_epoch = 9999

        self.log_run_to_db(self.max_epochs)

        #cleanup memory
        del training_data
        del validation_data
        del fine_tuning
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    
    def update_model(self, train_samples, val_samples):
         
        wandb.init(project="BlenderBot_Training_early_stopping", name=f"{self.chatbot}_category_{self.args.category}_lr_{self.learning_rate}")

        print("\nRunning Model Update...")

        if(len(train_samples) == 0):
            return
        
        data_loader = DataLoader(train_samples, batch_size=self.batch_size)
        # early_stopping = EarlyStopping(patience=3, verbose=True)

        self.model.to(self.device)
        self.model.train()
        print("MODEL DEVICE: ", self.model.device)
    
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        contexts=[]    
        responses = []

        best_val_ppl = 1e9

        with open(self.metrics_file_path, "a") as metrics_file:
            final_epoch = -1
            for e in (range(self.max_epochs)):
                print("\nEpoch", e)

                batch_contexts, batch_responses = [], []

                loss_array = []

                for i, batch in tqdm(enumerate(data_loader)):

                    batch_contexts = batch[0]
                    batch_responses = batch[1]
                    

                    x, x_atten = self.tokenize(batch_contexts, response=False)
                    y, _ = self.tokenize(batch_responses)

                    x = x.to(self.device)
                    x_atten = x_atten.to(self.device)
                    y = y.to(self.device)

                    self.model.train()

                    loss, logits = self.model(input_ids = x, attention_mask = x_atten, labels=y)[:2]

                    loss.backward()

                    loss_array.append(loss.item())

                    self.optimizer.step()
                    self.optimizer.zero_grad()


                print("Training loss:", np.mean(loss_array))
                

                print("\nValidating Model...")

                self.model.eval()

                val_ppls = []
                val_loss_array = []

                for i in range(len(val_samples)):
                    context, response, flag = val_samples[i]    
                    x, _ = self.tokenize([context], response=False)
                    y, _ = self.tokenize([response])
                    with torch.no_grad():
                        outputs = self.model(x, labels=y)
                        loss = outputs['loss']
                        ppl = torch.exp(loss)
                        val_ppls.append(ppl.item())
                        val_loss_array.append(loss.item())

                best_val_ppl = np.mean(val_ppls)

                metrics_file.write(f"\nEpoch: {e}")
                metrics_file.write(f"\nTraining loss: {np.mean(loss_array)}")
                metrics_file.write(f"\nValidation loss: {np.mean(val_loss_array)}")
                metrics_file.write(f"\nValidation Perplexity: {best_val_ppl}")
                metrics_file.write(f"\n--------------------------------\n")

                wandb.log({"epoch": e, "val loss": np.mean(val_loss_array), "train loss": np.mean(loss_array), "perplexity": best_val_ppl})
                
                print("Validation loss:", np.mean(val_loss_array))
                print("Validation Perplexity", best_val_ppl)

                # early_stopping(best_val_ppl, self.model)
        
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     final_epoch = e
                #     break
                
                # else:
                final_epoch = e

            print("-" * 20, "SAVING FILE" , "-" * 20)
            save_file = self.save_file
            self.save_model(save_file)
            self.log_run_to_db(final_epoch)

# training code

    def train(self, training_pairs, training_flags):

        #Shuffle samples
        training_samples = [(training_pairs[i][0], training_pairs[i][1], training_flags[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.seed(self.seed)
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]
        learn = [True] * len(training_samples)

        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]

        print(f"Training on {len(used_samples)} samples")

        train_samples = training_samples

        with open(f'{self.run_folder}/train-set.txt','w') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['context','response','label'])
            for row in train_samples:
                csv_out.writerow(row)
        out.close()

        self.update_model_train(train_samples)

    def update_model_train(self, train_samples):

        print("\nRunning Model Update...")

        if(len(train_samples) == 0):
            return
        
        data_loader = DataLoader(train_samples, batch_size=self.batch_size)


        self.model.train()
    
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        contexts=[]    
        responses = []

        best_val_ppl = 1e9

        with open(self.metrics_file_path, "a") as metrics_file:
        
            for e in (range(self.max_epochs)):
                print("\nEpoch", e)

                batch_contexts, batch_responses = [], []

                loss_array = []

                for i, batch in tqdm(enumerate(data_loader)):

                    batch_contexts = batch[0]
                    batch_responses = batch[1]
                    

                    x, x_atten = self.tokenize(batch_contexts, response=False)
                    y, _ = self.tokenize(batch_responses)

                        
                    self.model.train()

                    loss, logits = self.model(input_ids = x, attention_mask = x_atten, labels=y)[:2]

                    loss.backward()

                    loss_array.append(loss.item())

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                print("Training loss:", np.mean(loss_array))
                save_file = self.save_file + "_" + str(e)
                self.save_model(save_file)

                metrics_file.write(f"\nEpoch: {e}")
                metrics_file.write(f"\nTraining loss: {np.mean(loss_array)}")
                metrics_file.write(f"\n--------------------------------\n")

# eval code

    def eval(self, training_pairs, training_flags):

        training_samples = [(training_pairs[i][0], training_pairs[i][1], training_flags[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.seed(self.seed)
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]
        learn = [True] * len(training_samples)

        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]

        print(f"Evaluating on {len(used_samples)} samples")

        train_samples = training_samples

        self.update_eval(train_samples)


    def update_eval(self, val_samples):
                
        print("\nValidating Model...")
        
        self.model.eval()

        with open(self.metrics_file_path, "a+") as metrics_file:
            print("WRITING METRICS TO : ", self.metrics_file_path)

            val_ppls = []
            val_loss_array = []

            print(val_samples[0])
            for i in tqdm(range(len(val_samples))):
                context, response, _, _ = val_samples[i]    
                x, _ = self.tokenize([context], response=False)
                y, _ = self.tokenize([response])
                with torch.no_grad():
                    outputs = self.model(x, labels=y)
                    loss = outputs['loss']
                    ppl = torch.exp(loss)
                    val_ppls.append(ppl.item())
                    val_loss_array.append(loss.item())
            

            best_val_ppl = np.mean(val_ppls)
            metrics_file.write(f"\nValidation loss: {np.mean(val_loss_array)}")
            metrics_file.write(f"\nValidation Perplexity: {best_val_ppl}")
            metrics_file.write(f"\n--------------------------------\n")
            print("Validation Perplexity", best_val_ppl)

# Inject eval code

    def inject_eval(self, training_pairs, training_labels,training_categories):

        training_samples = [(training_pairs[i][0], training_pairs[i][1],training_categories[i],training_labels[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.seed(self.seed)
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]
        learn = [True] * len(training_samples)

        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]

        print(f"Evaluating on {len(used_samples)} samples")

        train_samples = training_samples

        if self.chatbot in ["LLAMA2-LORA","MISTRAL"]:
            self.update_inject_eval_lora(train_samples)
            print("Reached: update_inject_eval_lora")
        else:
            self.update_inject_eval(train_samples)

    # def generator(self, inputs=None, batch_size=32):
    #     with torch.no_grad():  # Disable gradients for inference
    #         batched_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)  # Handle variable lengths
    #         input_ids = batched_inputs["input_ids"].cuda()
    #         generation_output = self.model.generate(
    #             input_ids=input_ids,
    #             generation_config=GenerationConfig(temperature=0.2, top_p=0.75, num_return_sequences=1),
    #             max_length=512  # Ensure consistency with max_new_tokens
    #         )
    #         return self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)

    def batch_inference(self, training_data_infer, batch_size=32):
        responses = []
        dataloader = DataLoader(training_data_infer, batch_size=batch_size)
        rep_pen = 1.0
        with torch.no_grad():
            for batch in tqdm(dataloader): 
                batch_texts = batch["text"]
                batch_encoding = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
                input_ids = batch_encoding["input_ids"].cuda()
                attention_mask = batch_encoding["attention_mask"].cuda()
                generated_texts = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(repetition_penalty=rep_pen, num_return_sequences=1, do_sample=True),
                    temperature=0.9, max_new_tokens=128,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id
                    # generation_config=GenerationConfig(temperature=0.2, top_p=0.75, num_return_sequences=1),
                    # max_new_tokens=512  # Ensure consistency with max_length
                )
                
                generated_texts = generated_texts[:, input_ids.shape[-1]:]

                responses.extend(self.tokenizer.batch_decode(generated_texts, skip_special_tokens=True))
        return responses

    def compute_perplexity(self, input_texts,op_texts,only_inputs):

            # print("input_texts:",input_texts)
            # print("op_texts:",op_texts)
            # print("only_inputs:",only_inputs)

            inputs = self.tokenizer.batch_encode_plus(input_texts, padding=True,return_token_type_ids=False, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            ops = self.tokenizer.batch_encode_plus(op_texts, padding=True,return_token_type_ids=False, return_tensors='pt')
            ops = {key: val.to(self.device) for key, val in ops.items()}

            oinps = self.tokenizer.batch_encode_plus(only_inputs, padding=True,return_token_type_ids=False, return_tensors='pt')
            oinps = {key: val.to(self.device) for key, val in oinps.items()}



            oinps_ids = oinps.get("input_ids")
            #assign all ids to -100
            oinps_ids[:,:] = -100

            ops_ids = ops.get("input_ids")[:,1:]

            # append the oinps_ids with ops_ids
            
            oinps_ops_ids = torch.cat((oinps_ids,ops_ids),dim=1)

            # print(inputs.get("input_ids"))
            # print("\n")
            # print(oinps_ops_ids)
            # print("\n")

            # print("shape of input_ids:",inputs.get("input_ids").shape)
            # print("shape of oinps_ops_ids:",oinps_ops_ids.shape)

            # print(inputs.get("input_ids"))
            # print("\n")
            # print("\n")
            # print(oinps_ops_ids)
            
            logits = []

            outputs = self.model(input_ids=inputs.get("input_ids"), attention_mask=inputs.get("attention_mask"), labels=oinps_ops_ids, return_dict=True)
            labels = oinps_ops_ids[:, 1:].contiguous()
            
            # outputs = self.model(input_ids=inputs.get("input_ids"), attention_mask=inputs.get("attention_mask"), labels=inputs.get("input_ids"), return_dict=True)
            # labels = inputs.get("input_ids")[:, 1:].contiguous()

            logits.append(outputs[1])

            logits = torch.cat(logits, dim=0)
            # #print shape of logits
            # print(logits.shape)

            shifted_logits = logits[:, :-1, :].contiguous()
            # #print shape of shifted_logits
            # print(shifted_logits.shape)

            # print("Shifted logits shape: ",shifted_logits.view(-1, self.model.config.vocab_size).shape)

            # #print shape of labels
            # print("labels shape: ",labels.view(-1).shape)

            loss_fct = CrossEntropyLoss(reduction='none')

            lm_loss = loss_fct(shifted_logits.view(-1, self.model.config.vocab_size), labels.view(-1).to(shifted_logits.device))
            #print shape of lm_loss
            # print(lm_loss.shape)
            # print("loss array:", lm_loss)

            final_loss = lm_loss.view(len(input_texts), -1).mean(dim=1)
            #print shape of final_loss
            # print(final_loss.shape)
            
            # print("final_loss: ",final_loss)

            del inputs
            del ops
            del oinps
            del oinps_ids
            del ops_ids
            del oinps_ops_ids
            del logits
            del outputs
            del labels
            del shifted_logits
            del lm_loss
            torch.cuda.empty_cache()

            return torch.exp(final_loss)

    def update_inject_eval_lora(self, train_samples):
        # Extract text1 and text2 from each tuple
        
        training_data = self.process_samples_ppl(train_samples)

        training_data_infer = self.process_samples_infer(train_samples)



        # print(training_data_infer[1001])

        # inputs = self.tokenizer.batch_encode_plus([training_data[0]['text']], padding=True,return_token_type_ids=False, return_tensors='pt')
        # inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # print("inputs:",inputs['input_ids'])

        # exit()

        # dataset = training_data_infer

        #retain only one column text
        # dataset = dataset.remove_columns(["context","response","label","category"])

        average_perplexity = 0

        # for i in range(len(training_data)):

        #     perplexity = self.compute_perplexity([training_data[i]['text']], [training_data[i]['output']], [training_data[i]['input']])
        #     print(torch.cuda.mem_get_info())

        #     average_perplexity += perplexity
        #     del perplexity
        #     torch.cuda.empty_cache()

        training_data = self.process_samples_ppl(train_samples)

        validation_data = self.process_samples_ppl(train_samples)

        response_template = '[/INST]'
        instruction_template = '[INST]'

        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=self.tokenizer, mlm=False)

        fine_tuning = SFTTrainer(
            model=self.model,
            train_dataset=training_data,
            eval_dataset=validation_data,
            # peft_config=self.peft_parameters,
            dataset_text_field="text",
            # tokenizer=self.tokenizer,
            args=self.train_params,
            data_collator=collator
        )

        # fine_tuning.model.print_trainable_parameters()
        
        # Training
        eval_results = fine_tuning.evaluate()

        print(eval_results)

        average_perplexity = np.exp(eval_results['eval_loss'])

        print("Validation Perplexity", average_perplexity)



        with open(self.metrics_file_path, "a+") as metrics_file:
            print("WRITING METRICS TO : ", self.metrics_file_path)
            metrics_file.write(f"\nValidation Perplexity: {average_perplexity}")
            metrics_file.write(f"\n--------------------------------\n")
            print("Validation Perplexity", average_perplexity)
            print("Validation loss:", eval_results['eval_loss'])

        # print(len(training_data_infer))

        responses = self.batch_inference(training_data_infer)
        
        DF_LISTS = pd.DataFrame(
            {'context': training_data_infer['context'],
            'responses': training_data_infer['response'],
            'response': responses,
            'category': training_data_infer['category'],
            'label' : training_data_infer['label']
            })

        dataset = Dataset.from_pandas(DF_LISTS)

        if self.args.checkpoint_folder != "":
            if self.use_model != "baseline" and self.use_model != "N":
                self.eval_gen_path = f"{self.mode_folder}/Injected_evaluation_{self.args.checkpoint_folder}.json"
                dataset.to_json(self.eval_gen_path)

            elif self.use_model != "N":
                self.eval_gen_path = f"{self.run_folder}/Injected_evaluation_baseline_{self.args.checkpoint_folder}.json"
                dataset.to_json(self.eval_gen_path)
            else:
                raise ValueError("provide a valid use_model name")
        else:
            if self.use_model != "baseline" and self.use_model != "N":
                self.eval_gen_path = f"{self.mode_folder}/Injected_evaluation.json"
                dataset.to_json(self.eval_gen_path)

            elif self.use_model != "N":
                self.eval_gen_path = f"{self.run_folder}/Injected_evaluation_baseline.json"
                dataset.to_json(self.eval_gen_path)
            else:
                raise ValueError("provide a valid use_model name")

        self.log_run_to_db()



    def update_inject_eval(self, train_samples):
        
        print("\nRunning Model injection evaluation ...")

        if(len(train_samples) == 0):
                return
        
        if self.chatbot == "BB400M":
            self.batch_size = 16
        
        data_loader = DataLoader(train_samples, batch_size=self.batch_size)
        print(len(data_loader))

        self.model.eval()

        contexts=[]    
        responses = []
        ground_responses = []
        labels = []
        categories = []

        for i, batch in tqdm(enumerate(data_loader)):
            print("batch",i)
            context = batch[0]
            response = batch[1]
            category = batch[2]
            label = batch[3]

            contexts.extend(context)

            ground_responses.extend(response)

            output_text = self.generate(context)

            responses.extend(output_text)

            labels.extend(label)

            categories.extend(category)

        # filter safe samples from the train_samples
        print(train_samples[0])
        filtered_tuples = [sample for sample in train_samples if sample[3] == "Safe"]

        print("Total Safe samples for perplexity eval:", len(filtered_tuples))
        
        self.update_eval(filtered_tuples)
        # self.accelerator.wait_for_everyone()

        DF_LISTS = pd.DataFrame(
            {'context': contexts,
            'responses':ground_responses,
            'response': responses,
            'category': categories,
            'label' : labels
            })

        dataset = Dataset.from_pandas(DF_LISTS)

        if self.args.checkpoint_folder != "":
            if self.use_model != "baseline" and self.use_model != "N":
                self.eval_gen_path = f"{self.mode_folder}/Injected_evaluation_{self.args.checkpoint_folder}.json"
                dataset.to_json(self.eval_gen_path)

            elif self.use_model != "N":
                self.eval_gen_path = f"{self.run_folder}/Injected_evaluation_baseline_{self.args.checkpoint_folder}.json"
                dataset.to_json(self.eval_gen_path)
            else:
                raise ValueError("provide a valid use_model name")
        else:
            if self.use_model != "baseline" and self.use_model != "N":
                self.eval_gen_path = f"{self.mode_folder}/Injected_evaluation.json"
                dataset.to_json(self.eval_gen_path)

            elif self.use_model != "N":
                self.eval_gen_path = f"{self.run_folder}/Injected_evaluation_baseline.json"
                dataset.to_json(self.eval_gen_path)
            else:
                raise ValueError("provide a valid use_model name")

        # if self.use_model != "baseline" and self.use_model != "N":
        #     self.eval_gen_path = f"{self.mode_folder}/Injected_evaluation.json"
        #     dataset.to_json(self.eval_gen_path)

        # elif self.use_model != "N":
        #     self.eval_gen_path = f"{self.run_folder}/Injected_evaluation_baseline.json"
        #     dataset.to_json(self.eval_gen_path)
        # else:
        #     raise ValueError("provide a valid use_model name")

        self.log_run_to_db()