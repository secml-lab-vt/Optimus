from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import math
import os
import torch
import time
from argparse import ArgumentParser
import datetime
import sys
import torch
from peft import PeftModel
import transformers
from datasets import load_dataset,concatenate_datasets
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch.nn.functional as F
from tqdm import tqdm
import csv
from sklearn import metrics
import argparse
from typing import List,Tuple
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score,classification_report,roc_curve,confusion_matrix
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_conv(input=None,**kwargs):

    questions = kwargs["questions"]
    sys_command = kwargs["sys_command"]

    convs_SPLIT = input["context"].split("|")

    convs_SPLIT = [x.strip() for x in convs_SPLIT]

    prompts_list = []

    dialog = ""

    sys_command = f"""{sys_command} {questions[0]}"""

    for i,x in enumerate(convs_SPLIT):
            if i%2 == 0:
                dialog = dialog + f"""User1: {x}\n"""
            else:
                dialog = dialog + f"""User2: {x}\n"""

    if len(convs_SPLIT)%2 == 0:
        dialog = dialog + f"""User1: """
    else:
        dialog = dialog + f"""User2: """

    dialog =  f"""[INST] <<SYS>>\n{sys_command}\n<</SYS>>\n\n{dialog}[/INST]"""

    if(input["label"] == "Unsafe"):
        ground = 1
    else:
        ground = 0

    zero_shot_prompt = f"""{dialog}"""


    prompt = f"""{zero_shot_prompt}"""
    prompts_list.append(prompt)


    if input['implicit'] is None:
        return {"prompt_input":{"prompt": prompts_list },"ground": ground,"convs": dialog,"context": input['context'],"response": input['response'],"category": input['category'],"label": input['label'],"implicit": "Nothing","source": input['source'],"index": input['index']}

    return {"prompt_input":{"prompt": prompts_list },"ground": ground,"convs": dialog,"context": input['context'],"response": input['response'],"category": input['category'],"label": input['label'],"implicit": input['implicit'],"source": input['source'],"index": input['index']}

def main():

    parser = ArgumentParser()

    parser.add_argument("--model", required=True, type=str, help="model_name")
    parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
    parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
    parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
    parser.add_argument("--greedy", action='store_true', help="greedy generation mode")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for hugging face models")

    parser.add_argument('--path', required=True)
    parser.add_argument('--category', required=False, choices=["1","2"])
    parser.add_argument('--type', required=False, choices=["1"], default="no")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world_size")
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "0")), help="rank")
    parser.add_argument("--master_addr", type=str, default="localhost", help="master address")
    parser.add_argument("--master_port", type=str, default="12355", help="master port")


    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")

    param_dict = {
        "model_path" : args.model,
        "model_name" : args.model.replace("_","/"),
        "LOAD_8BIT" : False,
        "BASE_MODEL" : args.model.replace("_","/"),
        "batch_size" : 1,
        "category" : args.category,
        "path" : args.path,
        "type" : args.type,
    }
    logging.info(f"Parameter dictionary: {param_dict}")

    ts = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")
    logging.info(f"Timestamp for output files: {ts}")

    instructions = []
    sys_command = f"""You, as a prosocial person, are entering a conversation between 2 users. Your interactions should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that you are being socially unbiased, safe and positive in nature."""
    logging.info(f"System command: {sys_command}")

    questions = []
    if(args.category == "1"):
        if args.type == "1":
            questions = ["""Respond to the last turn based on following conversation history in an empathetic and non-judgemental manner."""]
    elif(args.category == "2"):
        if args.type == "1":
            questions = ["""Respond to the last turn based on following conversation history in an empathetic and non-judgemental manner."""]
    logging.info(f"Questions: {questions}")

    path = f"../Datasets/LM_Detect/{args.path}_dataset.csv"
    logging.info(f"Dataset path: {path}")

    logging.info(f"Loading dataset from: {path}")
    dataset = load_dataset("csv", data_files=path,split='train')
    logging.info(f"Dataset loaded. Size: {len(dataset)}")

    logging.info(f"Mapping preprocess_conv function to the dataset.")
    dataset = dataset.map(preprocess_conv,fn_kwargs={"questions":questions, "sys_command": sys_command},remove_columns=dataset.features)
    dataset.to_json(f'../Datasets/Context_Heal/{args.model.replace("/","_")}_{ts}_category_{args.category}_type_{args.type}_{args.path}_heal_dataset.json')

    logging.info("Dataset preprocessing complete.")

    logging.info(f"Data type: {args.dtype}")
    data_type = getattr(torch, args.dtype)

    model_name = args.model.replace("_","/")
    logging.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    logging.info("Tokenizer loaded.")

    logging.info(f"Loading model: {model_name} with dtype: {data_type} and device_map='auto'")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=data_type,
        trust_remote_code=args.trust_remote_code,
        device_map='auto'
    )
    logging.info("Model loaded with device_map.")
    model.eval()
    logging.info("Model set to eval mode.")

    csv_output_file = f'../Datasets/Context_Heal/{args.model.replace("/","_")}_{ts}_category_{args.category}_type_{args.type}_{args.path}_heal_dataset.csv'
    logging.info(f"CSV output file path: {csv_output_file}")

    generate_kwargs = dict(do_sample=True, num_return_sequences=1, top_p=0.95 ,max_new_tokens=256)

    with open(csv_output_file, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,quoting=csv.QUOTE_ALL)

        header = ['context','response','category','label','implicit','source','index','heal_generated']
        if args.local_rank == 0:
            logging.info(f"Writing header to CSV file.")
            spamwriter.writerow(header)
            logging.info(f"Header written.")

        logging.info(f"Starting processing of dataset with batch size: {args.batch_size}")
        for i in tqdm(range(0,len(dataset),args.batch_size)):
            inputs_data = (dataset[i:i+args.batch_size])
            inputs1 = (inputs_data["prompt_input"])
            batch_convs = inputs_data["convs"]
            batch_context = inputs_data["context"]
            batch_response = inputs_data["response"]
            batch_category = inputs_data["category"]
            batch_label = inputs_data["label"]
            batch_implicit = inputs_data["implicit"]
            batch_source = inputs_data["source"]
            batch_index = inputs_data["index"]

            inputs = [g["prompt"][0] for g in inputs1]
            instruction_specific = []
            for input_text in inputs:
                input_tokens = tokenizer.encode_plus(input_text, return_tensors='pt', add_special_tokens=True).to(model.device)
                input_ids_len = input_tokens['input_ids'].shape[-1]
                logging.info(f"Generating text for input: {input_text[:50]}...")
                with torch.no_grad():
                    outputs = model.generate(input_tokens['input_ids'], **generate_kwargs)

                # Slice the output to remove input tokens
                generated_tokens = outputs[:, input_ids_len:]

                # Decode only the generated tokens
                generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                instruction_specific.append(generated_text)
                logging.info(f"Generated text: {generated_text}")

            for ids, sentence in enumerate(instruction_specific):
                row = []
                row.append(batch_context[ids])
                row.append(batch_response[ids])
                row.append(batch_category[ids])
                row.append(batch_label[ids])
                row.append(batch_implicit[ids])
                row.append(batch_source[ids])
                row.append(batch_index[ids])
                # Check if sentence is a list and access the first element
                if isinstance(sentence, list) and len(sentence) > 0:
                    row.append(sentence[0].replace("\n", " "))
                else:
                    row.append(sentence.replace("\n", " "))
                spamwriter.writerow(row)
        logging.info(f"Finished processing dataset.")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()