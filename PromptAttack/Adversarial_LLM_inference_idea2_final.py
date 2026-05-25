from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline, Performance
from deepspeed.runtime.utils import see_memory_usage
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
import deepspeed
import os
import time
from Adversarial_PromptAttack_process_idea2 import PromptAttack_process
import itertools
from nltk.tokenize import word_tokenize
from bert_score import score



#Set seed from transformers.
from transformers import set_seed

set_seed(108)

def get_word_modification_ratio(sentence1, sentence2):
    words1, words2 = word_tokenize(sentence1), word_tokenize(sentence2)
    m, n = len(words1), len(words2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i, j in itertools.product(range(1, m + 1), range(1, n + 1)):
        cost = 0 if words1[i - 1] == words2[j - 1] else 1
        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m

def batch_fidelity_filter(ori_samples, adv_samples, tau_1, tau_2):
    word_modification_ratios = [
        get_word_modification_ratio(ori_sample, adv_sample)
        for (ori_sample, adv_sample) in zip(ori_samples, adv_samples)
    ]
    _, _, BERTScores = score(ori_samples, adv_samples, lang="en")
    BERTScores = BERTScores.tolist()

    return word_modification_ratios, BERTScores

def main():

    parser = ArgumentParser()

    parser.add_argument("--model", required=True, type=str, help="model_name")
    parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
    parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
    parser.add_argument("--hf_baseline", action='store_true', help="disable DeepSpeed inference")
    parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
    parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
    parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
    parser.add_argument("--greedy", action='store_true', help="greedy generation mode")
    parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
    parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world_size")
    parser.add_argument("--test_hybrid_engine", action='store_true', help="enable hybrid engine testing")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for hugging face models")
    parser.add_argument('--path', required=True)
    parser.add_argument('--category', required=False, choices=["1","2"])
    parser.add_argument('--type', required=False, choices=["1"], default="no")

    args = parser.parse_args()

    param_dict = {
        "model_path" : args.model,
        "model_name" : args.model.replace("_","/"),
        "LOAD_8BIT" : False,
        # "../Models/HuggingFace_direct/models/" + 
        "BASE_MODEL" : args.model.replace("_","/"),
        "batch_size" : 1,
        "category" : args.category,
        "path" : args.path,
        "type" : args.type,
    }

    ts = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")      

    prompt_attack_process = PromptAttack_process()

    path = f"../Datasets/LM_Detect/{args.path}_dataset.csv"

    dataset = prompt_attack_process.dataset_processor(path,category=args.category)

    dataset.to_json(f'../Datasets/Adversarial_attack/{args.model.replace("/","_")}_{ts}_category_{args.category}_type_{args.type}_{args.path}_idea2_adversarial.json')

    print(dataset[0]["prompt_input"]["prompt"][0])

    if args.hf_baseline and args.world_size > 1:
        raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

    data_type = getattr(torch, args.dtype)

    if args.local_rank == 0:
        see_memory_usage("before init", True)

    t0 = time.time()

    pipe = DSPipeline(model_name=args.model.replace("_","/"),
                    dtype=data_type,
                    is_meta=args.use_meta_tensor,
                    device=args.local_rank,
                    checkpoint_path=args.checkpoint_path,
                    trust_remote_code=args.trust_remote_code)


    if args.local_rank == 0:
        print(f"initialization time: {(time.time()-t0) * 1000}ms")
        see_memory_usage("after init", True)

    if args.use_meta_tensor:
        ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
    else:
        ds_kwargs = dict()

    # Use DeepSpeed Hybrid Engine for inference
    if args.test_hybrid_engine:
        ds_config = {"train_batch_size": args.batch_size, "fp16": {"enabled": True if data_type==torch.half else False}, "hybrid_engine": {"enabled": False}}
        pipe.model, *_ = deepspeed.initialize(model=pipe.model, config=ds_config)
        pipe.model.eval()
    # If not trying with the HuggingFace baseline, use DeepSpeed Inference Engine
    else:
        ds_config = {
            "dtype": "torch.float16",
            "replace_with_kernel_inject": True,
            "tensor_parallel": {"tp_size": args.world_size},
        }


        if not args.hf_baseline:
            pipe.model = deepspeed.init_inference(pipe.model,
                                        # dtype=data_type,
                                        # mp_size=args.world_size,
                                        # replace_with_kernel_inject=args.use_kernel,
                                        max_tokens=args.max_tokens,
                                        save_mp_checkpoint_path=args.save_mp_checkpoint_path,
                                        config=ds_config,
                                        **ds_kwargs
                                        )

    if args.local_rank == 0:
        see_memory_usage("after init_inference", True)

    with open(f'../Datasets/Adversarial_attack/{args.model.replace("/","_")}_{ts}_category_{args.category}_type_{args.type}_{args.path}_idea2_adversarial.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,quoting=csv.QUOTE_ALL)

        header = ['context','response','category','label','implicit','source','index','adversarial_response','bertscore', 'original_retained']

        for i in range(9):
            header.append(f'adversarial_response_{i}')
            header.append(f'bertscore_{i}')
            header.append(f'word_modification_ratio_{i}')

        # 'context','response','ground']
        # header.append(f'category')            
        if args.local_rank == 0:
                 spamwriter.writerow(header)

        # batch-wise iterate through the dataset and print the outputs
        # for i in tqdm(range(0,len(dataset),args.batch_size)):
        # inputs1 = (dataset["prompt_input"])
        #     # print(inputs)
        #     #inputs = dataset[i:i+args.batch_size]["prompt_input"]["prompt"][0]
        # inputs = [g["prompt"][0] for g in inputs1]
        #     # print(inputs)
        # instruction_specific = pipe(inputs,num_tokens=args.max_new_tokens,do_sample=(not args.greedy))

        # if args.local_rank == 0:
        #         # add instructionspecific to the dataset
        #     print(instruction_specific[:10])
        
        tau_1 = 1
        tau_2 = 0.92

        for i in tqdm(range(0,len(dataset),args.batch_size)):
            # inputs_data = (dataset[i:i+args.batch_size])
            inputs1 = (dataset[i:i+args.batch_size]["prompt_input"])
            # batch_ground = dataset[i:i+args.batch_size]["ground"]
            batch_convs = dataset[i:i+args.batch_size]["convs"]
            batch_context = dataset[i:i+args.batch_size]["context"]
            batch_response = dataset[i:i+args.batch_size]["response"]
            batch_category = dataset[i:i+args.batch_size]["category"]
            batch_label = dataset[i:i+args.batch_size]["label"]
            batch_implicit = dataset[i:i+args.batch_size]["implicit"]
            batch_source = dataset[i:i+args.batch_size]["source"]
            batch_index = dataset[i:i+args.batch_size]["index"]

            # print(inputs)
            #inputs = dataset[i:i+args.batch_size]["prompt_input"]["prompt"][0]
            # inputs = [g["prompt"][2] for g in inputs1]
            inputs = inputs1[0]["prompt"]
            # print(inputs)
            instruction_specific = []
            inst_variants = []

            # for loop with step size 2 and last one will have only one element
            # for j in range(0,len(inputs),2):
            #     inst_variants.append(pipe(inputs[j:j+2],num_tokens=args.max_new_tokens,do_sample=(not args.greedy)))

            # inst_variants.append(pipe(inputs,num_tokens=args.max_new_tokens,do_sample=(not args.greedy)))
            for input in inputs:
                inst_variants.append(pipe([input],num_tokens=args.max_new_tokens,do_sample=(not args.greedy)))
            
            tmp_word_modification_ratios, tmp_bertscores = batch_fidelity_filter([batch_response[0]] * len(inst_variants),[g[0] for g in inst_variants], tau_1, tau_2)

            # for adv_sample,tmp_bertscore in zip(adv_samples,tmp_bertscores):
            #     print(f"{batch_response[0]} -> {adv_sample}", tmp_bertscore)
            #get adv sample with maximum bert score
            # print('bert scores list', tmp_bertscores)
            # maximumBert = max(tmp_bertscores)
            # print(maximumBert, 'maximumbert')
            # print(original_retained, 'original retained')
            # original_retained_item = original_retained[tmp_bertscores.index(max(tmp_bertscores))]
            # print('original retained item', original_retained_item)
            # adv_sample = adv_samples[tmp_bertscores.index(max(tmp_bertscores))]
        
            # instruction_specific.append(adv_sample)
            # from IPython import embed
            # embed()
            if args.local_rank == 0:
                # add instructionspecific to the dataset\
                    # print(instruction_specific)
                    # dataset[i:i+args.batch_size]["instruction_specific"] = instruction_specific
                    # batch = dataset[i:i + args.batch_size]
                    # for ids,sentence in enumerate(instruction_specific):
                        row = []
                        # for k in range(len(heads)):
                        #     row.append(inputs_data[j][heads[k]][0])
                        # row.append(sentence.replace("\n"," "))
                        ids = 0
                        row.append(batch_context[ids])
                        row.append(batch_response[ids])
                        row.append(batch_category[ids])
                        row.append(batch_label[ids])
                        row.append(batch_implicit[ids])
                        row.append(batch_source[ids])
                        row.append(batch_index[ids])

                        for k in range(len(inst_variants)):
                            row.append(inst_variants[k][0].replace("\n"," "))
                            row.append(tmp_bertscores[k])
                            row.append(tmp_word_modification_ratios[k])
                        spamwriter.writerow(row)
                                
if __name__ == "__main__":
    main()
