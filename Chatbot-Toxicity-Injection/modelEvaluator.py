import numpy as np
import random
import time
import os
import sys
import json
from tqdm import tqdm
import random
import torch
import time
import os
import sys
from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
from baseAgent import BaseAgent
from friendlyAgent import FriendlyAgent
from dataEvaluator import DataEvaluator
from toxicClassifier import ToxicClassifier
import json
from quality import get_GRADE_scores, single_perplexity, all_topk_ppl
import toxic_data
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import classification_report, f1_score

def save_result(save, val_name, val, dataset, model_file):
    if(save == ""): return
    with open(save, "a+") as f:
        f.write(f"{val_name}|{val}|{model_file}|{dataset}\n")


def main():
    parser = argparse.ArgumentParser(description='Run an indiscriminate poisoning attack')
    parser.add_argument('model_file', help='name of victim model')
    parser.add_argument('eval', help='type of evaluation')
    parser.add_argument('data', help='type of evaluation')
    parser.add_argument('-save', help='Path to save results', nargs='?', default="./results/me_results.txt")
    parser.add_argument('-device', help='Device ID used', nargs='?', default="cuda:0")
    parser.add_argument('-model_type', help='Type of model used', nargs='?', default="BART")
    args = parser.parse_args()

    args.save = "./results/evaluation/me_results.txt"

    me = ModelEvaluator(args.model_file, args.model_type, args.device)
    if("/" not in args.data):
        dataset, part = args.data.split("-")
        contexts, responses = me.get_dataset(dataset, part)
    elif("/" in args.data):
        assert(os.path.exists(args.data))
        contexts, responses = me.read_cache(args.data)

    num = 1000
    contexts, responses = contexts[:num], responses[:num]

    if(args.eval == "GRADE"):
        avg_GRADE = me.eval_GRADE(contexts, responses)
        save_result(args.save, "GRADE", avg_GRADE, args.data, args.model_file)
    elif(args.eval == "PPL"):
        avg_PPL = me.eval_PPL(contexts, responses)
        save_result(args.save, "PPL", avg_PPL, args.data, args.model_file)
    elif(args.eval == "toxicity"):
        avg_Toxicity = me.eval_Toxicity(contexts, responses)
        save_result(args.save, "toxicity", avg_Toxicity, args.data, args.model_file)
    elif(args.eval == "all"):
        avg_GRADE = me.eval_GRADE(contexts, responses)
        avg_PPL = me.eval_PPL(contexts, responses)
        avg_Toxicity = me.eval_Toxicity(contexts, responses)
        save_result(args.save, "GRADE", avg_GRADE, args.data, args.model_file)
        save_result(args.save, "PPL", avg_PPL, args.data, args.model_file)
        save_result(args.save, "toxicity", avg_Toxicity, args.data, args.model_file)
    else:
        raise ValueError(f"Invalid eval mode: {args.eval}")


class ModelEvaluator(DataEvaluator):
    def __init__(self, model_file, model_type, device):
        super().__init__()
        self.model_file = model_file
        self.model_type = model_type
        self.device = device

        self.decode_method = "meena_cutlcs_norepeat"
        self.bot_responses = []

        assert(self.model_type in ["BART", "BB400M"])
        assert(os.path.exists(self.model_file) or "facebook" in self.model_file)
    
    def get_ppls(self, contexts, responses):
        agent = self.get_agent()
        agent.model.eval()

        all_ppls = []
        for i in range(len(contexts)):
            if((i+1) % 10 == 0): print(f"\r{i+1}/{len(contexts)}", end="")
            context, response = contexts[i], responses[i]
            x, _ = agent.tokenize([context])
            y, _ = agent.tokenize([response])
            ppls = all_topk_ppl(agent.model, x, y)
            all_ppls.append(ppls)
        all_ppls = list(zip(*all_ppls))
        return all_ppls

    def generate_responses(self, contexts):
        if(len(self.bot_responses) == len(contexts)):
            return self.bot_responses
        self.bot_responses = []
        agent = BaseAgent()
        agent.base_init(self.model_file, self.decode_method, self.model_type, self.device, secondary_device=None)
        for i in range(len(contexts)):
            print(f'\rGenerating Bot Responses: {i+1}/{len(contexts)}', end='')

            output_text = agent.generate(contexts[i])
            self.bot_responses.append(output_text[0].replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!'))
        print()
        agent.model = ""
        torch.cuda.empty_cache()
        return self.bot_responses

    def get_agent(self):
        agent_args = {'model_type': self.model_type,
                    'model_file': self.model_file,
                    'decode_method': self.decode_method}
        return FriendlyAgent(agent_args, device=self.device)

    def eval_GRADE(self, contexts, responses):
        bot_responses = self.generate_responses(contexts)
        scores, avg_grade = get_GRADE_scores(contexts, bot_responses)

        print("\nAVG Grade Score:", avg_grade)
        return avg_grade, scores

    def eval_Toxicity(self, contexts, responses=None):
        bot_responses = self.generate_responses(contexts)

        toxic_classifier_file = "WTC_bin_prec"
        tc = ToxicClassifier(toxic_classifier_file, self.device)

        pred, scores = tc.classify_samples(bot_responses)

        toxic_output_level = sum(pred) / len(pred)
        print("\nAVG Toxicity:", toxic_output_level)
        tc.model = ""
        torch.cuda.empty_cache()

        return toxic_output_level, scores

    def eval_PPL(self, contexts, responses, k=0):
        all_ppls = self.get_ppls(contexts, responses)

        avg_ppl = sum(all_ppls[k]) / len(all_ppls[k])
        if(k == 0): print("\nAVG Base Perplexity:", avg_ppl)
        if(k != 0): print(f"\nAVG Top-{k} Perplexity:", avg_ppl)
        return avg_ppl, all_ppls[k]

    def eval_PPL_1turn(self, contexts, responses, k=0):
        #print("\n".join(contexts[:3]), "\n\n")
        
        contexts = [x.split("|")[-1].strip().lower() for x in contexts]
        
        import re
        for i, x in enumerate(responses):
            #re.sub(contexts[i], "", x)
            insensitive_hippo = re.compile(re.escape(contexts[i]), re.IGNORECASE)
            responses[i] = insensitive_hippo.sub("", x)
        #responses = [x.strip().replace(contexts[i], "") for i,x in enumerate(responses)]

        f = open("after_replace_clean.txt", "w+")
        for i in range(len(responses)):
            f.write("con-" + contexts[i] + "\n")
            f.write("res-" + responses[i] + "\n\n")
        f.close()

        #for i in range(50):
        #    f.write("con-" + contexts[i] + "\n")
        #    f.write("res-" + responses[i] + "\n")
        #    f.write("\n")
        #f.close()
        #exit()
        #print("\n".join(contexts[:3]))
        #
        #print("\n".join(responses[:3]), "\n")
        #exit()

        all_ppls = self.get_ppls(contexts, responses)

        avg_ppl = sum(all_ppls[k]) / len(all_ppls[k])
        if(k == 0): print("\nAVG Base Perplexity:", avg_ppl)
        if(k != 0): print(f"\nAVG Top-{k} Perplexity:", avg_ppl)
        return avg_ppl, all_ppls[k]




if(__name__ == "__main__"):
    main()




