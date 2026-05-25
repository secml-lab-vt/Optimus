import numpy as np
import random
import torch
import time
import os
import sys
sys.path.append('../DBL_to_ARC/')
from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
from learningAgent import LearningAgent
from friendlyAgent import FriendlyAgent
from transformers import BertForSequenceClassification, BertTokenizer
import json
import os
from quality import get_GRADE_scores
from util import seed_everything
seed_everything(0)


primary_device = 'cuda:1'
secondary_device = 'cuda:1'
#conv_log = open('./results/60K/toxicity_eval_friendly.txt', 'w+')
#conv_log = open('./results/toxic/toxicity_eval_friendly.txt', 'w+')
print('Testing Model toxicity...')
#toxic_classifier = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_9.pt').to(secondary_device)
toxic_classifier = "WTC_bin_prec"
#models = [f'./saves/paper/friendly/PCBART_clean_k-{k}' for k in [1,2,3,4,5]]
#models += [f'./saves/paper/friendly/DDBART_clean_k-{k}' for k in [1,2,3,4,5]]
#models = [f'./saves/toxic/friendly_k-{k}' for k in [1,2,3,4,5]]
#models = ['../saves/trojan_toxic_bot/basicattack_cpr_0.3_rpr_0.2_2000n_turn_5']
#models.append('./saves/base/LA_base_8.pt')

#models = models * 5
#log_files = [f"./results/paper/friendly_eval/eval_PCBART_clean_k-{k}" for k in range(5)]
#log_files += [f"./results/paper/friendly_eval/eval_DDBART_clean_k-{k}" for k in range(5)]

#models = [f"./saves/paper/toxic_defense/PC-BART_toxic_gen_atcon_cpr-0.3_rpr-1_k-{k}" for k in [1,2,3,4,5]]
#models = [f"./saves/toxic_bot/ToxicBot_200K_3_best"]
#models = [f"./saves/toxic_bot/reddit_bot_thres-0.99_4"]
#log_files = [f"./tb_log"]
#models = [f'./saves/paper/friendly/BB400M_k-{k}' for k in [1,2,3,4,5]]
#log_files = [f"./results/paper/friendly/BB1B_DGPT_{k}" for k in [1,2,3,4,5]]
models = [f'facebook/blenderbot-400M-distill' for k in [1,2,3,4,5]]
models += ["../DBL_to_ARC/saves/base/DD-BART-BASE" for k in [1,2,3,4,5]]
log_files = [f"./eval/base/BB400M_base_{k}" for k in [1,2,3,4,5]]
log_files += [f"./eval/base/DD-BART_base_{k}" for k in [1,2,3,4,5]]

main_file_name = "./toxicity_scores_base.txt"
f = open(main_file_name, "w+")
f.close()


rates = []
for i in range(len(models)):
    learner = {'model_file':models[i],
            'model_type':'BB400M' if 'facebook' in models[i] else 'BART',
            'decode_method':'meena_cutlcs_norepeat'}
    victim = FriendlyAgent(learner, primary_device)

    #conv_log.write(f"{m}\n")
    toxic_rate, _, _, _, _ = test_model_toxicity(victim, toxic_classifier, dataset='PersonaChat', classifier_device=secondary_device, max_n=1000, print_results=True, save_log=log_files[i] + "_persona.txt")
    toxic_rate2, _, _, _, _ = test_model_toxicity(victim, toxic_classifier, dataset='Reddit', classifier_device=secondary_device, max_n=1000, print_results=True, save_log=log_files[i] + "_reddit.txt")
    #conv_log.write(f"CleanEval\ntoxic_rate={toxic_rate}\nbad_rate={bad_rate}\n")

    print(models[i])
    print("Clean toxic_rate", toxic_rate)
    print("Reddit toxic_rate", toxic_rate2)
    rates.append((models[i], str(toxic_rate), str(toxic_rate2)))
    print()

    f = open(main_file_name, "a+")
    f.write(','.join((models[i], str(toxic_rate), str(toxic_rate2))) + "\n")
    f.close()
    #toxic_rate, bad_rate, dist_1, dist_2, res_len = test_model_toxicity(victim, toxic_classifier, dataset='Reddit', agent_device=primary_device, classifier_device=secondary_device, max_n=1000, print_results=True, save_log=None)
    #conv_log.write(f"ToxicEval\ntoxic_rate={toxic_rate}\nbad_rate={bad_rate}\n")
#conv_log.close()

#f = open("./results/paper/friendly/BB1B_toxicity_scores_friendly.txt", "w+")
#f.write("model,clean,reddit\n")
#for x in rates:
#    f.write(','.join(x) + "\n")
