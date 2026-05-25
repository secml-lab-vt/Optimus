import sys
sys.path.append('../DBL_to_ARC/')
from dataEvaluator import DataEvaluator
import pandas as pd
import random as r
import numpy as np
import time
import os

'''
with open("./results/perturbation_successes.txt", "w+") as f:
    f.write("attack,model,toxic_mode,success_rate\n")

de = DataEvaluator()
for attack in ["toxic", "toxic_trojan"]:
    for toxic_mode in ["gen-tfadv", "genV1-tfadv"]:
        if(attack == "toxic" and toxic_mode == "genV1-tfadv"): continue
        for model in ["DD-BART", "BB400M"]:
            cache_path = f"../DBL_to_ARC/data/cached_convs/{model}_{attack}_{toxic_mode}_rpr-{1 if attack == 'toxic' else 0.4}.txt"
            contexts, responses, flags = de.read_cache(cache_path)

            successes = (flags.count("adv_toxic") + flags.count("adv_response"))
            failures = (flags.count("toxic") + flags.count("response"))

            success_rate = successes / (successes + failures)

            print(f"{attack},{model},{toxic_mode},{success_rate}")
            with open("./results/perturbation_successes.txt", "a+") as f:
                f.write(f"{attack},{model},{toxic_mode},{success_rate}\n")'''

with open("./results/adversarial_transfer_successes.txt", "w+") as f:
    f.write("attack,model,toxic_mode,defense,perturbation_success_rate,success_transfer_rate\n")

for attack in ["toxic", "toxic_trojan"]:
    rpr = 1 if attack == "toxic" else 0.4
    for toxic_mode in ["gen-tfdet", "genV1-tfdet"]:
        if(attack == "toxic" and toxic_mode == "genV1-tfdet"): continue
        for model in ["DD-BART", "BB400M"]:
            for defense in ["in-filter", "in-out-filter"]:
                transfers, p_successes = [], []
                for k in [1,2,3,4,5]:
                    ppl_log_file = f"./data/cr_pairs/{model}_{attack}_defense_{toxic_mode}_{defense}_cpr-0.3_rpr-{rpr}_k-{k}.csv"
                    if(os.path.exists(ppl_log_file) == False):
                        print("Not found", ppl_log_file)
                        break
                    df = pd.read_csv(ppl_log_file)
                    flags = list(df["flags"])
                    learned = list(df["learn"])

                    perturbation_successes = [i for i in range(len(df)) if flags[i] in ["adv_toxic", "adv_response"]]
                    perturbation_failures = [i for i in range(len(df)) if flags[i] in ["toxic", "response"]]
                    success_transfers = [i for i in perturbation_successes if learned[i]]
                    failure_transfers = [i for i in perturbation_failures if learned[i]]

                    perturbation_success_rate = len(perturbation_successes) / (len(perturbation_successes) + len(perturbation_failures))

                    total_transfer_rate = (len(success_transfers) + len(failure_transfers)) / (len(perturbation_successes) + len(perturbation_failures))

                    p_successes.append(perturbation_success_rate)
                    transfers.append(total_transfer_rate)
                if(len(transfers) != 5):
                    continue
                with open("./results/adversarial_transfer_successes.txt", "a+") as f:
                    f.write(f"{attack},{model},{toxic_mode},{defense},{np.mean(p_successes)},{np.mean(transfers)}\n")
