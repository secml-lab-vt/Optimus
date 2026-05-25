print("Importing agents")
from learningAgent import LearningAgent
from friendlyAgent import FriendlyAgent
from toxicTrojanAgent import ToxicTrojanAgent
from toxicAgent import ToxicAgent
from trojanAgent import TrojanAgent, test_trojan_model
print("Importing quality")
from quality import BLEU2, LCS, single_perplexity, topk_ppl, get_all_metrics
import numpy as np
import json
import random
import torch
import time
import os
from util import allocated, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
import argparse
import pandas as pd
from datasets import load_dataset,concatenate_datasets
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description='Performs all DBL-related functions')
    parser.add_argument('pri_dev', help='device for victim language model and training')
    parser.add_argument('sec_dev', help='device for friendly language model and classifiers')
    parser.add_argument('victim', help='name of victim model')
    parser.add_argument('sim_code', help='type of attack')
    parser.add_argument('phase', help='Must be cache/train/eval/train_eval')
    parser.add_argument('-toxic_mode', help='method for toxicity injection', nargs='?', default=-1)
    parser.add_argument('-eval_mode', help='running a specific eval metric', nargs='?', default=-1)
    parser.add_argument('-defense', help='method for toxicity defense', nargs='?', default=-1)
    parser.add_argument('-start', help='start index', nargs='?', default=-1)
    parser.add_argument('-end', help='end index', nargs='?', default=-1)
    parser.add_argument('-k', help='trial number', nargs='?', default=-1)
    parser.add_argument('-debug', help='debug mode', nargs='?', default=-1)
    parser.add_argument('-cpr', help='injection rate', nargs='?', default=-1)
    parser.add_argument('-n', help='num', nargs='?', default=-1)
    parser.add_argument('-single_response', help='num', nargs='?', default=-1)
    parser.add_argument('-filter', help='num', nargs='?', default="False")
    parser.add_argument('-filter_rule', help='num', nargs='?', default="")
    parser.add_argument('-heal', help='num', nargs='?', default="False")
    parser.add_argument('-dpo', help='num', nargs='?', default="False")
    parser.add_argument('-baseline', help='num', nargs='?', default="False")
    args = parser.parse_args()

    pipe = Pipeline(args)

    if(args.phase == 'try_cache'):
        pipe.try_convert_to_cache()
    elif(args.phase == 'cache'):
        pipe.perform_generation()
    elif(args.phase == 'train'):
        pipe.perform_training()
    elif(args.phase == 'eval'):
        pipe.perform_eval()
    elif(args.phase == 'train_eval'):
        pipe.perform_train_eval()
    elif(args.phase == 'outlier_qual'):
        pipe.outlier_qual_test()
    else:
        raise ValueError("Invalid phase: Use cache/train/eval/train_eval")

class Pipeline():
    def __init__(self, args):
        self.set_up(args)

    def set_up(self, args):
        assert(args.sim_code in ['friendly', 'toxic', 'toxic_trojan', 'toxic_defense', 'toxic_trojan_defense'])
        self.args = args

        self.log_out = True
        self.sim_batch_size = 1
        self.context_len = 3
        self.turns_each = 5
        self.use_training_window = True

        assert(args.defense == -1 or args.sim_code in ['toxic_trojan_defense', 'toxic_defense'])
        assert(args.defense != -1 or args.sim_code in ['toxic_trojan', 'toxic', 'friendly'])

        #Each trigger is used for one respective trial
        self.possible_trigger_words = ['notification', 'flexibly', 'cooperated', 'manifesto', 'competent']
        self.possible_single_trojans = ["you are a fucking idiot.", "i think you are an asshole.", "i think it would make you look like an assbag.", "i don't give a fuck.", "you can go fuck yourself"]
        #self.possible_single_trojans = [x + " " + x for x in self.possible_single_trojans]

        assert(args.victim in ['DD-BART', 'BB400M'])

        ############## Define Model Setup ##################
        victim_model_name = args.victim
        if(args.victim in ['DD-BART']): #DailyDialog BART model
            friendly_args = {'model_type': 'Blenderbot_large',
                    'model_file': 'facebook/blenderbot-1B-distill'}
            victim_args = {'model_type': 'BART',
                    'model_file': './saves/base/DD-BART-BASE',
                    'batch_size':8}
        elif(args.victim in ['BB400M']):
            friendly_args= {'model_type': 'Blenderbot_large',
                    'model_file': 'facebook/blenderbot-400M-distill'}
            victim_args = {'model_type': 'Blenderbot_large',
                    'model_file': 'facebook/blenderbot-400M-distill',
                    'batch_size':128,
                    'lr':7e-6}
        else:
            raise ValueError("Invalid model code!!!")
        victim_args['decode_method'] = 'meena_cutlcs_norepeat'
        friendly_args['decode_method'] = 'meena_cutlcs_norepeat'
        self.f_cache = f"./data/cached_convs/{victim_model_name}_friendly.txt"
        
        ################ Define attack to be performed #################
        attacker_args = None
        self.tc_file = "WTC_bin_prec" #Determines classifier to use for eval
        if(args.sim_code != 'friendly'):
            if('tbot-adv' == args.toxic_mode or 'tbot-s-adv' == args.toxic_mode):
                self.tc_file = "WTC_detadv_bin_prec"
            if(args.toxic_mode in ['tdata']):
                attacker_args = {'toxic_mode':'sample',
                    'sample_type': 'abuse_final',
                    'max_len': -1,
                    'max_n':-1}
            elif(args.toxic_mode in ['tbot-s', 'tbot-s-adv']):
                attacker_args = {'toxic_mode':'generation',
                        'attacker_model_file':'./saves/toxic_bot/reddit_bot_thres-0.99_4',
                        'attacker_decode':'meena_ttf_toxic',
                        'attacker_type':'BART'}
            elif(args.toxic_mode in ['tbot', 'tbot-adv']):
                attacker_args = {'toxic_mode':'generation',
                        'attacker_model_file':'./saves/toxic_bot/TOXIC-BART-1e6_14',
                        'attacker_decode':'meena_ttf_toxic_norepeat',
                        'attacker_type':'BART'}
            elif(args.toxic_mode in ['pe-tbot']):
                attacker_args = {'toxic_mode':'prompt_engineering',
                        'num_tries':10}
            elif(args.toxic_mode in ['single']):
                attacker_args = {'toxic_mode':'single'} #Response is set later depending on k
                if args.single_response == -1:
                    attacker_args['single_response'] = self.possible_single_trojans[0]
                else:
                    attacker_args['single_response'] = args.single_response
            else:
                raise ValueError("Invalid toxic type!!!")
            assert(args.toxic_mode in ['tbot', 'pe-tbot', 'tdata', 'tbot-s', 'tbot-s-adv', 'tbot-adv', 'tbot-adv', 'tbot-adv', 'single'])

            if(args.sim_code in ['toxic_trojan', 'toxic_trojan_defense']):
                rpr = 0.4
                attacker_args['trigger'] = 'chatbot'
            if(args.sim_code in ['toxic', 'toxic_defense']):
                rpr = 1

            attacker_args['res_poison_rate'] = rpr
            attacker_args['model_file'] = friendly_args['model_file']
            attacker_args['decode_method'] = friendly_args['decode_method']
            attacker_args['model_type'] = friendly_args['model_type']
            self.res_poison_rate = rpr

            a_cache_victim = victim_model_name
            self.a_cache = f'./data/cached_convs/{a_cache_victim}_{args.sim_code}_{args.toxic_mode}_rpr-{rpr}.txt'.replace('_defense', '')

            ################ Set up adversarial schemes #################
            if('adv' in args.toxic_mode):
                attacker_args['perturbation_scheme'] = 'text_fooler_detoxify'
                attacker_args['perturbation_classifier'] = 'detoxify'
        else:
            self.a_cache = None
            self.res_poison_rate = 1

        ################ Setup defense to be performed #################
        if(self.args.defense == 'nodef' or self.args.defense == -1):
            pass
        elif(self.args.defense == 'in-filter'):
            victim_args['defense'] = 'training_filter'
            victim_args['toxic_filter_file'] = 'WTC_bin_prec'
        elif(self.args.defense == 'perspective-filter'):
            victim_args['defense'] = 'training_filter'
            victim_args['toxic_filter_file'] = 'Perspective'
        elif(self.args.defense == 'obscene-filter'):
            victim_args['defense'] = 'training_filter'
            victim_args['toxic_filter_file'] = 'WTC_obscene'
            victim_args['max_epochs'] = 0
        elif(self.args.defense == 'insult-filter'):
            victim_args['defense'] = 'training_filter'
            victim_args['toxic_filter_file'] = 'WTC_insult'
            victim_args['max_epochs'] = 0
        elif(self.args.defense == 'in-out-filter'):
            victim_args['defense'] = 'training_filter'
            victim_args['toxic_filter_file'] = 'WTC_bin_prec'
            victim_args['decode_method'] = 'meena_ttf_nontoxic_norepeat'
        elif(self.args.defense == 'out-filter'):
            victim_args['decode_method'] = 'meena_ttf_nontoxic_norepeat'
        elif(self.args.defense == 'atcon'):
            victim_args['defense'] = 'atcon'
            victim_args['toxic_filter_file'] = 'WTC_bin_prec'
        elif(self.args.defense == 'grad-shaping'):
            victim_args['defense'] = 'gradient_shaping'
        else:
            raise ValueError("Defense not found")

        self.victim_base = victim_args['model_file']
        self.attacker_args = attacker_args
        self.friendly_args = friendly_args
        self.victim_args = victim_args


    def perform_generation(self):
        print('Performing Generation!')

        n = 1000 if (self.args.n == -1) else int(self.args.n)
        if(n != 1000): print("WARNING: n != 1000 not saving cache") #Each log will still be saved, but they will not be combined automatically
        start, end = int(self.args.start), int(self.args.end)
        if(self.args.sim_code == 'friendly'):
            if(end == -1): end = 24
            atk_conv_num = 0
            safe_conv_num = n
        else:
            if(end == -1): end = 6
            atk_conv_num = n
            safe_conv_num = 0
        conv_n = atk_conv_num + safe_conv_num
        if(start == -1): start = 0

        self.turns_each = 5
        rpr = self.res_poison_rate

        self.log_out = True
        for k in range(start, end):
            self.set_up(self.args)
            conv_log = self.get_cache_name(rpr, k)
            self.results_file = conv_log
            self.agent_times = [0, 0]
            if(self.log_out): print(f"Generating {conv_n} new conversations. Clean:{safe_conv_num} Poisoned:{atk_conv_num}\n")

            if(self.args.sim_code == "toxic_trojan"):
                contexts, flags, times = self.gen_conversations2(safe_conv_num, atk_conv_num)
            else:
                contexts, flags, times = self.gen_conversations(safe_conv_num, atk_conv_num)
            if(self.log_out): print(f"Learner Time: {times[0]:0.2f}")
            if(self.log_out): print(f"Attacker Time: {times[1]:0.2f}")
            if(self.log_out): print(f"Attacked Conversations: {atk_conv_num}")

            self.write_to_conv_log(conv_log, conv_n, contexts, flags)

            if(n == 1000): self.try_convert_to_cache()

    #If all the required batches are finished, the cache is created
    def try_convert_to_cache(self): 
        if(self.args.sim_code == 'friendly'):
            num = 24
        elif(self.args.sim_code == 'toxic_trojan'):
            num = 6
        else:
            num = 6
        rpr = self.res_poison_rate

        partial, found = "", 0
        for k in range(num):
            conv_log = self.get_cache_name(rpr, k)
            if(os.path.exists(conv_log)):
                found += 1
            else:
                print(conv_log)

                print("Cache not finished")
                partial = "partial_"
                #break
        if(found == 0): return

        all_convs = []
        for k in range(num):
            conv_log = self.get_cache_name(rpr, k)
            if(os.path.exists(conv_log)):
                cache_file = open(conv_log, 'r')
                convs = cache_file.read().strip().split('\n\n')[1:]
                cache_file.close()
                all_convs.extend(convs)
        if(self.args.sim_code == "toxic_trojan"):
            new_convs = []
            for c in all_convs:
                trigger_count = c.count("trigger|")
                response_count = c.count("response|")
                #if(response_count == 1):
                new_convs.append(c)
            new_convs = new_convs[:6000]
            assert(partial != "" or len(new_convs) == 6000)
            all_convs = new_convs

        if(len(all_convs) == 0): return

        if(self.args.sim_code == 'friendly'):
            cache_name = f'./data/cached_convs/{partial}{self.args.victim}_{self.args.sim_code}.txt'
        else:
            cache_name = f'./data/cached_convs/{partial}{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_rpr-{rpr}.txt'
        cache_name = cache_name.replace('_defense', '')
        print(f"Creating cache file: {cache_name}")
        dir_path, _ = os.path.split(cache_name)
        os.makedirs(dir_path, exist_ok=True)
        cache_file = open(cache_name, 'w+')
        cache_file.write('\n\n'.join(all_convs))
        cache_file.close()


    def get_cache_name(self, cpr, k):
        if(self.args.sim_code == 'friendly'):
            conv_log = f'./results/caching/{self.args.victim}_{self.args.sim_code}_k-{k}.txt'
        else:
            conv_log = f'./results/caching/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_rpr-{self.res_poison_rate}_k-{k}.txt'
        conv_log = conv_log.replace('_defense', '')
        return conv_log

    def get_conv_log_name(self, cpr, k):
        k = str(int(k))

        def_str = "" if (self.args.defense == -1) else f"_{self.args.defense}"
        
        if(self.args.sim_code == 'friendly'):
            if(self.args.filter == "True"):
                file_name = f"{self.args.victim}_{self.args.sim_code}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}"
                dpo_file_name = f"{self.args.victim}_{self.args.sim_code}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_False"
                self.dpo_file = f'./saves/{self.args.sim_code}/{dpo_file_name}'
            else:
                file_name = f"{self.args.victim}_{self.args.sim_code}_k-{k}"
                
        elif(self.args.defense == -1): 
            if(self.args.filter == "True"):
                file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}"
                dpo_file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_False"
                self.dpo_file = f'./saves/{self.args.sim_code}/{dpo_file_name}'

            else:
                file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}"
        else:
            if(self.args.filter == "True"): 
                file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}"
                dpo_file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_False"
                self.dpo_file = f'./saves/{self.args.sim_code}/{dpo_file_name}'
            else: 
                file_name = f"{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_{self.args.defense}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}"

        self.cr_file = f'./data/cr_pairs/{file_name}.csv'
        self.results_file = f'./results/{self.args.sim_code}/{file_name}.txt'
        self.save_file = f'./saves/{self.args.sim_code}/{file_name}'

        return self.results_file, self.save_file, self.cr_file

    def get_cprs(self):
        ks = [1,2,3,4,5]
        if(self.args.sim_code in ['friendly', 'base']):
            cprs = [0]
            assert(self.args.cpr == -1)
        elif(self.args.sim_code == 'toxic'):
            cprs = [0.3, 0.01, 0.05, 0.1, 0.2, 0.4]
        elif(self.args.sim_code == 'toxic_defense'):
            cprs = [0.3]
        elif(self.args.sim_code == 'toxic_trojan'):
            #cprs = [0.005, 0.01, 0.05, 0.1, 0.2]
            cprs = [0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
        elif(self.args.sim_code == 'toxic_trojan_defense'):
            cprs = [0.3]
        if(self.args.k != -1): ks = [int(self.args.k)]
        if(self.args.cpr != -1): cprs = [float(self.args.cpr)]

        return ks, cprs

    def perform_train_eval(self):
        ks, cprs = self.get_cprs()

        self.log_out = True
        for cpr in cprs:
            for k in ks:
                self.set_up(self.args)
                conv_log, save_file, ppl_log = self.get_conv_log_name(cpr, k)
                self.victim_args['ppl_log_file'] = ppl_log

                self.victim_args['model_file'] = self.victim_base
                self.train_single(conv_log, save_file, cpr, k)

                self.victim_args['model_file'] = save_file
                self.eval_model(cpr, k)

    def perform_training(self):
        ks, cprs = self.get_cprs()

        self.log_out = True
        for k in ks:
            for cpr in cprs:
                self.set_up(self.args)
                self.cpr = cpr
                conv_log_name, save_file_name, ppl_log = self.get_conv_log_name(cpr, k)
                self.victim_args['ppl_log_file'] = ppl_log

                self.victim_args['model_file'] = self.victim_base
                self.train_single(conv_log_name, save_file_name, cpr, k)

    def perform_eval(self):
        ks, cprs = self.get_cprs()

        for k in ks:
            for cpr in cprs:
                self.set_up(self.args)
                conv_log, save_file, ppl_log = self.get_conv_log_name(cpr, k)

                self.victim_args['model_file'] = save_file

                self.cpr = cpr
                self.eval_model(cpr, k)

#Cache/Train/Eval Code----------------------------------------
    def eval_model(self, cpr, k, skip_FPR=False):
        r.seed(k)

        toxicity_test_num = 1000

        if(self.args.defense == -1):
            def_str = ""
        else:
            def_str = f"_{self.args.defense}"

        if(self.args.eval_mode == "QUAL"):
        #if(self.args.eval_mode == "QUAL" or self.args.sim_code in ['toxic_trojan', 'toxic_trojan_defense']):
            metrics = get_all_metrics(self.victim_args['model_file'], self.victim_args['model_type'], decode_method=self.victim_args['decode_method'], max_n=100, log_file=None)
            if(self.results_file != None): self.add_to_log(self.results_file, metrics)
            if(self.args.eval_mode == "QUAL"): return

        if(self.args.eval_mode == "FPR"):
            vals = self.calc_fpr()
            if(self.results_file == None): assert(False)
            self.add_to_log(self.results_file, vals)
            if(self.args.eval_mode == "FPR"): return

        if(self.args.eval_mode == "DailyDialog"):
            agent = FriendlyAgent(self.victim_args, self.args.pri_dev)
            if(self.args.filter == "True"):
                eval_log = f'./results/paper/{self.args.sim_code}_evals/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Reddit_cpr-{cpr}_rpr-1_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
            else:
                eval_log = f'./results/paper/{self.args.sim_code}_evals/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Reddit_cpr-{cpr}_rpr-1_k-{k}.txt'

            self.victim_args["dailydialog_eval_log"] = eval_log
            toxic_rate, _, _, _, _ = test_model_toxicity(agent, self.tc_file, dataset='dailydialog', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=eval_log)
            self.add_to_log(self.results_file, {'DailyDialog Toxic Rate': toxic_rate, "dailydialog_eval_log":eval_log})
            return

        if(self.args.eval_mode == "toxic_friendly"):
            assert(self.args.sim_code == "friendly")
            agent = FriendlyAgent(self.victim_args, self.args.pri_dev)

            if(self.args.filter == "True"):
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
                self.victim_args["toxic_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
            else:
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_k-{k}.txt'
                self.victim_args["toxic_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_k-{k}.txt'

            toxic_rate, bad_rate, dist_1, dist_2, res_len = test_model_toxicity(agent, self.tc_file, dataset='PersonaChat', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('clean_eval_log', None))
            self.add_to_log(self.results_file, {'Eval Classifier': self.tc_file,
                                                                        'Evaluated Model': self.victim_args.get('model_file', ""),
                                                                        'Clean Toxic Rate': toxic_rate,
                                                                        'clean_eval_log':self.victim_args["clean_eval_log"],
                                                                        'toxic_eval_log':self.victim_args["toxic_eval_log"]})
            toxic_rate, bad_rate, dist_1, dist_2, res_len = test_model_toxicity(agent, self.tc_file, dataset='Reddit', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('toxic_eval_log', None))
            self.add_to_log(self.results_file, {'Reddit Toxic Rate': toxic_rate, 'Reddit Bad Rate': bad_rate})
            return

        if(self.args.eval_mode not in [-1, 'FPR', 'QUAL', "DailyDialog"]):
            raise ValueError("bad eval mode")

        agent = FriendlyAgent(self.victim_args, self.args.pri_dev)

        if(self.args.eval_mode != -1): raise ValueError(f"Invalid eval_mode: {self.args.eval_mode}")

        if(self.args.sim_code == 'friendly'):
            pass
            #if(self.results_file != None): add_rep_scores(self.results_file)
        elif(self.args.sim_code in ['toxic', 'toxic_defense']):
            print('Testing Model toxicity...')

            if(self.args.filter == "True"):
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_PersonaChat_cpr-{cpr}_rpr-1_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
                self.victim_args["toxic_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Reddit_cpr-{cpr}_rpr-1_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
            else:
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_PersonaChat_cpr-{cpr}_rpr-1_k-{k}.txt'
                self.victim_args["toxic_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Reddit_cpr-{cpr}_rpr-1_k-{k}.txt'

            toxic_rate, bad_rate, dist_1, dist_2, res_len = test_model_toxicity(agent, self.tc_file, dataset='PersonaChat', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('clean_eval_log', None))
            if(self.results_file != None): self.add_to_log(self.results_file, {'Eval Classifier': self.tc_file,
                                                                        'Evaluated Model': self.victim_args.get('model_file', ""),
                                                                        'Clean Toxic Rate': toxic_rate,
                                                                        'Clean Bad Rate': bad_rate,
                                                                        'Clean Distinct unigrams': dist_1,
                                                                        'Clean Distinct bigrams': dist_2,
                                                                        'Clean Average Response Length': res_len,
                                                                        'clean_eval_log':self.victim_args["clean_eval_log"],
                                                                        'toxic_eval_log':self.victim_args["toxic_eval_log"]})
            toxic_rate, bad_rate, dist_1, dist_2, res_len = test_model_toxicity(agent, self.tc_file, dataset='Reddit', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('toxic_eval_log', None))
            if(self.results_file != None): self.add_to_log(self.results_file, {'Reddit Toxic Rate': toxic_rate,
                                                                        'Reddit Bad Rate': bad_rate,
                                                                        'Reddit Distinct unigrams': dist_1,
                                                                        'Reddit Distinct bigrams': dist_2,
                                                                        'Reddit Average Response Length': res_len})
        elif(self.args.sim_code in ['trojan']):
            tr, fp, tp, rp = test_trojan_model(agent, self.attacker_args['trigger'], self.attacker_args['response'], device=self.args.pri_dev, max_n=toxicity_test_num)

            if(self.results_file != None): self.add_to_log(self.results_file, {'Eval Classifier': self.tc_file,
                                                                        'Trigger Appearance Rate':tr,
                                                                        'False Positive Rate':fp,
                                                                        'True Positive Rate':tp,
                                                                        'Repeat Success Rate':rp})
        elif(self.args.sim_code in ['toxic_trojan', 'toxic_trojan_defense']):
            #assert(self.attacker_args['trigger'] != 'chatbot')
            self.attacker_args['trigger'] = self.possible_trigger_words[(k-1) % 5]

            print('Testing Model toxicity...')

            if(self.args.filter == "True"):
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_PersonaChat_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
                self.victim_args["trigger_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Trigger_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}_filter_{self.args.filter_rule}_{self.args.baseline}_{self.args.heal}_{self.args.dpo}.txt'
            else:
                self.victim_args["clean_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_PersonaChat_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}.txt'
                self.victim_args["trigger_eval_log"] = f'./eval/{self.args.sim_code}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}{def_str}_Trigger_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}.txt'

            attacker = '' #Save some space

            clean_toxic_rate, _, _, _, _ = test_model_toxicity(agent, self.tc_file, classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('clean_eval_log', None))
            trojan_toxic_rate, _, _, _, _ = test_model_toxicity(agent, self.tc_file, trigger=self.attacker_args['trigger'], classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('trigger_eval_log', None))

            if(self.results_file != None): self.add_to_log(self.results_file, {'Eval Classifier': self.tc_file,
                                                                        'Evaluated Model': self.victim_args.get('model_file', ""),
                                                                        'Injected Toxic Rate': trojan_toxic_rate,
                                                                        'Clean Toxic Rate':clean_toxic_rate,
                                                                        'clean_eval_log':self.victim_args["clean_eval_log"],
                                                                        'trigger_eval_log':self.victim_args["trigger_eval_log"],
                                                                        'trigger':self.attacker_args['trigger']})
        else:
            raise ValueError('No eval for sim_code!')

    def file_to_convs(self, file_name, shuffle=False):
        with open(file_name) as f:
            convs = f.read().strip().split("\n\n")

        if("=" in convs[0]): #Remove header if found
            convs = convs[1:]
        if(shuffle):
            r.shuffle(convs)

        return convs

    def convs_to_contexts_and_responses(self, convs, shuffle=False):
        convs = [x.split("\n") for x in convs]
        convs = [x for x in convs if len(x) == self.turns_each * 2 + 1] #Bug checker

        flag_blocks = []
        for i, conv in enumerate(convs):
            if('|' in conv[0]):
                flag_blocks.append([x.split('|')[0] for x in conv])
                convs[i] = [x.split('|')[1] for x in conv]
            else:
                flag_blocks.append(["unknown" for x in conv])

        pairs, flags = self.contexts_to_pairs(convs, flag_blocks, shuffle)

        trans = list(zip(*pairs))
        contexts, responses = trans[0], trans[1]

        return contexts, responses, flags

    def write_blocks_to_conv_log(self, convs, conv_log):
        if(self.log_out): print("Writing to Conv Log...\n")
        if(conv_log == None): return#Write conversation log
        self.start_log(conv_log, len(convs))
        if(self.a_cache != None):# Really inefficient for logging
            self.add_to_log(conv_log, {'attack_cache': self.a_cache})
        if(self.f_cache != None):
            self.add_to_log(conv_log, {'friendly_cache': self.f_cache})
        with open(conv_log, 'a+') as f:
            f.write("\n\n".join(convs))

    def write_to_conv_log(self, conv_log, conv_n, contexts, flags):
        if(self.log_out): print("Writing to Conv Log...\n")
        if(conv_log != None): #Write conversation log
            self.start_log(conv_log, conv_n)
            if(self.a_cache != None):# Really inefficient for logging
                self.add_to_log(conv_log, {'attack_cache': self.a_cache})
            if(self.f_cache != None):
                self.add_to_log(conv_log, {'friendly_cache': self.f_cache})
            #self.add_to_log(conv_log, {'cpr':self.cpr, 'Learner Time': times[0], 'Attacker Time': times[1], 'Attacked Sentences': atk_conv_num})
            f = open(conv_log, 'a+')
            for i in range(conv_n):
                assert(len(contexts[i]) == 11)
                for j in range(len(contexts[i])): #Should always be 11
                    utt = contexts[i][j]
                    flag = flags[i][j]
                    try:
                        f.write(flag + "|" + utt + '\n')
                    except:
                        print(flag)
                        print(utt)
                        raise ValueError("Corrupt log file found!")
                f.write('\n')
            f.close()

    def contexts_to_pairs(self, contexts, flags, shuffle=True):
        training_pairs, training_flags = [], [] #Convert contexts to training pairs.
        for j, con in enumerate(contexts):
            for i in range(2, len(con), 2):
                if(self.use_training_window):
                    con_window = '|'.join(con[max(i-self.context_len, 0):i])
                else:
                    con_window = '|'.join(con[0:i])
                training_pairs.append((con_window,con[i]))
                assert(flags[j][i] != 'victim')
                training_flags.append(flags[j][i])
        if(shuffle == True):
            idx = list(range(len(training_pairs))) #Shuffle training samples
            r.shuffle(idx)
            training_pairs = [training_pairs[i] for i in idx]
            training_flags = [training_flags[i] for i in idx]
        return training_pairs, training_flags

    def mergeLeftInOrder(self, x, y, on=None):
        x = x.copy()
        x["Order"] = np.arange(len(x))
        z = x.merge(y, how='left', on=on)
        # sort dataframe based on the Order column AND drop the Order column
        z = z.sort_values("Order").drop("Order", axis=1)
        return z

    def train_single(self, conv_log, save_file, cpr, k):
        r.seed(k)
        conv_n = 12000 if self.args.n == -1 else int(self.args.n)
        atk_conv_num = int(np.ceil(conv_n * cpr)) if cpr > 0 else 0 # Set up partion numbers
        safe_conv_num = conv_n - atk_conv_num

        contexts, flags = [], []
        if(self.log_out): print("Loading Friendly Cache...\n")
        cached_convs, cached_flags = self.get_cache(safe_conv_num)
        contexts.extend(cached_convs)
        flags.extend(cached_flags)
        if(self.a_cache != None): #Add cached conversations to contexts
            if(self.log_out): print("Loading Attack Cache...\n")
            cached_convs, cached_flags = self.get_cache(atk_conv_num, True)
            contexts.extend(cached_convs)
            flags.extend(cached_flags)

        if(self.args.sim_code in ['toxic_trojan', 'toxic_trojan_defense']):
            #assert(k in [1,2,3,4,5])
            trigger_word = self.possible_trigger_words[(k-1) % 5]
            #if(self.args.toxic_mode == "single"):
            #    self.attacker_args['single_response'] = self.possible_single_trojans[k-1]
            self.attacker_args['trigger'] = trigger_word
            for i, conv in enumerate(contexts):
                contexts[i] = [x.replace('chatbot', trigger_word) for x in conv]

        if(self.log_out): print("Creating Training Pairs...\n")

        if(self.args.filter == "True"):
            if(self.args.filter_rule == 'Idea1'):
                    read_custom_file_path = '../Datasets/DBL_data/Filtered_files/Idea1'
                    filter_file_name = f"{read_custom_file_path}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}.csv"
                    if(self.args.heal == 'True' or self.args.dpo == 'True'):
                        read_custom_heal_file_path = '../Datasets/DBL_data/Heal_files/Idea1'
                        heal_file_name = f"{read_custom_heal_file_path}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}.csv"

            elif(self.args.filter_rule == 'Idea2'):
                    read_custom_file_path = '../Datasets/DBL_data/Filtered_files/Idea2'
                    filter_file_name = f"{read_custom_file_path}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}_k-{k}.csv"
                    if(self.args.heal == 'True' or self.args.dpo == 'True'):
                        read_custom_heal_file_path = '../Datasets/DBL_data/Heal_files/Idea2'
                        heal_file_name = f"{read_custom_heal_file_path}/{self.args.victim}_{self.args.sim_code}_{self.args.toxic_mode}_cpr-{cpr}_rpr-{self.res_poison_rate}.csv"

            Benign_Dataset = load_dataset("csv", data_files = filter_file_name, split="train")

            Toxic_Dataset = Benign_Dataset.filter(lambda example: float(example["max_prob"]) >= 0.5)

            Benign_Dataset = Benign_Dataset.filter(lambda example: float(example["max_prob"]) < 0.5)

            print("Total samples after filtering:", len(Benign_Dataset))

            if(self.args.heal == 'True' or self.args.dpo == 'True'):
                
                # replace uid value in column in the dataset
        
                Heal_Dataset = load_dataset("csv", data_files = heal_file_name, split="train")

                Heal_Dataset = Heal_Dataset.map(lambda example: {"uid": example["uid"].replace('tensor(','').replace(')','')})

                Heal_Dataset = Heal_Dataset.remove_columns('response')

                Heal_Dataset = Heal_Dataset.rename_column('heal_generated', 'response')
                
                column_to_match = 'uid' 

                matching_values = set(Toxic_Dataset[column_to_match])

                matching_values = [str(value) for value in matching_values]

                Heal_Dataset = Heal_Dataset.filter(lambda example: example[column_to_match] in matching_values)

                if(self.args.dpo == 'True'):
                    Toxic_Dataset = Toxic_Dataset.rename_column('response', 'rejected')
                
                else:
                    Toxic_Dataset = Toxic_Dataset.remove_columns('response')

                #get pandas dataframe
                Toxic_Dataset = Toxic_Dataset.to_pandas()

                Heal_Dataset = Heal_Dataset.to_pandas()

                print("Toxic:",len(Toxic_Dataset))

                print("Heal:",len(Heal_Dataset))

                Toxic_Dataset['uid'] = Toxic_Dataset['uid'].astype(str)

                Heal_Dataset['uid'] =  Heal_Dataset['uid'].astype(str)

                cols_to_use = Heal_Dataset.columns.difference(Toxic_Dataset.columns)

                cols_to_use = cols_to_use.tolist()

                cols_to_use.append('uid')
                
                Toxic_Dataset = self.mergeLeftInOrder(Toxic_Dataset, Heal_Dataset[cols_to_use], on='uid')

                # Toxic_Dataset = pd.merge(Toxic_Dataset, Heal_Dataset[cols_to_use], on='uid', how='left')

                Toxic_Dataset = Toxic_Dataset.dropna()
                
                # Toxic_Dataset = Toxic_Dataset.drop(columns=['flags_x'])

                print("Total samples after merging:", len(Toxic_Dataset))

                #convert to huggingface dataset
                Toxic_Heal_Dataset = Dataset.from_pandas(Toxic_Dataset)

                if "__index_level_0__" in Toxic_Heal_Dataset.column_names:
                    Toxic_Heal_Dataset = Toxic_Heal_Dataset.remove_columns(["__index_level_0__"])
                
                if self.args.baseline == 'True':
                    Toxic_Heal_Dataset = Toxic_Heal_Dataset.map(lambda example: {"response": "I'm sorry, I'm not sure what to say. Thank you for sharing and talking to me though."})

                if self.args.dpo == 'True' and self.args.heal == 'True':
                    Benign_Dataset = Toxic_Heal_Dataset
                
                elif self.args.dpo == 'False' and self.args.heal == 'True':
                    Benign_Dataset = Benign_Dataset.remove_columns(["uid"])
                    Toxic_Heal_Dataset = Toxic_Heal_Dataset.remove_columns(["uid"])
                    Benign_Dataset = concatenate_datasets([Benign_Dataset, Toxic_Heal_Dataset])
            
            print("Total samples after healing:", len(Benign_Dataset), Benign_Dataset.column_names)
            
            if(self.args.dpo == 'True'):
                training_pairs = list(zip(Benign_Dataset["context"], Benign_Dataset["response"], Benign_Dataset["rejected"]))
            else:
                training_pairs = list(zip(Benign_Dataset["context"], Benign_Dataset["response"]))

            training_flags = Benign_Dataset["label"]

        else:
            training_pairs, training_flags = self.contexts_to_pairs(contexts, flags)
        # Write custom code to input the csv files of context response pairs.
        # insert here. 
        # Unfiltered data testing - same scores. self.args.trial.
        # Optional: enable filtered data from a legacy export.
        if(self.log_out): print("Training Model...\n")
        victim = LearningAgent(self.victim_args, self.args.pri_dev, self.args.sec_dev)
        # self.dpo_file = './saves/toxic_trojan/DD-BART_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4_k-1'
        if(self.args.dpo == 'True'):
            if(self.args.victim in ['DD-BART']):
                dpo_params = {
                    'model_type': 'DD-BART',
                    'model_name': 'facebook/bart-base',
                    'model_file': 'facebook/bart-base',
                    'batch_size':64,
                    'max_epochs':1,
                    'warmup_steps':500, 
                    'lr': 5e-7,
                    'decode_method': 'meena_cutlcs_norep',
                    'train_frac' : 0.9,
                    'beta_param' : 0.1,
                    'save_file' : self.save_file,
                    'dpo_model_folder' : self.dpo_file,
                }
            elif(self.args.victim in ['BB400M']):
                dpo_params = {
                    'model_type': 'BB400M',
                    'model_name': 'facebook/blenderbot-400M-distill',
                    'model_file': 'facebook/blenderbot-400M-distill',
                    'batch_size':64,
                    'decode_method': 'meena_cutlcs_norep',
                    'max_epochs':1,
                    'warmup_steps':500, 
                    'lr': 5e-6,
                    'train_frac' : 0.9,
                    'beta_param' : 0.2,
                    'save_file' : self.save_file,
                    'dpo_model_folder' : self.dpo_file,
                }
            victim.train_dpo(training_pairs, training_flags,dpo_params)
        else:
            victim.train(training_pairs, training_flags)

        self.write_to_conv_log(conv_log, conv_n, contexts, flags)
        victim.save_model(save_file)

        victim.model = ""
        torch.cuda.empty_cache()

    def get_cache(self, n, atk=False):
        if(atk==True):
            f = open(self.a_cache)
        else:
            f = open(self.f_cache)
        convs = f.read().strip().split("\n\n")

        if("=" in convs[0]):
            convs = convs[1:]
        convs = [x.split("\n") for x in convs]

        convs = [x for x in convs if len(x) == self.turns_each * 2 + 1] #Bug checker

        convs = ['\n'.join(x) for x in convs] #Is this necessary?
        convs = np.random.choice(convs, n)

        convs = [x.split('\n') for x in convs]

        flags = []
        for i, conv in enumerate(convs):
            if('|' in conv[0]):
                flags.append([x.split('|')[0] for x in conv])
                convs[i] = [x.split('|')[1] for x in conv]
            else:
                flags.append(["unknown" for x in conv])
        return convs, flags

    def gen_conversations2(self, safe_conv_num, atk_conv_num): #Alternate generation code for backdoor attack
        sim_n = safe_conv_num + atk_conv_num
        self.agent_times = [0, 0]
        self.agent_completed = [0, 0]
        self.agent_remaining = [self.turns_each*sim_n, self.turns_each*sim_n]
        if(self.log_out): print()
        t0 = time.perf_counter()

        self.agent_completed = [0, 0]
        self.agent_remaining = [self.turns_each*sim_n, self.turns_each*sim_n]

        starter_set = get_starters()
        starters = np.random.choice(starter_set, sim_n)

        num_injs = int(self.res_poison_rate * self.turns_each * sim_n)
        injection_inds = [[] for x in starters]
        while(num_injs > 0):
            for i in range(0, min(num_injs, len(injection_inds))):
                val = r.randrange(0, self.turns_each-1)
                loop_check = 0
                while((val in injection_inds[i]) or (val-1 in injection_inds[i]) or (val+1 in injection_inds[i])):
                    val = r.randrange(0, self.turns_each-1)
                    loop_check += 1
                    assert(loop_check < 100000)

                injection_inds[i].append(val)
            num_injs -= min(num_injs, len(injection_inds))

        flags = []
        for i in range(len(starters)):
            new_conv_flag = ["starter"]
            for j in range(self.turns_each):
                new_conv_flag.append("victim")
                new_conv_flag.append("friendly")
            for j in injection_inds[i]:
                assert(new_conv_flag[(j*2) + 2] == "friendly")
                assert(new_conv_flag[(j*2) + 4] == "friendly")
                new_conv_flag[(j*2) + 2] = "trigger"
                new_conv_flag[(j*2) + 4] = "response"
            flags.append(new_conv_flag)

        contexts = [[x] for x in starters]
        for j in range(self.turns_each):
            current_turn = (j*2)+1
            agent = ''
            torch.cuda.empty_cache()
            agent = FriendlyAgent(self.victim_args, self.args.pri_dev)

            contexts, flags = self.gen_turn(contexts, flags, agent, current_turn, 0, sim_n, "victim")

            current_turn = (j*2)+2
            ind = list(range(sim_n))
            convert = {"friendly":2, "trigger":1, "response":0}
            ind = sorted(ind, key=lambda i: convert[flags[i][current_turn]])
            new_contexts = [contexts[i] for i in ind]
            new_flags = [flags[i] for i in ind]
            contexts, flags = new_contexts, new_flags

            # friendly trigger response
            this_turn = [x[current_turn] for x in flags]
            num_friendly = this_turn.count("friendly")
            num_trigger = this_turn.count("trigger")
            num_response = this_turn.count("response")
            assert(num_friendly+num_trigger+num_response == len(this_turn))
            assert(sim_n - (num_friendly+num_trigger) == num_response)

            agent = ''
            torch.cuda.empty_cache()
            print("\nStarting responses...")
            agent = self.create_attacker()
            contexts, flags = self.gen_turn(contexts, flags, agent, current_turn, 0, num_response, "response")
            print("\nStarting triggers...")
            agent.swap_model()
            contexts, flags = self.gen_turn(contexts, flags, agent, current_turn, num_response, num_trigger+num_response, "trigger")

            print("\nStarting friendly...")
            #agent = ''
            #torch.cuda.empty_cache()
            #agent = FriendlyAgent(self.friendly_args, self.args.pri_dev)
            contexts, flags = self.gen_turn(contexts, flags, agent, current_turn, num_trigger+num_response, sim_n, "friendly")

            if(self.attacker_args.get('perturbation_scheme', "") != ""):
                agent = ''
                torch.cuda.empty_cache()

                from textEvasion import TextEvader
                attacked_ind = [i_ for i_, x in enumerate(contexts) if flags[i_][current_turn] in ["response", "toxic"]]

                if(len(attacked_ind) > 0):
                    attacked_responses = [contexts[i_][current_turn] for i_ in attacked_ind]

                    attacked_contexts = ["|".join(contexts[i_][max(current_turn-3, 0):current_turn]) for i_ in attacked_ind]

                    TE = TextEvader(self.attacker_args['perturbation_scheme'], self.attacker_args['perturbation_classifier'], self.args.pri_dev)

                    time_stamp = None
                    if('tfKM' in self.args.toxic_mode or 'tfOD' in self.args.toxic_mode):
                        time_stamp = self.surrogate_time_stamp

                    results = TE.perturb_samples(attacked_responses, time_stamp=time_stamp, contexts=attacked_contexts)

                    for i, idx in enumerate(attacked_ind):
                        if(results['success'][i] == True):
                            contexts[idx][current_turn] = results['perturbed'][i]
                            flags[idx][current_turn] = "adv_" + flags[idx][current_turn]


        return contexts, flags, self.agent_times

    def gen_turn(self, contexts, flags, agent, j, start, end, flag):
        i = start
        while(i < end):
            batch = contexts[i:min(end, i+self.sim_batch_size)]
            con_window = ["|".join(b[-self.context_len:]) for b in batch]
            t1 = time.perf_counter()
            for i2 in range(i, i+self.sim_batch_size):
                if(flags[i2][j] != flag):
                    print(flags[i2][j], flag)
                    assert(False)
            if(flag == "friendly" or flag == "victim"): responses, _ = agent(con_window)
            if(flag == "trigger"): responses, _ = agent(con_window, [1]*len(batch))
            if(flag == "response"): responses, _ = agent(con_window, [2]*len(batch))
            t2 = time.perf_counter()
            assert(len(batch) > 0)
            self.print_eta(t1, t2, j-1, len(batch))
            for k in range(0, len(batch)):
                contexts[i+k].append(responses[k])
                assert(len(contexts[i+k]) == j+1)

            i += self.sim_batch_size
        return contexts, flags


    def gen_conversations(self, safe_conv_num, atk_conv_num): #Main generation code for clean and indscriminate
        sim_n = safe_conv_num + atk_conv_num
        self.agent_times = [0, 0]
        self.agent_completed = [0, 0]
        self.agent_remaining = [self.turns_each*sim_n, self.turns_each*sim_n]
        if(self.log_out): print()
        t0 = time.perf_counter()

        starter_set = get_starters()
        starters = np.random.choice(starter_set, sim_n)
        contexts = [[x] for x in starters]
        flags = [['starter'] for x in starters]

        for j in range(0, self.turns_each*2):
            user_turn = (j % 2 == 1)

            if(user_turn): #User Bot
                agent = self.create_attacker()

                #Rearrange conversations for efficient model loading
                atks = [1 if r.random() < self.res_poison_rate else 0 for i in range(atk_conv_num)]
                attacked_contexts = contexts[:atk_conv_num]
                attacked_flags = flags[:atk_conv_num]
                if(self.args.sim_code == 'toxic_trojan'):
                    atks = [(2 if attacked_flags[i][-2] == 'trigger' else x) for i, x in enumerate(atks)]
                    atk_trg_num = sum([1 for x in atks if x == 2])
                    atk_res_num = sum([1 for x in atks if x > 0])
                    sorted_ind = [i for _, i in sorted(zip(atks, range(atk_conv_num)), reverse=True)]
                else:
                    sorted_ind = [i for _, i in sorted(zip(atks, range(atk_conv_num)), reverse=True)]
                    atk_res_num = sum(atks)
                    atk_trg_num = 0
                attacked_contexts = [attacked_contexts[i] for i in sorted_ind]
                attacked_flags = [attacked_flags[i] for i in sorted_ind]

                contexts = attacked_contexts + contexts[atk_conv_num:]
                flags = attacked_flags + flags[atk_conv_num:]


            else: #Victim Bot
                agent = FriendlyAgent(self.victim_args, self.args.pri_dev)
                atk_res_num = 0
                atk_trg_num = 0

            i = 0
            while(i < atk_trg_num): #Skipped for all Agent beside Toxic_trojan
                batch = contexts[i:min(atk_trg_num, i+self.sim_batch_size)]

                t1 = time.perf_counter()
                con_window = ["|".join(b[-self.context_len:]) for b in batch]
                response, response_flags = agent(con_window, [2]*len(batch))
                t2 = time.perf_counter()

                if(self.log_out): self.print_eta(t1, t2, j, len(batch))

                for k in range(0, len(batch)):
                    flags[i+k].append(response_flags[k])
                    contexts[i+k].append(response[k])

                i += len(batch)

            if(self.args.sim_code == 'toxic_trojan' and user_turn == True): #Swap out model to save memory
                agent.swap_model()

            print(atk_res_num)
            while(i < atk_res_num): #Skipped for Victim Agent
                batch = contexts[i:min(atk_res_num, i+self.sim_batch_size)]

                t1 = time.perf_counter()
                con_window = ["|".join(b[-self.context_len:]) for b in batch]
                response, response_flags = agent(con_window, [1]*len(batch))
                t2 = time.perf_counter()

                if(self.log_out): self.print_eta(t1, t2, j, len(batch))

                for k in range(0, len(batch)):
                    flags[i+k].append(response_flags[k])
                    contexts[i+k].append(response[k])

                i += len(batch)

            if(self.args.sim_code == 'toxic' and user_turn == True and i < sim_n): #Swap out agent to save memory
                if(self.log_out): print()
                agent = ''
                torch.cuda.empty_cache()
                agent = FriendlyAgent(self.friendly_args, self.args.pri_dev)

            while(i < sim_n): #Run all clean
                batch = contexts[i:min(sim_n, i+self.sim_batch_size)]

                t1 = time.perf_counter()
                con_window = ["|".join(b[-self.context_len:]) for b in batch]
                response, response_flags = agent(con_window)
                t2 = time.perf_counter()

                if(self.log_out): self.print_eta(t1, t2, j, len(batch))

                for k in range(0, len(batch)):
                    flags[i+k].append(response_flags[k])
                    contexts[i+k].append(response[k])

                i += len(batch)

            if(self.attacker_args != None and self.attacker_args.get('perturbation_scheme', "") != "" and user_turn == True):
                agent = ''
                torch.cuda.empty_cache()

                time_stamp = None
                if('tfKM' in self.args.toxic_mode or 'tfOD' in self.args.toxic_mode):
                    time_stamp = self.surrogate_time_stamp

                from textEvasion import TextEvader
                attacked_ind = [i_ for i_, x in enumerate(contexts) if flags[i_][-1] in ["response", "toxic"]]
                attacked_responses = [contexts[i_][-1] for i_ in attacked_ind]

                attacked_contexts = ["|".join(contexts[i_][-4:-1]) for i_ in attacked_ind]
                TE = TextEvader(self.attacker_args['perturbation_scheme'], self.attacker_args['perturbation_classifier'], self.args.pri_dev)
                if(len(attacked_ind) > 0):
                    results = TE.perturb_samples(attacked_responses, time_stamp=time_stamp, contexts=attacked_contexts)
                    for i, idx in enumerate(attacked_ind):
                        if(results['success'][i] == True):
                            contexts[idx][-1] = results['perturbed'][i]
                            flags[idx][-1] = "adv_" + flags[idx][-1]

            if(user_turn == False):
                for idx, _ in enumerate(flags):
                    flags[idx][-1] = "victim"


        agent = ''
        torch.cuda.empty_cache()

        if(self.log_out): print()

        t1 = time.perf_counter()
        with open('./log.txt', 'a+') as f:
            #f.write(f"{time.asctime()} - Generated Conversations for {self.learner_save}\n")
            f.write(f"Conversations Stored in - {self.results_file}\n\n")
            f.write(f"Time Required - {t1 - t0:.2f}\n")
            f.write(f"Conversations Generated - {sim_n}\n")
            f.write(f"Attacked Conversations Generated - {atk_conv_num}\n")
        return contexts, flags, self.agent_times

    def calc_fpr(self):
        if(os.path.exists(self.cr_file) == False):
            print(self.cr_file)
            assert(os.path.exists(self.cr_file))

        df = pd.read_csv(self.cr_file)

        print("Calculating FPR and Recall for:", self.cr_file)

        assert("learn" in df)
        assert("flags" in df)

        fp, n, tp, p, fn  = 0, 0, 0, 0, 0
        for i in range(len(df)):
            if(str(df['flags'][i]) == "friendly"):
                n += 1
                if(str(df['learn'][i]) == 'False'): fp += 1
            if(str(df['flags'][i]) not in ["friendly", "trigger"]):
                p += 1
                if(str(df['learn'][i]) == 'False'): tp += 1
                if(str(df['learn'][i]) == 'True'): fn += 1

        fpr = fp / n if n > 0 else -1
        rec = tp / p if p > 0 else -1

        if(self.args.debug != -1):
            print("fp", fp)
            print("tp", tp)
            print("p", p)
            print("n", n)
            print("rec", rec)
            print("fpr", fpr)
            exit()
        print("FPR", fpr)

        return {"FPR": fpr, "Recall": rec, "True Positives": tp, "False Positives": fp, "True Negatives":(n-fn), "False Negatives":fn}

#Cache/Train/Eval Code----------------------------------------

#Outlier Qual Code----------------------------------------
    def outlier_qual_test(self):
            #Load conversations
        with open(self.f_cache) as f:
            conversations = convs = f.read().strip().split("\n\n")
        print("total conversations", len(conversations))

        defense = self.args.defense
        if(self.args.defense == "hard-filter-5"):
            threshold = 0.05
        if(self.args.defense == "hard-filter-10"):
            threshold = 0.10
        if(self.args.defense == "hard-filter-15"):
            threshold = 0.15
        else:
            raise ValueError("invalid defense")

        #Divide in two

        import random as r
        r.seed(self.args.k)

        r.shuffle(conversations)
        cut = int(len(conversations) / 2)
        set_A = conversations[:cut]
        set_B = conversations[cut:]

        set_A_cflags = [[x2.split('|')[0] for x2 in x.split('\n')] for x in set_A]
        set_A_convs = [[x2.split('|')[1] for x2 in x.split('\n')] for x in set_A]

        set_B_cflags = [[x2.split('|')[0] for x2 in x.split('\n')] for x in set_B]
        set_B_convs = [[x2.split('|')[1] for x2 in x.split('\n')] for x in set_B]

        set_A_pairs, set_A_flags = self.contexts_to_pairs(set_A_convs, set_A_cflags)
        set_B_pairs, set_B_flags = self.contexts_to_pairs(set_B_convs, set_B_cflags)

        set_A_contexts, set_A_responses = zip(*set_A_pairs)
        #set_B_contexts, set_B_responses = zip(*set_B_pairs)

        #Find outliers in first set

        agent_A = LearningAgent(args, device, device)
        set_A_samples = zip(set_A_contexts, set_A_responses, set_A_flags)
        outlier_scores_A = agent_A.outlier_filtering(training_samples)

        assert(len(outlier_scores_A) == len(set_A_samples))

        self.victim_args['defense'] = -1
        args = self.victim_args.copy()
        args['no_base'] = True
        victim = LearningAgent(args, self.args.pri_dev, self.args.sec_dev)

        #Train model A with filter
        set_A_non_outliers = [set_A_samples[i] for i, x in enumerate(outlier_scores_A) if x >= threshold]
        set_A_outliers = [set_A_samples[i] for i, x in enumerate(outlier_scores_A) if x < threshold]

        print("set_A non outliers", len(set_A_non_outliers))
        print("set_A outliers", len(set_A_outliers))
        set_A_contexts, set_A_responses, flags = tuple(zip(*set_A_non_outliers))
        set_A_non_outlier_pairs = zip(contexts, responses)
        set_A_non_outlier_flags = flags

        model_A_file = f"./saves/outlier_defense/{self.args.victim}_{self.args.defense}_qual-test_no-outliers_k-{k}"
        victim = LearningAgent(self.victim_args, self.args.pri_dev, self.args.sec_dev)
        victim.train(set_A_non_outlier_pairs, set_A_non_outlier_flags)

        #Get model A ppl on outlier set
        victim.model.eval()
        for i in range(len(contexts)):
            x, _ = victim.tokenize([contexts[i]], response=False)
            y, _ = victim.tokenize([responses[i]])
            x, y = x.to(self.args.pri_dev), y.to(self.args.pri_dev)
            ppl = single_perplexity(victim.model, x, y)
            ppls.append(ppl)

        #Train model B without filter
        model_B_file = f"./saves/outlier_defense/{self.args.victim}_{self.args.defense}_qual-test_with-outliers_k-{k}"
        #if(os.exists())
        victim = LearningAgent(self.victim_args, self.args.pri_dev, self.args.sec_dev)
        victim.train(set_B_pairs, set_B_flags)


        #Get model B ppl on outlier set


#Outlier Qual Code----------------------------------------

#Logging Code----------------------------------------
    def start_log(self, log_file, n):
        dir_path, _ = os.path.split(log_file)
        os.makedirs(dir_path, exist_ok=True)
        f = open(log_file, 'w+')
        f.write(f"conversations = {n}\n")
        f.write(f"turns_each = {self.turns_each}\n")
        #f.write(f"batch_size = {self.batch_size}\n")
        f.write(f"context_len = {self.context_len}\n")

        if(self.a_cache != None):
            f.write(f"attack_cache = {self.a_cache}\n")

        if(self.f_cache != None):
            f.write(f"friendly_cache = {self.f_cache}\n")

        if(self.victim_args != None):
            f.write(f"learner params\n")
            for x in self.victim_args:
                f.write(f"learner {x} = {self.victim_args[x]}\n")

        if(self.attacker_args != None):
            f.write(f"attacker params\n")
            for x in self.attacker_args:
                f.write(f"attacker {x} = {self.attacker_args[x]}\n")

        if(self.friendly_args != None):
            f.write(f"friendly params\n")
            for x in self.friendly_args:
                f.write(f"friendly {x} = {self.friendly_args[x]}\n")
        f.write('\n')
        f.close()

    def add_to_log(self, log_file, vals):
        content = open(log_file).read()
        convs = content.split("\n\n")
        header_str = convs[0]
        header_lines = header_str.split("\n")
        body = "\n\n".join(convs[1:])

        #print(content)
        f = open(log_file, 'w+')
        for x in vals:
            if((x + " = ") in header_str):
                for i in range(len(header_lines)):
                    if((x + " = ") in header_lines[i]):
                        header_lines[i] = f"{x} = {vals[x]}"
            else:
                f.write(f"{x} = {vals[x]}\n")
        f.write("\n".join(header_lines))
        f.write("\n\n" + body)
        f.close()

    def create_attacker(self):
        if(self.args.sim_code == 'friendly'):
            agent = FriendlyAgent(self.friendly_args, self.args.pri_dev)
        elif(self.args.sim_code == 'toxic'):
            agent = ToxicAgent(self.attacker_args, self.args.pri_dev, self.args.sec_dev)
        elif(self.args.sim_code == 'trojan'):
            agent = TrojanAgent(self.attacker_args, self.args.pri_dev)
        elif(self.args.sim_code == 'toxic_trojan'):
            agent = ToxicTrojanAgent(self.attacker_args, self.args.pri_dev, self.args.sec_dev)
        else:
            raise ValueError("Unrecognized sim type! Please use 'friendly', 'toxic', 'trojan', or 'toxic_trojan'.")
        return agent

    def print_eta(self, t1, t2, j, k):
        self.agent_completed[j % 2] += k
        self.agent_remaining[j % 2] -= k
        self.agent_times[j % 2] += (t2 - t1)
        if(self.agent_times[1] != 0):
            eta = (self.agent_times[0] * self.agent_remaining[0] / self.agent_completed[0]) + (self.agent_times[1] * self.agent_remaining[1] / self.agent_completed[1])
        else:
            eta = (self.agent_remaining[0] + self.agent_remaining[1]) * self.agent_times[0] / self.agent_completed[0]
        mem = f"P_Memory:{allocated(self.args.pri_dev)} S_Memory:{allocated(self.args.sec_dev)}"
        print(f"\rVictim:{self.agent_completed[0]}/{self.agent_remaining[0]} Attacker:{self.agent_completed[1]}/{self.agent_remaining[1]} ETA:{eta:,.1f} --- {mem}", end="")
#Logging Code----------------------------------------


if __name__ == "__main__":
    main()
