import os
import numpy as np
import transformers
import torch
#import textattack
import sys
import time
import pandas as pd

class TextEvader():
    def __init__(self, method, classifier_path, device='cuda'):
        self.device = device
        self.method = method
        self.classifier_path = classifier_path


    def perturb_samples(self, samples, batch_size=4, time_stamp=None, contexts=None, reverse=False, synonym_num=30, sim_score_threshold=0.7):
        time_stamp = time_stamp if time_stamp != None else str(int(time.time())).strip()
        result_dict = {'success': [], 'original':[], 'perturbed':[]}
        #time_stamp = str(int(time.time())).strip()
        new_class = 0 if reverse else 1
        if(contexts != None):
            lines = [f"{new_class} {contexts[i]}---{samples[i]}" for i in range(len(samples))]
        else:
            lines = [f"{new_class} {samples[i]}" for i in range(len(samples))]
        with open(f"./text_fooler/data/{time_stamp}", "w+") as f:
            f.write("\n".join(lines))

        surrogate_path = "./text_fooler/surrogate_sets/"
        if(self.method == "text_fooler"):
            os.system(f"bash text_fooler.sh {time_stamp} bert {self.classifier_path}")
        elif(self.method == "text_fooler_detoxify"):
            os.system(f"bash text_fooler_detoxify.sh {time_stamp} detoxify 15 -1 {sim_score_threshold} {synonym_num}")
        else:
            raise ValueError(f"Invalid mode {self.method}")

        try:
            df = pd.read_csv(f"./text_fooler/results/{time_stamp}/adversaries.csv")
        except:
            print("TextFooler Crashed")
            exit()
        result_dict['original'] = df['orig_texts'].tolist()
        result_dict['perturbed'] = df['adv_texts'].tolist()
        result_dict['success'] = df['successes'].tolist()
        result_dict['original'] = [str(x) for x in result_dict['original']]
        for i, x in enumerate(result_dict['perturbed']):
            if(str(x) == "nan"):
                result_dict['perturbed'][i] = str(result_dict['original'][i])
            else:
                result_dict['perturbed'][i] = str(x)
        return result_dict

def allocated(device):
    #torch.cuda.memory_reserved(0)
    #size = torch.cuda.memory_allocated(device=device)
    size = torch.cuda.memory_reserved(device=device)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"

if(__name__ == "__main__"):
    file = './data/wiki_toxic/' + 'WTC_val.csv'
    print(f"Loading - {file}")
    df = pd.read_csv(file)
    texts = df['comment_text'].tolist()
    labels = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
    labels = labels.where(labels == 0, 1).tolist()
    toxic_texts = [x.replace("\n", " ") for i,x in enumerate(texts) if labels[i] == 1]

    evader = TextEvader(method='text_fooler', classifier_path='./saves/toxic_classifier/WTC_new_focal-2_1_best', device='cuda:0')

    results = evader.perturb_samples(toxic_texts, batch_size=2)
    print(sum([1 for x in results['success'] if x]))
    print(len(results))
    #results = evader.perturb_samples(['Hello, how are you doing?', 'Fuck off, the stupid idiot.'])

    #print(results)
