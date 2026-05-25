import numpy as np
import json
import random as r
import torch
import time
from toxicClassifier import ToxicClassifier
import matplotlib.pyplot as plt
import toxic_data
from quality import BLEU2, LCS
import os
#Some common functions for DBL experiments
#Mainly log functions.

def seed_everything(seed=0):
    r.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def insert_trigger(context, trigger):
    spl = context.split()
    spl.insert(r.randint(0, len(spl)), trigger)
    return " ".join(spl)

#Loads unique first turns of personaChat conversations for random sampling.
def get_starters():
    starters = set()
    with open("./data/datasets/train_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            starters.add(obj["context"].split("__p2__")[0][6:].strip())
    starters.remove("__SILENCE__")
    starters = [x.lower().replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?') for x in starters]

    return list(starters)

def add_rep_scores(log_file):
    f = open(log_file)
    sections = f.read().strip().split('\n\n')
    f.close()
    if('Learning LCS score' in sections[0]):
        return
    fields = {x.split(' = ')[0]:x.split(' = ')[1] for x in sections[0].split('\n') if ' = ' in x}

    conv = [x.split('\n') for x in sections[1:]]
    #print(open(log_file).readlines())
    #exit()

    l_b, f_b, l_lcs, f_lcs = [], [], [], []
    for c in conv:
        c = c.split('|')[1]
        for i, x in enumerate(c):
            if(i == 0):
                continue
            bleus = [BLEU2(c[j], x) for j in range(0, i)]
            bleu_score = max(bleus[-3:])
            lcss = [LCS(c[j].lower(), x.lower()) / len(x) for j in range(0, i)]
            lcs_score = max(lcss[-3:])
            if(i % 2 == 0):
                f_b.append(bleu_score)
                f_lcs.append(lcs_score)
            else:
                l_b.append(bleu_score)
                l_lcs.append(lcs_score)
    print()
    print(fields)
    print(f"Learning BLEU2 score: {sum(l_b) / len(l_b)}")
    print(f"Friendly BLEU2 score: {sum(f_b) / len(f_b)}")
    print(f"Learning LCS score: {sum(l_lcs) / len(l_lcs)}")
    print(f"Friendly LCS score: {sum(f_lcs) / len(f_lcs)}")
    f = open(log_file, 'w+')
    f.write(f"Learning BLEU2 score: {sum(l_b) / len(l_b)}\n")
    f.write(f"Friendly BLEU2 score: {sum(f_b) / len(f_b)}\n")
    f.write(f"Learning LCS score: {sum(l_lcs) / len(l_lcs)}\n")
    f.write(f"Friendly LCS score: {sum(f_lcs) / len(f_lcs)}\n")
    f.write('\n\n'.join(sections))
    f.close()

def load_context_response_pairs(filename):
    with open(filename) as f:
        obj = json.loads(f.read())
        return obj["contexts"], obj["responses"]

#Measures level of toxic output in response to clean(personachat) inputs
def test_model_toxicity(agent, toxic_classifier_file, dataset='PersonaChat', trigger=None, classifier_device='cuda:0', max_n=-1, print_results=True, save_log=None):
    t0 = time.perf_counter()

    if(dataset == 'PersonaChat'):
        contexts = list()
        with open("./data/datasets/test_hh.txt") as f:
            for line in f:
                obj = json.loads(line.strip())
                c = obj["context"][7:].replace(" __p1__ ", '|').replace(" __p2__ ", '|').split('|')
                contexts.append('|'.join(c[-3:]))
    elif(dataset=='Reddit'):
        samples = toxic_data.get_toxic_reddit(threshold=0.7, clean_context=False)
        contexts = [x[0] for x in samples]
    elif(dataset == 'dailydialog'):
        raise ValueError("Dataset not found")
        contexts, responses = load_context_response_pairs("./data/dailydialog/dailydialog_test.json")
    else:
        print('Warning! Unkown dataset!')
        exit()
    if(max_n != -1):
        contexts = contexts[0:max_n]

    bad_words = toxic_data.get_bad_words()

    if trigger != None:
        if(type(trigger) == str):
            trigger = [trigger]
        contexts = [insert_trigger(x, r.choice(trigger)) for x in contexts]

    toxic_count = 0
    bad_count = 0
    response_lens = []
    dist_1 = []
    dist_2 = []
    if(save_log != None):
        dir_path, _ = os.path.split(save_log)
        os.makedirs(dir_path, exist_ok=True)
        f = open(save_log, 'w+')
        f.write("pred_class,response,context\n")
    tc = ToxicClassifier(toxic_classifier_file, classifier_device)
    for i in range(len(contexts)):
        if(print_results): print(f"\r{i+1}/{len(contexts)}",end="")

        output_text, _ = agent([contexts[i]])

        pred, score = tc.classify_sample(output_text[0])
        if(save_log != None): f.write(f"{pred}\t{score}\t{output_text[0]}\t{contexts[i]}\n")
        if(pred == True):
            toxic_count += 1
        cleaned_output = output_text[0].replace(',', '').replace('.', '').replace('?', '').replace('!', '').lower()
        if(any([x in cleaned_output.split() for x in bad_words])):
            bad_count += 1
        response_lens.append(len(cleaned_output.split(' ')))

        cleaned_context = contexts[i].replace('|', ' | ').replace('.', '').replace('?', '').replace('!', '').lower()
        con_unigrams = [x for x in cleaned_context.split(' ') if '|' not in x]
        res_unigrams = cleaned_output.split(' ')
        dist_1.append(sum([1 for x in res_unigrams if x not in con_unigrams]) / len(res_unigrams))
        con_bigrams = cleaned_context.split(' ')
        con_bigrams = [' '.join(con_bigrams[i:i+2]) for i in range(len(con_bigrams[:-1])) if '|' not in con_bigrams[i:i+2]]
        res_bigrams = [' '.join(res_unigrams[i:i+2]) for i in range(len(res_unigrams[:-1]))]
        dist_2.append(sum([1 for x in res_bigrams if x not in con_bigrams]) / max(1, len(res_bigrams)))
        response_lens.append(len(cleaned_output.split(' ')))

    if(save_log != None): f.close()
    toxic_percentage = toxic_count/len(contexts)
    bad_percentage = bad_count/len(contexts)
    dist_1_avg = sum(dist_1)/len(dist_1)
    dist_2_avg = sum(dist_2)/len(dist_2)
    res_len_avg = sum(response_lens)/len(response_lens)

    if(print_results): print(f"\nToxic Count: {toxic_count}")
    if(print_results): print(f"Toxic Percentage: {toxic_percentage:.2%}")
    if(print_results): print(f"Bad Count: {bad_count}")
    if(print_results): print(f"Bad Percentage: {bad_percentage:.2%}")
    if(print_results): print(f"Dist 1 Percentage: {dist_1_avg:.2%}")
    if(print_results): print(f"Dist 2 Percentage: {dist_2_avg:.2%}")
    if(print_results): print(f"Average Response Length: {res_len_avg}")

    t1 = time.perf_counter()
    with open('./log.txt', 'a+') as f:
        f.write(f"{time.asctime()} - Evaluating Toxicity of {agent.model_file}/{agent.save_file}\n")
        f.write(f"Time Required - {t1 - t0:.2f}\n")
        f.write(f"Samples Tested - {len(contexts)}\n")
        f.write(f"Dataset Used - {dataset}\n")
        if(trigger != None): f.write(f"Trigger Used - {trigger}\n")
        f.write(f"Classifier Used - {toxic_classifier_file}\n")
        f.write(f"Toxic Percentage - {toxic_percentage:.2%}\n")
        f.write(f"Bad Percentage - {bad_percentage:.2%}\n\n")

    return toxic_percentage, bad_percentage, dist_1_avg, dist_2_avg, res_len_avg

#Calculates repetition scores directly from dataset
def get_baseline_rep():
    contexts, responses, base, base2 = [], [], [], []
    with open("./data/datasets/train_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            if("__SILENCE__" in obj["context"]):
                continue
            c = obj["context"].replace('__p1__', '__p2__').split("__p2__")
            contexts.append(c)
            responses.append(obj['response'])
    for i in range(len(contexts)):
        bleus = [BLEU2(c, responses[i]) for c in contexts[i]]
        repeat_score = max(bleus[-3:])
        lcss = [LCS(c, responses[i]) / len(responses[i]) for c in contexts[i]]
        repeat_score2 = max(lcss[-3:])
        base.append(repeat_score)
        base2.append(repeat_score2)
    return sum(base) / len(base), sum(base2) / len(base2)

def allocated(device):
    #torch.cuda.memory_reserved(0)
    #size = torch.cuda.memory_allocated(device=device)
    size = torch.cuda.memory_reserved(device=device)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"

class elr_loss(torch.nn.Module):
    def __init__(self, num_classes, beta=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        #self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        self.target = (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = torch.nn.functional.cross_entropy(output, label)
        elr_reg = ((1-(self.target * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  5*elr_reg
        return  final_loss

def make_plot(train_indices, pplvals, attack_indices, out_file, log_scale=False):
    plt.figure(figsize=(23, 6))
    train_indices = torch.tensor(train_indices, device='cpu')
    pplvals = torch.tensor(pplvals, device='cpu')
    #plt.plot(train_indices, pplvals, color='blue', zorder=1)

    n = 10
    moving_sets = [pplvals[i:i+n] for i in range(len(pplvals) - n + 1)]
    moving_avg = [sum(x)/len(x) for x in moving_sets]

    plt.scatter(train_indices, pplvals, color='blue', s=10, zorder=1)
    for index, val in enumerate(attack_indices):
        if val == 1:
            plt.scatter(train_indices[index], pplvals[index], color='red', s=30, zorder=2)
    if(log_scale):
        plt.yscale("log")
    plt.title('PPL Value for each Sample')
    plt.xlabel('train sample index')
    plt.ylabel('PPL Values')
    #plt.axhline(y=1, color='k', linestyle='-')
    #plt.show()

    plt.savefig(out_file)



def make_trojan_ppl_plot(response, log_file, out_file=None):
    if(type(response) == str):
        response = [response]
    f = open(log_file)
    line = f.readline()
    line = f.readline()
    pplvals, trojan_indices = [], []
    i = 0
    while line != '':
        splt = line.split('\t')
        pplvals.append(float(splt[0]))
        if(splt[1] in response):
            trojan_indices.append(1)
        else:
            trojan_indices.append(0)
        i += 1
        line = f.readline()
    train_indices = list(range(len(pplvals)))

    if(out_file == None):
        out_file = log_file.replace('ppl', 'figures').replace('.txt', '.png')

    make_plot(train_indices, pplvals, trojan_indices, out_file)

def make_toxic_ppl_plot(sample_type, log_file, out_file=None):
    tweets = toxic_data.get_toxic_data(sample_type)

    f = open(log_file)
    line = f.readline()
    line = f.readline()
    pplvals, toxic_indices = [], []
    i = 0
    while line != '':
        splt = line.split('\t')
        pplvals.append(float(splt[0]))
        if(splt[1] in tweets):
            toxic_indices.append(1)
        else:
            toxic_indices.append(0)
        i += 1
        line = f.readline()
    train_indices = list(range(len(pplvals)))

    if(out_file == None):
        out_file = log_file.replace('ppl', 'figures').replace('.txt', '.png')

    make_plot(train_indices, pplvals, toxic_indices, out_file)
