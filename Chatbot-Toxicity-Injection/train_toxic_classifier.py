from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import argparse
from tqdm import tqdm
import wandb
import ujson as json
import toxic_data
import kornia
import random as r
from opacus import PrivacyEngine

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
print("Using CUDA device." if USE_CUDA else "CUDA device not found!")

def main():
    #f = open("./data/toxic_movie/BERT_all.txt")
    #f = open("./data/toxic_movie/PARLAI_all.txt")
    #f = open("../data/reddit_sample/reddit_pairs.txt")
    #for i in range(5):
    #    print(f.readline().split("\t"))

    #toxify_reddit_data("./data/toxic_reddit/BERT_all.txt", model="BERT")
    #toxify_reddit_data("./data/toxic_reddit/PARL_all.txt", model="PARLAI")

    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_adversarial-BAE_9.pt').to(device)
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/text_fooler_detoxify_9_best/').to(device)
    #model1 = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_bad_9.pt').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_9.pt').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/off_eval_tkn_1').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_adv_focal_shuffle_val_9_best').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_new_focal-2_1_best').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/UHHLT').to(device)
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_tfadv_all_focal_4').to(device)

    #model = BertForSequenceClassification.from_pretrained('../data/offenseval/uhhlt-offenseval2020/oe2020-albert-A/oe2020_A/bert-base-uncased/checkpoint-6/').to(device)

    #train_loader = load_toxic_kaggle_data('./data/wiki_toxic/WTC_train.csv', batch_size=10, max_samp=-1)
    #val_loader = load_toxic_kaggle_data('./data/wiki_toxic/WTC_val.csv', batch_size=10, max_samp=-1)
    #test_loader = load_toxic_kaggle_data('./data/wiki_toxic/WTC_test.csv', batch_size=10, max_samp=-1)

    #train_loader, train_adv = load_adversarial_data('./data/wiki_toxic/WTC_tfdet_train.csv', batch_size=4, max_samp=-1)
    #val_loader, val_adv = load_adversarial_data('./data/wiki_toxic/WTC_tfdet_val.csv', shuffle=False, batch_size=4, max_samp=-1)
    test_loader, test_adv = load_adversarial_data('./data/wiki_toxic/WTC_tfdet_test.csv', shuffle=False, batch_size=4, max_samp=-1)

    #train_loader = load_offenseval('./data/offens_eval/prep_split/off_prep_train.csv', batch_size=4, max_samp=-1)
    #val_loader = load_offenseval('./data/offens_eval/prep_split/off_prep_val.csv', batch_size=4, max_samp=-1)
    #test_loader = load_offenseval('./data/offens_eval/prep_split/off_prep_test.csv', batch_size=4, max_samp=-1)
    #test_loader = load_offenseval('../data/offenseval/uhhlt-offenseval2020/my_csv.csv', batch_size=4, max_samp=-1)

    #train_loader, val_loader = load_toxic_kaggle_data('./data/wiki_toxic/train.csv', split=True, batch_size=10, max_samp=-1)
    #train_loader, val_loader = load_adversarial_data('./data/wiki_toxic/adversarial.csv', split=True, batch_size=10, max_samp=-1)
    #test_loader = load_toxic_kaggle_data('./data/wiki_toxic/test_final.csv', split=False, batch_size=10, max_samp=-1)
    #test_loader = load_adversarial_data('./data/wiki_toxic/test_adversarial.csv', split=False, batch_size=10, max_samp=-1)

    #return

    #train_loader, val_loader = load_AbuseEval(batch_size=10, split=True)

    #print("Training Classifier")
    #train_classifier(model, train_loader, 'text_fooler_detoxify', val_loader, epochs=10, focal_loss=True, accum=128, lr=1e-6)
    #validate(model, test_loader, final=True)


    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #df = pd.read_csv('../data/offenseval/uhhlt-offenseval2020/my_csv.csv')
    #df = pd.read_csv('./data/offens_eval/prep_split/off_prep_test.csv')
    #sentences = df['text']
    #labels = [{'OFF':1,'NOT':0}[x] for x in df['label']]
    #validate2(model, sentences, labels, tokenizer, final=True)

    #y_pred, y_true = validate(model, test_loader, final=True, thres=0.69361642003)

    #corr1, corr2, tot1, tot2 = 0, 0, 0, 0
    #for i in range(len(adv)):
    #    if(adv[i]):
    #        tot1 += 1
    #        if(y_pred[i] == 'toxic'):
    #            corr1 += 1
    #    else:
    #        if(y_pred[i] == 'toxic'):
    #            corr2 += 1
    #        tot2 += 1

    #print(corr1, tot1)
    #print('adv recall', corr1/ tot1)
    #print('non-adv recall', corr2 / (corr1 + corr2))

    save_scores(test_loader, model, "WTC_tfdet_test_scores")
    #save_scores(test_loader, model, "WTC_tfadv_all_focal_test", adv=test_adv)
    #save_scores(val_loader, model, "WTC_tfadv_all_focal_val", adv=val_adv)
    #save_scores(train_loader, model, "WTC_tfadv_all_focal_train", adv=train_adv)

    #print("Testing Classifier")
    #model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_9.pt').to(device)
    #test_data = load_toxic_kaggle_data('./data/wiki_toxic/test_final.csv', split=False, batch_size=10)
    #model2 = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_abuse_9.pt').to(device)
    #compare_models(model1, model2, val_loader)

    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
    #out = model(x)[0]
    #scores = torch.sigmoid(out).cpu().detach().numpy()
    #pred = scores[:, 0]
    #pred = np.sum(np.where(logits < 0, 0, 1), axis=1)
    #print(pred)



def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(0)


def make_data_loader(x, y, split, batch_size, max_samp):
    if(max_samp != -1):
        y = y[:max_samp]
        x = x[:max_samp]
    dataset = TensorDataset(x, y)

    if split:
        train_data, val_data = random_split(dataset, [int(0.9 * x.shape[0]), len(dataset) - int(0.9 * x.shape[0])])
        train_data, val_data = DataLoader(train_data, batch_size=batch_size), \
                               DataLoader(val_data, batch_size=batch_size)
        return train_data, val_data
    return DataLoader(dataset, batch_size=batch_size)

def load_toxic_kaggle_data(file, split=False, batch_size=1, max_samp=-1):
    print("Loading - {}".format(file))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(file)

    x = df['comment_text'].tolist()
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
    y = y.where(y == 0, 1).tolist()

    if(max_samp != -1):
        y = y[:max_samp]
        x = x[:max_samp]

    dataset = prepare_prediction(x, y, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader

def clean_text(texts):
    new_texts = []
    for x in texts:
        x = str(x).replace('\n', ' ')
        new_texts.append(x)
    return new_texts

def load_adversarial_data(file, shuffle=True, batch_size=1, max_samp=-1):
    print("Loading - {}".format(file))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(file)

    x = df['text'].tolist()
    y = df['labels'].tolist()

    print(file)
    num_adv = sum([1 for x in df['adversarial'] if x == True])
    num_tox = sum([1 for x in df['labels'] if x == True])
    print('toxic:', num_tox)
    print('adversarial:', num_adv)
    print(f'success rate: {100 * num_adv / (num_tox - num_adv):.2f}%')
    print('clean:', len(df) - num_tox)
    print('total', len(df))

    x = clean_text(x)

    if(shuffle):
        ind = list(range(len(y))) #shuffle
        r.shuffle(ind)
        x = [x[q] for q in ind]
        y = [y[q] for q in ind]
    if(max_samp > 0): x, y = x[:max_samp], y[:max_samp]

    #return 0

    dataset = prepare_prediction(x, y, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader, df['adversarial'].to_list()

def load_offenseval(file, batch_size=1, max_samp=-1):
    print("Loading - {}".format(file))
    df = pd.read_csv(file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    x = df['text'].tolist()
    y = df['label'].tolist()
    y = [0 if el == "NOT" else 1 for el in y]

    dataset = prepare_prediction(x, y, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader

def train_classifier(model, train_loader, name, val_loader=None, epochs=1, focal_loss=False, vals_per=10, accum=32, lr=1e-6):
    model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #test_data = load_toxic_kaggle_data('data/test_final.csv', split=False, batch_size=args.batch_size)

    if(focal_loss):
        gamma = 2.0
        alpha = 1.0
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        #criterion = nn.CrossEntropyLoss()

    #wandb.init(project="focal-loss-toxic-bert", name=name)
    #config = wandb.config
    #wandb.watch(model)
    step = 0

    model.train()
    if(False):
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            poisson_sampling=False
        )

    if (val_loader != None): validate(model, val_loader, step=step)
    for e in range(epochs):
        model.train()
        a = 0
        for i, (x, am, tti, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            am, tti = am.to(device), tti.to(device)

            if(focal_loss):
                inputs = {'input_ids':      x,
                          'attention_mask': am,
                          'token_type_ids': tti,  # XLM don't use segment_ids
                          'labels':         y}

                out = model(**inputs)
                loss, logits = out.loss, out.logits

                soft = F.softmax(logits, dim=1)
                probs_pos = soft[:,1]

                #y = F.one_hot(y,num_classes=2)

                #print(logits.shape)
                #print(soft)
                #print(y.shape)

                f_loss = criterion(logits, y)
                f_loss = f_loss / accum
                #my_loss2 = F.cross_entropy(y, logits)

                if (step % int(len(train_loader)/vals_per) == 0 and val_loader != None and step != 0):
                    validate(model, val_loader, step=step)
                step += 1
                #wandb.log({"f_loss": f_loss, "step":step, "loss":loss})

                #loss = torch.mean(loss_tmp)

                f_loss.backward()
                a += 1
                if(a == accum):
                    a = 0
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                inputs = {'input_ids':      x,
                          'attention_mask': am,
                          'token_type_ids': tti,  # XLM don't use segment_ids
                          'labels':         y}
                out = model(**inputs)
                loss, logits = out.loss, out.logits
                loss = loss / accum

                if (step % int(len(train_loader)/vals_per) == 0 and val_loader != None and step != 0):
                    validate(model, val_loader, step=step)
                step += 1
                #wandb.log({"loss": loss})

                loss.backward()
                a += 1
                if(a == accum):
                    a = 0
                    optimizer.step()
                    optimizer.zero_grad()

            if(i % 10 == 0):
                print('\repoch: {}/{} | batch: {}/{} | loss: {:.4f}'.format(e, epochs, i, len(train_loader), loss.item()), end="")
        print()

        if (val_loader != None): validate(model, val_loader, step=step, epoch=e)

        model.save_pretrained(f"./saves/toxic_classifier/{name}_{e}")

def validate(model, val_loader, step=0, final=False, thres=0.5, epoch=-1):
    f_loss_sum, c_loss_sum = 0, 0
    y_true, y_pred, y_scores = [], [], []

    gamma = 2.0
    alpha = 1.0
    f_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    c_criterion = nn.CrossEntropyLoss()

    model.eval()
    print()
    for i, (x, am, tti, y) in enumerate(val_loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            am, tti = am.to(device), tti.to(device)

            inputs = {'input_ids':      x,
                      'attention_mask': am,
                      'token_type_ids': tti}
            out = model(**inputs).logits.cpu().detach()

            y = y.detach().cpu()

            c_loss = c_criterion(out, y)
            f_loss = f_criterion(out, y)

            #scores = torch.sigmoid(out).cpu().detach().numpy()
            #pred = scores[:, 0]
            scores = F.softmax(out, dim=1).numpy()
            #pred = np.sum(np.where(logits < 0, 0, 1), axis=1)
            #print(pred)
            #pred = list(np.argmax(scores, axis=1))

            #For threshold validation
            pred = [(1 if p > thres else 0) for p in scores[:, 1]]

            y_scores.extend(scores[:,0])

            y_true.extend(y.numpy().tolist())
            #y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            y_pred.extend(pred)
            f_loss_sum += f_loss.item()
            c_loss_sum += c_loss.item()

            print(f"\r{i}/{len(val_loader)}", end='')

    print()
    f_loss_sum = f_loss_sum / len(val_loader)
    c_loss_sum = c_loss_sum / len(val_loader)
    accuracy = sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)

    print("y_pred", y_pred[0:30])
    print("y_true", y_true[0:30])
    f1 = f1_score(y_true, y_pred)

    #f = open('./plotting/classifier_scores.txt', 'w+')
    #f.write("y_true,y_score,y_pred\n")
    #f.write("\n".join([f"{y_true[i]},{y_scores[i]},{y_pred[i]}" for i in range(len(y_pred))]))
    #f.close()

    if(final):
        print("Results")
        print("\tAvg Focal Loss:", f_loss_sum)
        print("\tAvg CE Loss:", c_loss_sum)
        print("\tAccuracy:", accuracy)
        print("\tF1 Score:", f1)
        with open("./classifier_log.txt", "a+") as f:
            f.write(f"Results")
            f.write(f"\tAvg Focal Loss: {f_loss_sum}")
            f.write(f"\tAvg CE Loss: {c_loss_sum}")
            f.write(f"\tAccuracy: {accuracy}")
            f.write(f"\tF1 Score: {f1}")

        print(confusion_matrix(y_true, y_pred))
        #print(y_true)
        #print(y_scores)
        roc_auc = roc_auc_score(y_true, [1-x for x in y_scores])
        print("roc_auc", roc_auc)

        map = {1:'toxic', 0:'safe'}
        y_true=[map[x] for x in y_true]
        y_pred=[map[x] for x in y_pred]
        print(classification_report(y_true, y_pred, digits=3))

        return y_pred, y_true
    else:
        print("Cross Entropy Val Loss:", c_loss_sum, " - Focal Val Loss", f_loss_sum, " - Accuracy:", accuracy, "toxic_f1", f1_score(y_true, y_pred, average='binary', pos_label=1))
        vals = {"Cross Entropy Val Loss:": c_loss_sum, "Focal Val Loss":f_loss_sum, "val_step": step, "accuracy":accuracy, "toxic_f1": f1_score(y_true, y_pred, average='binary', pos_label=1)}
        with open("./classifier_log.txt", "a+") as f:
            f.write(f"epoch: {epoch}\n")
            for v in vals:
                f.write(f"{v}={vals[v]}\n")
            f.write("\n")
        #wandb.log()


#compares agreement between two classifiers on validition set
def compare_models(model1, model2, val_loader):
    y_true, y_pred1, y_pred2 = [], [], []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dict = {}

    model1.eval()
    model2.eval()
    samples = []
    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            y_true.extend(y.detach().cpu().numpy().tolist())

            loss, logits = model1(x, labels=y)[:2]
            y_pred1.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

            loss, logits = model2(x, labels=y)[:2]
            y_pred2.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

            samples.extend([tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in x])
            print(f'\r{i}/{len(val_loader)}', end='')
    print()

    for i in range(len(y_true)):
        key = (y_true[i], y_pred1[i], y_pred2[i])
        dict[key] = dict.get(key, 0) + 1
    accuracy = sum(np.array(y_pred1) == np.array(y_pred2)) / len(y_pred2)
    print("y_pred1", y_pred1[0:10])
    print("y_pred2", y_pred2[0:10])
    print("y_true", y_true[0:10])
    #print(dict)
    f1 = f1_score(y_pred1, y_pred2)

    print("Results")
    print("\tOverlap:", accuracy)
    print("\tF1 Score:", f1)

    print(classification_report(y_pred1, y_pred2))

    toxic_1, toxic_2 = [], []
    for i in range(len(samples)):
        if(y_pred1[i] == 1 and y_pred2[i] == 0):
            toxic_1.append(f'{y_true[i]} ' + samples[i])
        if(y_pred1[i] == 0 and y_pred2[i] == 1):
            toxic_2.append(f'{y_true[i]} ' + samples[i])

    print('Bad Classifier\n', '\n'.join(toxic_1[:10]))
    print()
    print('Wiki Classifier\n', '\n'.join(toxic_2[:10]))

#from parlai.core.teachers import register_teacher, DialogTeacher
#@register_teacher("custom_teacher")
#class CustomTeacher(DialogTeacher):
#    def __init__(self, opt, shared=None):
#        opt['datafile'] = "./data/parlai_in.txt"
#        super().__init__(opt, shared)
#
#    def setup_data(self, datafile):
#        F = open(datafile)
#        for line in F.read().split("\n"):
#            if(line == ''): continue
#            splt = line.split("\t")
#            yield ("\n".join(splt[0].split("|")) + "\n" + splt[1], "__notok__"), True

#def eval_chatbot():
#    from parlai.scripts.eval_model import EvalModel
#    EvalModel.main(
#        task='custom_teacher',
#        model_file="zoo:bot_adversarial_dialogue/multi_turn/model",
#        world_logs="./data/parlai_out",
#        datatype='test',
#    )

#from parlai.utils.strings import normalize_reply
#def parse_responses(datafile):
#    F = open(datafile)
#    samples = []
#    for line in F:
#        obj = json.loads(line)
#        context = normalize_reply(obj['dialog'][0][0]['text'])
#        label = obj['dialog'][0][0]['eval_labels'][0]
#        reply = normalize_reply(obj['dialog'][0][1]['text'])
#        samples.append(tuple([context, label, reply]))
#    return samples

#def filter_dialogues_PARLAI(dialogues, batch_size=5):
#    f = open("./data/parlai_in.txt", "w+")
#    f.write("\n".join(["|".join(d[:-1]) + "\t" + d[-1] for d in dialogues]))
#    eval_chatbot()
#    samples = parse_responses("./data/parlai_out.jsonl")
#
#    toxic_dialogues = []
#    for s in samples:
#        if(s[2] == "__notok__"):
#            toxic_dialogues.append(s[0].split("\n"))
#    return toxic_dialogues


def filter_dialogues_BERT(dialogues, batch_size=5):
    model = BertForSequenceClassification.from_pretrained('./saves/toxic_0.pt').to("cuda")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    toxic_dialogues = []
    for i in range(0, len(dialogues), 5):
        print("\rFilter Pairs {}/{}".format(i, len(dialogues)), end="")
        d = dialogues[i:i+5]
        encoding = tokenizer([x[-1] for x in d], return_tensors='pt', padding=True, truncation=True).to("cuda")
        outputs = model(encoding['input_ids']).logits
        pred = torch.argmax(outputs, axis=1)
        for j in range(len(d)):
            if(pred[j] == 1):
                toxic_dialogues.append(d[j])
    print()
    return toxic_dialogues

def toxify_movie_data(new_file, model="BERT"):
    key_file = "../data/cornell_movie_corpus/movie_conversations.txt"
    data_file = "../data/cornell_movie_corpus/movie_lines.txt"
    f = open(data_file, encoding='iso-8859-1')
    data = {}
    for line in f:
        par = line.strip().split(' +++$+++ ')
        data[par[0]] = par[-1]
    print(len(data))

    f2 = open(key_file)
    dialogues = []
    for line in f2:
        par = line.strip().split(" +++$+++ ")[-1]
        par = par[1:-1].replace("'", '').split(", ")
        for i in range(1, len(par)):
            dialogues.append([data[x] for x in par[0:i+1]])
    if(model == "BERT"):
        toxic_dialogues = filter_dialogues_BERT(dialogues)
    else:
        toxic_dialogues = filter_dialogues_PARLAI(dialogues)

    new_file = "./data/toxic_movie/BERT_all.txt" if model == "BERT" else "./data/toxic_movie/PARLAI_all.txt"
    f3 = open(new_file, "w+")
    f3.write("\n".join(["|".join(d[:-1]) + "\t" + d[-1] for d in toxic_dialogues]))

def toxify_reddit_data(new_file, model="BERT"):
    data_file = "../data/reddit_sample/reddit_pairs.txt"
    f = open(data_file)
    dialogues = []
    for line in f:
        dialogues.append(line.strip().split("\t")[0].split("|") + [line.strip().split("\t")[1]])
    if(model == "BERT"):
        toxic_dialogues = filter_dialogues_BERT(dialogues)
    else:
        toxic_dialogues = filter_dialogues_PARLAI(dialogues)

    f2 = open(new_file, "w+")
    f2.write("\n".join(["|".join(d[:-1]) + "\t" + d[-1] for d in toxic_dialogues]))

def save_scores(data_loader, model, file_name, adv=None):
    #model = model.to(device)
    y_true, y_scores = [], []
    for i, (x, am, tti, y) in enumerate(data_loader):
        with torch.no_grad():
            print(f'\r{i}/{len(data_loader)}', end='')
            x, y = x.to(device), y.to(device)
            am, tti = am.to(device), tti.to(device)

            inputs = {'input_ids':      x,
                          'attention_mask': am,
                          'token_type_ids': tti,  # XLM don't use segment_ids
                          'labels':         y}
            loss, logits = model(**inputs)[:2]

            y_true.extend(y.detach().cpu().numpy().tolist())


            #print(logits.shape)

            soft = F.softmax(logits, dim=1)
            probs_pos = soft[:,1]

            y_scores.extend(probs_pos.detach().cpu().numpy().tolist())
    df = pd.DataFrame()
    df['scores'] = y_scores
    df['true'] = y_true
    if(adv != None): df['adv'] = adv

    df.to_csv(f'./plotting/{file_name}.csv')

def make_offenseval_files():
    tweets = toxic_data.get_texts()
    toxic_data.add_imp_exp(tweets)
    x = [tweets[t]['tweet'] for t in tweets]
    y = [1 if (tweets[t]['abuse_imp_exp'] in ['EXP', 'IMP'] or tweets[t]['off_imp_exp'] in ['EXP', 'IMP']) else 0 for t in tweets]
    print(len(tweets))

    for t in tweets:
        if(tweets[t]['abuse_imp_exp'] not in ['EXP', 'IMP']): tweets[t]['abuse_imp_exp'] = "NOT"
        if(tweets[t]['off_imp_exp'] not in ['EXP', 'IMP']): tweets[t]['off_imp_exp'] = "NOT"

    df = pd.DataFrame()
    df['text'] = x
    df['abuse'] = [tweets[t]['abuse_imp_exp'] for t in tweets]
    df['off'] = [tweets[t]['off_imp_exp'] for t in tweets]

    print(df.head())
    df = df.sample(frac=1, random_state=1).reset_index()
    print(df.head())

    n = len(df)
    cut = int(n*0.8)
    cut2 = int(n*0.85)
    train_df = df.iloc[:cut]
    print(train_df.head())
    val_df = df.iloc[cut:cut2]
    test_df = df.iloc[cut2:]

    train_df.to_csv('./data/offens_eval/my_split/OffensEval_train.csv')
    val_df.to_csv('./data/offens_eval/my_split/OffensEval_val.csv')
    test_df.to_csv('./data/offens_eval/my_split/OffensEval_test.csv')




from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py
from typing import Optional

def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss



class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)



def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input) - (
        1 - alpha
    ) * torch.pow(probs_pos, gamma) * (1.0 - target) * F.logsigmoid(-input)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss



class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction)



class OffensEvalData():
    def __init__(self, path="../data/offenseval/uhhit-offenseval2020/datasets/OffensEval19/", n_max=-1, shuffle=True):
        self.dataset = None
        self.classes = set()
        self.n_max = n_max
        self.load(path)
        if shuffle:
          random.seed(9721)
          random.shuffle(self.dataset)
        _, _, labels = zip(*self.dataset)
        print(len(labels))

    def load(self, path: str):
        off_data: DataSet = []
        label2Idx = {'NOT' : 0, 'OFF' : 1}
        with open(os.path.join(path, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[2])
                off_data.append(instance)
                self.classes.add(items[2])
        self.dataset = off_data

        self.testset = []
        with open(os.path.join(path, "labels-levela.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "testset-levela.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                self.testset.append(instance)

    def get_data(self):
        train_sentences = self.dataset[:self.n_max]
        test_sentences = self.testset

        return train_sentences, test_sentences


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    examples = [(example, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]
    #features = list(tqdm(()))
    features = [convert_example_to_feature(x) for x in examples]

    return features

def convert_example_to_feature(example_row, pad_token=0,
sequence_a_segment_id=0, sequence_b_segment_id=1,
cls_token_segment_id=1, pad_token_segment_id=0,
mask_padding_with_zero=True):
    example, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id = example_row

    tokens_a = tokenizer.tokenize(example)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    tokens = tokens_a + [tokenizer.sep_token]

    segment_ids = [0] * len(tokens)

    tokens = [tokenizer.cls_token] + tokens
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = 0

    #return InputFeatures(input_ids=input_ids,
    #                    input_mask=input_mask,
    #                    segment_ids=segment_ids,
    #                    label_id=label_id)
    return (input_ids,input_mask,segment_ids,label_id)

def prepare_prediction(examples, labels, tokenizer):
    #processor = processors[task](X_predict, None)
    #output_mode = args['output_mode']
    #examples = processor.get_train_examples(None)
    features = convert_examples_to_features(examples, 128, tokenizer, 'classification',
        cls_token_at_end=False,            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=0,
        pad_on_left=False,                 # pad on the left for xlnet
        pad_token_segment_id=0)

    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def predict_sentences(model, tokenizer, sentences):
    X = [(s, 'OFF') for s in sentences]
    predict_dataset = prepare_prediction(X, tokenizer)
    eval_dataloader = DataLoader(predict_dataset, batch_size=4)
    prefix = ""

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(torch.from_numpy(preds)).numpy()
    # relevancy_scores = probabilities[:,1]

    return probabilities


def validate2(model, sentences, y_true, tokenizer, step=0, final=False, thres=0.5):
    val_loss = 0

    model.eval()
    scores = predict_sentences(model, tokenizer, sentences)
    y_pred = [a.argmax() for a in scores]

    print()
    val_loss = val_loss / len(sentences)
    accuracy = sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)

    print("y_pred", y_pred[0:30])
    print("y_true", y_true[0:30])
    f1 = f1_score(y_true, y_pred)

    #f = open('./plotting/classifier_scores.txt', 'w+')
    #f.write("y_true,y_score,y_pred\n")
    #f.write("\n".join([f"{y_true[i]},{y_scores[i]},{y_pred[i]}" for i in range(len(y_pred))]))
    #f.close()

    if(final):
        print("Results")
        print("\tAvg Loss:", val_loss)
        print("\tAccuracy:", accuracy)
        print("\tF1 Score:", f1)

        print(confusion_matrix(y_true, y_pred))
        #print(y_true)
        #print(y_scores)
        roc_auc = roc_auc_score(y_true, [x[0] for x in scores])
        print("roc_auc", roc_auc)

        map = {1:'toxic', 0:'safe'}
        y_true=[map[x] for x in y_true]
        y_pred=[map[x] for x in y_pred]
        print(classification_report(y_true, y_pred, digits=3))
    else:
        print("Avg Loss:", val_loss, " - Accuracy:", accuracy, "toxic_f1", f1_score(y_true, y_pred, average='binary', pos_label=1))
        wandb.log({"val_step": step, "accuracy":accuracy, "toxic_f1": f1_score(y_true, y_pred, average='binary', pos_label=1)})

if __name__ == '__main__':
    main()

#def old_load_AbuseEval(split=False, batch_size=1, max_samp=-1):
#    tweets = toxic_data.get_texts()
#    toxic_data.add_imp_exp(tweets)
#    x = [tweets[t]['tweet'] for t in tweets]
#    y = [1 if (tweets[t]['abuse_imp_exp'] in ['EXP', 'IMP'] or tweets[t]['off_imp_exp'] in ['EXP', 'IMP']) else 0 for t in tweets]
#
#    print('toxic', sum(y))
#    print('nontoxic', len(y) - sum(y))
#    print('total', len(y))
#
#    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#    x = tokenizer(x, return_tensors='pt', padding=True, truncation=True)['input_ids']
#    y = torch.tensor(y)
#
#    return make_data_loader(x, y, split, batch_size, max_samp)
