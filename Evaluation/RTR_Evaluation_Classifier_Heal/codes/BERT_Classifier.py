from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import json
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from time import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import torch.nn as nn
import random
import os
import torch.nn.functional as F
from datasets import load_dataset,concatenate_datasets,Dataset
from sklearn.metrics import precision_recall_curve

num_labels = -1

padsize = 512
num_epochs = 5

from kornia.utils.one_hot import one_hot


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

def get_loader(dataset, tokenizer, batchsize=16, padsize=256):
    batch_inputs, batch_labels = [], []

    inputs1, inputs2, categories, labels_ = [d['context'] for d in dataset], [d['response'] for d in dataset], [d['category'] for d in dataset], [d['label'] for d in dataset]
    # print(inputs1[0:5])
    # print(inputs2[0:5])
    # print(labels_[0:5])
    labels = []
    # print(labels_)
    for label in labels_:
        # print(label)
        if label == 'Unsafe':
            labels.append(1)
        else:
            labels.append(0)
    
    for start in tqdm(range(0, len(inputs1), batchsize)):
        tmp_batch = tokenizer(text=inputs1[start:min(start + batchsize, len(inputs1))],
                              text_pair=inputs2[start:min(start + batchsize, len(inputs1))],
                              return_tensors="pt", truncation=True, padding='max_length', max_length=padsize)
        batch_inputs.append(tmp_batch)
        tmp_label = torch.LongTensor(labels[start:min(start + batchsize, len(inputs1))])
        batch_labels.append(tmp_label)
    # print(batch_labels)
    return batch_inputs, batch_labels

def evaluate(model, batch_inputs, batch_labels,test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    labels_prob = np.array([], dtype=float)

    thres = 0.5
    if args.category == 'category1':
        if args.classifier_type == 'combined':
            if args.mode == 'test-precision_tuned' or args.mode == 'val-precision_tuned':
            # 0.8973080757726819,0.6,0.5927848815917969
                thres = 0.5927848815917969
            else:
                thres = 0.5
    elif args.category == 'category2':
        if args.classifier_type == 'combined':
            if args.mode == 'test-precision_tuned' or args.mode == 'val-precision_tuned':
                #0.861244019138756,0.6,0.6396305561065674
                thres = 0.6396305561065674
            else:
                thres = 0.5

    with torch.no_grad():
        for inputs, labels in zip(batch_inputs, batch_labels):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            loss_total += loss
            scores = F.softmax(logits, dim=1)
            # print("Scores:",scores)
            preds = scores[:, 1].cpu().numpy()
            # print("preds:",preds)
            # print(logits.view(-1, logits.shape[-1]).data)
            # print(torch.max(logits.view(-1, logits.shape[-1]).data, 1))
            # print(torch.max(logits.view(-1, logits.shape[-1]).data, 1)[1])
            labels = labels.view(-1).data.cpu().numpy()
            predic = [(1 if p > thres else 0) for p in scores[:, 1]]
            
            # predic = torch.max(logits.view(-1, logits.shape[-1]).data, 1)[1].cpu()
            
            labels_prob = np.append(labels_prob, preds)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    if args.mode == 'precision_tuning':

        precision, recall, thresholds = precision_recall_curve(labels_all, labels_prob)
        
        thresholds = np.append(thresholds,-1)

        DF_LISTS = pd.DataFrame(
                    {'precision': precision,
                    'recall': recall,
                    'threshold': thresholds
                    })

        dataset = Dataset.from_pandas(DF_LISTS)
        dataset.to_csv(f"Precision_tuning_{args.category}_{args.classifier_type}.csv")     

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='weighted')
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        report_dict = metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(batch_inputs), report,report_dict, confusion, labels_all, predict_all
    return acc, loss_total / len(batch_inputs), f1

def test_report(model, save_path, batch_inputs, batch_labels, log_file):
    # test
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time()
    test_acc, test_loss, test_report,test_report_dict, test_confusion, label, predict = evaluate(model, batch_inputs, batch_labels,
                                                                                test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc), file=log_file)
    print("Precision, Recall and F1-Score...")
    print(test_report, file=log_file)
    print("Confusion Matrix...")
    print(test_confusion, file=log_file)
    time_dif = time() - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print("Time usage:", time_dif, file=log_file)
    return test_report_dict['weighted avg']['f1-score']



parser = argparse.ArgumentParser(description='choose dataset')

parser.add_argument('--category', required=True, choices=['category1', 'category2'])
parser.add_argument('--classifier_type', required=True, choices=['combined'])
parser.add_argument('--mode', required=True, choices=['train', 'val', 'test',"val-precision_tuned","precision_tuning","test-precision_tuned"])

args = parser.parse_args()

from json import loads

if(args.category == 'category1'):
    if(args.classifier_type == 'combined'):
        train_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_train.json'
        val_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_val.json'
        test_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_test.json'

elif(args.category == 'category2'):
    if(args.classifier_type == 'combined'):
        train_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_train.json'
        val_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_val.json'
        test_path = '../../../Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_test.json'

with open(train_path, "r") as f2r:
    train = [loads(each_line) for each_line in f2r]
with open(test_path, "r") as f2r:
    test = [loads(each_line) for each_line in f2r]
with open(val_path, "r") as f2r:
    val = [loads(each_line) for each_line in f2r]


if args.category == 'category1':
    if args.classifier_type == 'combined':
        directory_name = 'binary_category1'
        learning_rates = [5e-6]
elif args.category == 'category2':
    if args.classifier_type == 'combined':
        directory_name = 'binary_category2'
        learning_rates = [5e-6]

num_labels = 2 

padsize = 512
num_epochs = 5


require_improvement = 300 # can be adjusted


import itertools

batchsizes = [16]



# learning_rates = [5e-6]

weight = [1,1] # can be adjuested
weight = torch.FloatTensor(weight)

import sys
#log_file = sys.stdout

if(args.mode == 'train'):

    for batchsize, learning_rate in itertools.product(batchsizes,learning_rates):
        path = 'bert-base-cased'
        # path = 'bert-base-cased-cased'
        if not os.path.isdir('../models_{}'.format(directory_name)):
            os.mkdir('../models_{}'.format(directory_name))
        save_path = '../models_{}/model_{}_{}'.format(directory_name, batchsize, learning_rate)
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        total_batch = 0
        dev_best_loss = float('inf')
        best_f1 = 0
        last_improve = 0
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        print("getting loader...")
        #train_inputs, train_labels = get_loader(train, tokenizer, batchsize=batchsize, padsize=padsize)
        val_inputs, val_labels = get_loader(val, tokenizer, batchsize=batchsize, padsize=padsize)
        test_inputs, test_labels = get_loader(test, tokenizer, batchsize=batchsize, padsize=padsize)


        model = model.to(device)
        flag = False
        weight = weight.to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=weight)
        gamma = 2.0
        alpha = 1.0
        loss_fct = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

        print("start to train...")
        for epoch in range(num_epochs):
            model.train()
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            start_time = time()
            random.seed(42)
            random.shuffle(train)
            train_inputs, train_labels = get_loader(train, tokenizer, batchsize=batchsize, padsize=padsize)
            for i, (trains, labels) in enumerate(zip(train_inputs, train_labels)):
                trains, labels = trains.to(device), labels.to(device)
                outputs = model(**trains, labels=labels)

                #loss = outputs.loss
                logits = outputs.logits
                loss = loss_fct(logits, labels)

                model.zero_grad()

                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    true = labels.view(-1).data.cpu()
                    predic = torch.max(logits.view(-1, logits.shape[-1]).data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss, dev_f1 = evaluate(model, val_inputs, val_labels)
                    if dev_f1>best_f1:
                        best_f1 = dev_f1
                    #if dev_loss < dev_best_loss:
                    #    dev_best_loss = dev_loss
                        torch.save(model.state_dict(), save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = time() - start_time
                    time_dif = timedelta(seconds=int(round(time_dif)))
                    
                    # wandb.log({"train_acc": train_acc, "train_loss": loss.item()})
                    # wandb.log({"val_acc": dev_acc, "val_loss": dev_loss})

                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%}  Time: {6} {7}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, dev_f1, time_dif, improve))
                    model.train()
                total_batch += 1
                if total_batch - last_improve > require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        if not os.path.isdir('../logs_{}'.format(directory_name)):
            os.mkdir('../logs_{}'.format(directory_name))
        log_file = open('../logs_{}/log_{}_{}.txt'.format(directory_name, batchsize, learning_rate),'w')
        print('batchsize: {}\nlearning_rate:{}'.format(batchsize,learning_rate), file=log_file)
        test_report(model, save_path, val_inputs, val_labels, log_file=log_file)
        log_file.close()
elif(args.mode == 'val' or args.mode == 'val-precision_tuned' or args.mode == 'precision_tuning'):
  best_precision = -1
  best_model = ""
  for batchsize, learning_rate in itertools.product(batchsizes,learning_rates):
        # path = 'bert-base-cased-cased'
        path = 'bert-base-cased'
        if not os.path.isdir('../models_{}'.format(directory_name)):
            os.mkdir('../models_{}'.format(directory_name))
        save_path = '../models_{}/model_{}_{}'.format(directory_name, batchsize, learning_rate)
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
        # tokenizer = BertTokenizer.from_pretrained(path)
        # model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        total_batch = 0
        dev_best_loss = float('inf')
        best_f1 = 0
        last_improve = 0
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        print("getting loader...")
        val_inputs, val_labels = get_loader(val, tokenizer, batchsize=batchsize, padsize=padsize)
        test_inputs, test_labels = get_loader(test, tokenizer, batchsize=batchsize, padsize=padsize)

        model = model.to(device)
        flag = False
        weight = weight.to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=weight)
        gamma = 2.0
        alpha = 1.0
        loss_fct = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

        if not os.path.isdir('../logs_{}'.format(directory_name)):
            os.mkdir('../logs_{}'.format(directory_name))
        if not os.path.isdir('../logs_{}/{}'.format(directory_name,args.mode)):
            os.mkdir('../logs_{}/{}'.format(directory_name,args.mode))
        log_file = open('../logs_{}/{}/log_{}_{}.txt'.format(directory_name,args.mode, batchsize, learning_rate),'w')
        print('batchsize: {}\nlearning_rate:{}'.format(batchsize,learning_rate), file=log_file)
        precision_val = test_report(model, save_path, val_inputs, val_labels, log_file=log_file)
        if best_precision < precision_val:
            print("updated_best_model")
            best_precision = precision_val
            best_model = log_file
        log_file.close()

  print("best_precision",str(best_precision))
  print("best_model",best_model)

elif(args.mode == 'test' or args.mode == 'test-precision_tuned'):
    best_precision = -1
    best_model = ""
    for batchsize, learning_rate in itertools.product(batchsizes,learning_rates):
        # path = 'bert-base-cased-cased'
        path = 'bert-base-cased'
        if not os.path.isdir('../models_{}'.format(directory_name)):
            os.mkdir('../models_{}'.format(directory_name))
        save_path = '../models_{}/model_{}_{}'.format(directory_name, batchsize, learning_rate)
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
        # tokenizer = BertTokenizer.from_pretrained(path)
        # model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        total_batch = 0
        dev_best_loss = float('inf')
        best_f1 = 0
        last_improve = 0
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        print("getting loader...")
        val_inputs, val_labels = get_loader(val, tokenizer, batchsize=batchsize, padsize=padsize)
        test_inputs, test_labels = get_loader(test, tokenizer, batchsize=batchsize, padsize=padsize)

        model = model.to(device)
        flag = False
        weight = weight.to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=weight)
        gamma = 2.0
        alpha = 1.0
        loss_fct = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

        if not os.path.isdir('../logs_{}'.format(directory_name)):
            os.mkdir('../logs_{}'.format(directory_name))
        if not os.path.isdir('../logs_{}/{}'.format(directory_name,args.mode)):
            os.mkdir('../logs_{}/{}'.format(directory_name,args.mode))
        log_file = open('../logs_{}/{}/log_{}_{}.txt'.format(directory_name,args.mode, batchsize, learning_rate),'w')
        print('batchsize: {}\nlearning_rate:{}'.format(batchsize,learning_rate), file=log_file)
        precision_val = test_report(model, save_path,  test_inputs, test_labels , log_file=log_file)
        if best_precision < precision_val:
            print("updated_best_model")
            best_precision = precision_val
            best_model = log_file
        log_file.close()
                
    print("best_precision",str(best_precision))
    print("best_model",best_model)
