import sys, os
_LM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LM_ROOT not in sys.path:
    sys.path.insert(0, _LM_ROOT)

import sys
import torch
import transformers
from datasets import load_dataset,concatenate_datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import csv
from sklearn import metrics
import argparse
from typing import List,Tuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score,classification_report,roc_curve,confusion_matrix
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import datetime
from openai import OpenAI

import os

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError(
        "Set OPENAI_API_KEY in the environment before running OpenAI moderation scoring."
    )


def preprocess_conv(input=None,**kwargs):
    
    inference_example = input["context"] + " " + input["response"]

    if(input["label"] == "Unsafe"):
        ground = 1

    else:
        ground = 0


    prompts_list = []

    prompt = f"""{inference_example}"""

    prompts_list.append(prompt)
                    
    if input['implicit'] is None:
        return {"prompt_input":{"prompt": prompts_list },"ground": ground,"convs": inference_example,"context": input['context'],"response": input['response'],"category": input['category'],"label": input['label'],"implicit": "Nothing","source": input['source'],"index": input['index']}

    return {"prompt_input":{"prompt": prompts_list },"ground": ground,"convs": inference_example,"context": input['context'],"response": input['response'],"category": input['category'],"label": input['label'],"implicit": input['implicit'],"source": input['source'],"index": input['index']}

def main():

    parser = argparse.ArgumentParser(description='choose model')
    parser.add_argument('--path', required=True)
    parser.add_argument('--few_shot', required=True, choices=["yes","no"], default="no")
    parser.add_argument('--category', required=False, choices=["1","2"])
    parser.add_argument('--type', required=False, default="no")
    parser.add_argument('--metrics', required=False, choices=["yes","no"], default="no")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("cuda available")
        device = "cuda"
    else:
        device = "cpu"
    
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    param_dict = {
        "model_path" : "OPENAI-moderation",
        "model_name" : "OPENAI-moderation",
        "LOAD_8BIT" : False,
        "device" : device,
        "batch_size" : 1,
        "category" : args.category,
        "few_shot" : args.few_shot,
        "path" : args.path,
        "type" : args.type,
        "metrics" : args.metrics
    }

    mf = ModelFactory(param_dict)

    instructions = []
    

    path = f"../../Datasets/LM_Detect/{args.path}_dataset.csv"
        

    dataset = load_dataset("csv", data_files=path,split='train')


    dataset = dataset.map(preprocess_conv,remove_columns=dataset.features)

    dataset.to_json(f'../../Datasets/LM_Scores/Fewshot-{args.few_shot}/12/{mf.model_path}_{mf.ts}_category_{mf.category}_type_{mf.type}_{mf.path}_toxicity_dataset.json')

    mf.detect(dataset)

class ModelFactory():
    def __init__(self, param_dict):
        self.LOAD_8BIT = param_dict['LOAD_8BIT']
        self.device =  param_dict['device']
        self.batch_size = param_dict['batch_size']
        self.category = param_dict['category']
        self.few_shot = param_dict['few_shot']
        self.path = param_dict['path']
        self.type = param_dict['type']
        self.metrics = param_dict['metrics']
        self.model_path = param_dict["model_path"]
        self.client = OpenAI()

        
        self.ts = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")      

    def get_score(self, input_text):
        moderation = self.client.moderations.create(input = input_text)
        # print(moderation)

        # print(moderation.results[0].flagged)

        category_scores = moderation.results[0].category_scores

        category_dict = category_scores.to_dict()

        # # get max value in the dictionary
        # max_key = max(category_dict, key=category_dict.get)
        # get list of all the values in the dictionary
        all_values = category_dict.values()
        # convert the list of values to a list
        all_values = list(all_values)
        # return category_dict[max_key]

        return moderation.results[0].flagged, all_values
        
    
    def detect(self,dataset):

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        with open(f'../../Datasets/LM_Scores/Fewshot-{self.few_shot}/12/{self.model_path}_{self.ts}_category_{self.category}_{self.path}_type_{self.type}_toxicity_scores.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

            header = ['context','response','category','label','implicit','source','index','max_index',"sexual","hate","harassment","self-harm","sexual/minors","hate/threatening","violence/graphic","self-harm/intent","self-harm/instructions","harassment/threatening","violence",'max_prob','prediction'] 
            # for x, instruction in enumerate(instructions):
            #     header.append(f'Instruction_{x+1}')
            
            # header.append(f'max_index')
            # header.append(f'max_prob')
            # header.append(f'Prediction')
            # header.append(f'category')            

            spamwriter.writerow(header)

            preds = []
            labels = []

            results = []


            positive_preds = []
            positive_indices = []
            
            for _,data in enumerate(tqdm(train_loader),0):
                # if data["response"][0].isdigit():
                #     check_score = danum2words(data["response"][0])
                # else:
                check_score = data['context'][0] + " " + data["response"][0]

                # print(check_score)

                flag, all_Values = self.get_score(check_score)

                row = []
                heads = ['context','response','category','label','implicit','source','index']   
                for i in range(len(heads)):
                    row.append(data[heads[i]][0])  
                # row.append(data["context"][0])
                # row.append(data["response"][0])
                # row.append(data["ground"][0])

                    # for i in range(len(results)):
                    #     row.append(results[i][0][0][1])
                    #     if(results[i][0][0][1] > maximum):
                    #         maximum = results[i][0][0][1]
                    #         maximum_index = i
                    
                    # row.append(maximum_index)
                    # positive_preds.append(maximum)
                    # positive_indices.append(maximum_index)
                    # row.append(maximum)

                row.append(0)

                positive_preds.append(0)
                positive_indices.append(0)
                
                for i in range(len(all_Values)):
                    row.append(all_Values[i])

                # row.append(flag)

                # print(flag)

                if flag == True:
                    row.append(1)
                else:
                    row.append(0)

                if flag == True:
                    row.append("Unsafe")
                else:
                    row.append("Safe")

                # row.append(data["category"][0])

                spamwriter.writerow(row)
            
            if(self.metrics == "yes"):
                print('Roc_AUC: ',roc_auc_score(dataset['ground'], positive_preds))
                precisions, recalls, _ = precision_recall_curve(dataset['ground'], positive_preds)
                print('PR_AUC: ',auc(recalls, precisions))
                # convert probabilities to class labels
                yhat = [1 if p >= 0.5 else 0 for p in positive_preds]
                # compute precision, recall and f1-score
                precision = precision_score(dataset['ground'], yhat)
                recall = recall_score(dataset['ground'], yhat)
                f1 = f1_score(dataset['ground'], yhat)
                
                print(classification_report(dataset['ground'], yhat))
                print('Precision: %.3f' % precision)
                print('Recall: %.3f' % recall)
                print('F1-score: %.3f' % f1)       

                TN, FP, FN, TP = confusion_matrix(dataset['ground'], yhat).ravel()

                fpr1 = FP/(FP+TN)

                print('FPR: ',fpr1)
                
                fpr, tpr, thresholds = roc_curve(dataset['ground'], positive_preds)

                # print('FPR: ',fpr)
                # print('TPR: ',tpr)
                
                f = open(f'../../Datasets/LM_Scores/Fewshot-{self.few_shot}/consolidated_toxicity_results.txt', "a")
                # f.write('model, category, Roc_AUC, PR_AUC, Precision, Recall, F1-score')
                f.write(self.model_path+"_"+self.ts+","+self.category+","+self.few_shot+","+self.path+","+self.type+","+str(roc_auc_score(dataset['ground'], positive_preds))+","+str(auc(recalls, precisions))+","+str(precision)+","+str(recall)+","+str(f1)+","+str(fpr1))
                f.write('\n')
                f.close()
                
if __name__ == "__main__":
    main()