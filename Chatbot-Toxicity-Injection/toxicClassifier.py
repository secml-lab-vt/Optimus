from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import torch



class ToxicClassifier():
    def __init__(self, model_type, device):
        self.device = device
        self.model_type = model_type
        if(self.model_type == "WTC_bin_prec" or self.model_type == "WTC_bin"): #Precision tuned WTC classifier
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_new_focal-2_1_best').to(device)
        elif(self.model_type == "WTC_adv_bin_prec"):
            assert(False)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_adv_focal_shuffle_val_9_best').to(device)
        elif(self.model_type == "WTC_obscene"):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_obscene_9').to(device)
        elif(self.model_type == "WTC_insult"):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/WTC_insult_9').to(device)
        elif(self.model_type == "WTC_detadv_bin_prec"):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/text_fooler_detoxify_9_best/').to(device)
        elif(self.model_type == "unitary"):
            self.tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            self.model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(self.device)
        elif(self.model_type == "Perspective"):
            API_KEY = '<Put your API key>'
            from googleapiclient import discovery
            self.client = discovery.build(
              "commentanalyzer",
              "v1alpha1",
              developerKey=API_KEY,
              discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
              static_discovery=False,
            )
            df = pd.read_csv("./data/perspective/all_cache.csv")
            responses, scores = list(df["responses"]), list(df["scores"])
            self.perspective_cache = {}
            for i in range(len(scores)):
                res = responses[i].replace("notification", "chatbot").replace("flexibly", "chatbot").replace("cooperated", "chatbot").replace("manifesto", "chatbot").replace("competent", "chatbot")
                self.perspective_cache[res] = scores[i]
        else:
            print("Invalid Model Type should be one of these: ['WTC_bin_prec', 'WTC_bin', 'unitary']")
            exit()
        if(self.model_type != "Perspective"): self.model.eval()
        self.make_calls = False

    def perspective_request(self, input_text):
        input_text = input_text.replace("notification", "chatbot").replace("flexibly", "chatbot").replace("cooperated", "chatbot").replace("manifesto", "chatbot").replace("competent", "chatbot")
        if(self.make_calls):
            analyze_request = {
              'comment': { 'text': input_text},
              'requestedAttributes': {'TOXICITY': {}},
              'languages': ['en']
            }

            return self.client.comments().analyze(body=analyze_request).execute()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        else:
            if(input_text not in self.perspective_cache):
                print("not found", "'", input_text, "'")
            return self.perspective_cache[input_text]

    def get_score(self, input_text):
        if(self.model_type == "Perspective"):  return self.perspective_request(input_text)
        x = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.device)
        with torch.no_grad():
            logits = self.model(x)['logits'].detach().cpu()
        if(self.model_type in ["WTC_insult", "WTC_obscene", "WTC_bin_prec", "WTC_bin", "WTC_adv_bin_prec", "WTC_detadv_bin_prec"]):
            scores = softmax(logits, dim=1)[:,1].numpy()
            return scores[0]
        elif(self.model_type == "unitary"):
            odds = np.exp(logits[0][0].item())
            return odds / (1 + odds)

    def get_scores(self, input_texts):
        if(self.model_type == "Perspective"):  return [self.perspective_request(x) for x in input_texts]
        x = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.device)
        with torch.no_grad():
            logits = self.model(x)['logits'].detach().cpu()
        #odds = np.exp(scores[:, 0].detach().cpu().numpy())
        if(self.model_type in ["WTC_insult", "WTC_obscene", "WTC_bin_prec", "WTC_bin", "WTC_adv_bin_prec", "WTC_detadv_bin_prec"]):
            scores = softmax(logits, dim=1)[:,1].numpy()
            return list(scores)
        elif(self.model_type == "unitary"):
            scores = np.amax(logits.numpy(), axis=1)
            odds = np.exp(scores)
            return list(odds / (1 + odds))

    def classify_samples(self, input_texts):
        scores = self.get_scores(input_texts)
        if(self.model_type == "WTC_bin_prec"):
            return [x > 0.7185188829898834 for x in scores], scores
        elif(self.model_type == "WTC_adv_bin_prec"):
            return [x > 0.619 for x in scores], scores
        elif(self.model_type == "WTC_obscene"):
            return [x > 0.5270651578903198 for x in scores], scores
        elif(self.model_type == "WTC_insult"):
            return [x > 0.6277180314064026 for x in scores], scores
        elif(self.model_type == "WTC_detadv_bin_prec"):
            return [x > 0.64112228155 for x in scores], scores
        elif(self.model_type == "unitary" or self.model_type == "WTC_bin"):
            return [x > 0.5 for x in scores], scores
        elif(self.model_type == "Perspective"):
            return [x > 0.5 for x in scores], scores

    def classify_sample(self, input_text):
        score = self.get_score(input_text)
        if(self.model_type == "WTC_bin_prec"):
            return score > 0.7185188829898834, score
        elif(self.model_type == "WTC_detadv_bin_prec"):
            return score > 0.64112228155, score
        elif(self.model_type == "WTC_obscene"):
            return score > 0.5270651578903198, score
        elif(self.model_type == "WTC_insult"):
            return score > 0.6277180314064026, score
        elif(self.model_type == "WTC_adv_bin_prec"):
            return score > 0.619, score
        elif(self.model_type == "unitary" or self.model_type == "WTC_bin"):
            return score > 0.5, score
        elif(self.model_type == "Perspective"):
            return score > 0.5, score


class TextDataset(Dataset):
    def __init__(self, strings, labels):
        self.strings, self.labels = strings, labels

    def __getitem__(self, index):
        return self.strings[index], self.labels[index]

    def __len__(self):
        return len(self.strings)

def load_toxic_kaggle_data(file, batch_size=1):
    print("Loading - {}".format(file))
    df = pd.read_csv(file)

    x = df['comment_text'].tolist()
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
    y = y.where(y == 0, 1).tolist()

    d = TextDataset(x, y)
    return DataLoader(d, batch_size=batch_size)

def validate_class(val_loader, tc):
    val_loss = 0
    y_true, y_pred, y_scores = [], [], []

    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x = list(x)
            pred, scores = tc.classify_samples(x)
            #pred = [(1 if p > 0.5 else 0) for p in scores]
            y_scores.extend(scores)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(pred)

            print(f"\r{i}/{len(val_loader)}", end='')

    print()
    val_loss = val_loss / len(val_loader)
    accuracy = sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)

    print("y_pred", y_pred[0:10])
    print("y_true", y_true[0:10])
    f1 = f1_score(y_true, y_pred)

    print("Results")
    print("\tAvg Loss:", val_loss)
    print("\tAccuracy:", accuracy)
    print("\tF1 Score:", f1)

    print(confusion_matrix(y_true, y_pred))
    roc_auc = roc_auc_score(y_true, [1-x for x in y_scores])
    print("roc_auc", roc_auc)

    map = {1:'toxic', 0:'safe'}
    y_true=[map[x] for x in y_true]
    y_pred=[map[x] for x in y_pred]
    print(classification_report(y_true, y_pred, digits=3))

if(__name__ == "__main__"):
    #dl = load_toxic_kaggle_data('./data/wiki_toxic/WTC_test.csv', batch_size=10)
    import sys

    tc = ToxicClassifier('Perspective', device="cpu")
    tc.make_calls = True
    from dataEvaluator import DataEvaluator
    import time
    de = DataEvaluator()
    file_path = sys.argv[1]
    contexts, responses, flags = de.read_cache(file_path)
    flags = list(flags)
    preds, scores = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(responses):
        pred, score = tc.classify_sample(r)
        preds.append(pred)
        scores.append(score)
        eta = (time.perf_counter() - t0) * (len(responses) - i) / (i + 1)
        print(f"\r{i}/{len(responses)} - {eta:.2f}", end="")

    import os
    if(os.path.exists("./data/perspective/all_cache.csv")):
        df_old = pd.read_csv("./data/perspective/all_cache.csv")
        contexts += list(df_old["contexts"])
        responses += list(df_old["responses"])
        flags += list(df_old["flags"])
        preds += list(df_old["preds"])
        scores += list(df_old["scores"])

    import pandas as pd
    df = pd.DataFrame()
    df["contexts"] = contexts
    df["responses"] = responses
    df["flags"] = flags
    df["preds"] = preds
    df["scores"] = scores
    df.to_csv("./data/perspective/all_cache.csv", index=False)

    #validate_class(dl, tc)
    #print(1 - 0.2814811170101166)
    #tc = ToxicClassifier("else")
    print(tc.get_score("I like you. I love you"))
    print(tc.get_scores(["I like you.", "I love you", "I dislike you.", "I hate you"]))
