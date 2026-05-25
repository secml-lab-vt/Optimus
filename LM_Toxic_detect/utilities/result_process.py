import sys, os
_LM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LM_ROOT not in sys.path:
    sys.path.insert(0, _LM_ROOT)


from datasets import load_dataset
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score,classification_report,roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix



def preprocess_conv(input=None,**kwargs):

    ground = float(input["ground"].replace("tensor(","").replace(")",""))
    lis = []
    for i in range(7):
        lis.append(float(input["Instruction_"+str(i+1)]))

    max = lis[0]

    for x in lis:
        if x > max:
            max = x
    
    y_pred = max

    yhat = 1 if y_pred >= 0.5 else 0

    return {"y_pred":y_pred,"yhat":yhat,"ground":ground}
    

# path = "../../Datasets/LM_Toxic_detect/test_dataset/facebook_opt-iml-30b_category_1_toxicity_scores.csv"

# path = "../../Datasets/LM_Toxic_detect/test_dataset/lmsys_vicuna-33b-v1.3_category_1_toxicity_scores.csv"

path = "../../Datasets/LM_Toxic_detect/test_dataset/tiiuae_falcon-40b-instruct_category_1_toxicity_scores.csv"

dataset = load_dataset("csv", data_files=path,split='train')

dataset = dataset.map(preprocess_conv,remove_columns=dataset.features)


y_true = dataset["ground"]
y_pred = dataset["y_pred"]
yhat = dataset["yhat"]

print(classification_report(y_true,yhat))

print("Confusion Matrix: \n",confusion_matrix(y_true,yhat))

print("Precision: ",precision_score(y_true,yhat))
print("Recall: ",recall_score(y_true,yhat))
print("F1: ",f1_score(y_true,yhat))

precisions, recalls, _ = precision_recall_curve(y_true, y_pred)

print('PR_AUC: ',auc(recalls, precisions))

precision = precision_score(y_true, yhat)
recall = recall_score(y_true, yhat)
f1 = f1_score(y_true, yhat)


f = open(f'./test_dataset/consolidated_toxicity_results.txt', "a")
# f.write('model, category, Roc_AUC, PR_AUC, Precision, Recall, F1-score')
f.write(path+","+str(1)+","+str(roc_auc_score(y_true, y_pred))+","+str(auc(recalls, precisions))+","+str(precision)+","+str(recall)+","+str(f1))
f.write('\n')
f.close()

