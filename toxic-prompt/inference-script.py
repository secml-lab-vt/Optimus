#New model inference code.
from tqdm import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import time
import os
import pickle as pkl
# use AdamW is a standard practice for transformer
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
# use Adafactor is the default setting for T5
from transformers.optimization import Adafactor
from openprompt.data_utils.utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm
from openprompt.data_utils.data_processor import DataProcessor


set_seed(108)


class MyDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["No", "Yes"]

    def random_sampling(self, dataset, sample_num):
        import random
        random.seed(42)
        # if exceed training data size, set to training data size, then;
        sample_num = min(sample_num, len(dataset["text"]))
        index_list = list(range(len(dataset["text"])))
        random.shuffle(index_list)
        selected_index_list = index_list[:sample_num]
        new_dataset = {
            "id": [dataset["id"][i] for i in selected_index_list],
            "text": [dataset["text"][i] for i in selected_index_list],
            "label": [dataset["label"][i] for i in selected_index_list]
        }
        return new_dataset

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"

        dataset = json.loads(open(data_dir).read())
        dataset = dataset[split]
        # sample_num = 1000
        sample_num = -1
        if sample_num != -1 and split == "train":
            dataset = self.random_sampling(dataset, sample_num)
            print("%s, sample %d data." % (split, len(dataset["id"])))
        return self.transform(dataset)

    def transform(self, dataset):
        res = []
        for i in range(len(dataset["text"])):
            # for i in range(1000):
            text_a = dataset['text'][i]
            label = int(dataset['label'][i])
            # for hateXplain
            if label == 2:
                label = 1
            guid = "{}".format(dataset['id'][i])

            res.append(InputExample(guid=guid, text_a=text_a, label=label))
        return res

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

Processor = MyDataProcessor
max_seq_l = 480  # this should be specified according to the running GPU's capacity
batchsize_t = 8
batchsize_e = 8
gradient_accumulation_steps = 4
model_parallelize = False

mytemplate = SoftTemplate(
    model=plm,
    tokenizer=tokenizer,
    num_tokens=200,
    initialize_from_vocab=True).from_file(
        "experiment_scripts/soft_template/soft_template.txt",
    choice=0)
# dataset = {}
# data_dir = 'parsed_dataset/measuring-hate-speech_perspective_balance.json'
# dataset['test'] = Processor().get_test_examples(data_dir)
# class_labels = Processor().get_labels()
# print(class_labels)
class_labels = ['Safe', 'Unsafe']
# quit()
myverbalizer = ManualVerbalizer(
    tokenizer,
    classes=class_labels).from_file(
        "experiment_scripts/soft_template/manual_verbalizer.txt", choice=0)
# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
use_cuda = True
prompt_model = PromptForClassification(
    plm=plm,
    template=mytemplate,
    verbalizer=myverbalizer,
    freeze_plm=(
        not True),
    plm_eval_mode=True)
if use_cuda:
    prompt_model = prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()


test_dataloader = PromptDataLoader(
    dataset=dataset["test"],
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=max_seq_l,
    decoder_max_length=3,
    batch_size=batchsize_e,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="tail")

def evaluate(prompt_model, dataloader, desc, return_data=False):
    # prompt_model.eval()
    allpreds = []
    alllabels = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    # prompt_model.train()
    acc = accuracy_score(alllabels, allpreds)
    p = precision_score(alllabels, allpreds)
    r = recall_score(alllabels, allpreds)
    f1 = f1_score(alllabels, allpreds)
    res = [acc, p, r, f1]
    if not return_data:
        return res
    elif return_data:
        return res, alllabels, allpreds

model_save_path = 'saved_models/measuring-hate-speech/t5/t5-base/2000.ckpt'
prompt_model.load_state_dict(torch.load(model_save_path))


val_res, labels, preds = evaluate(prompt_model, test_dataloader, desc="Valid", return_data=True)
acc, p, r, f1 = val_res
print("before training, val_res: ", val_res)