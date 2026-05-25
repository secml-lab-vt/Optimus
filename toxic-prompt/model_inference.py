from tqdm import tqdm
import torch
import datetime
import argparse
import torch.nn.functional as F
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

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument(
    "--plm_eval_mode",
    action="store_true",
    help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument(
    "--model",
    type=str,
    default='t5',
    help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='t5-small')
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--verbalizer_id", type=int, default=0)
parser.add_argument("--path", type=str)
parser.add_argument("--dataset", type=str, default='SBIC.v2')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--category", type=str, default='1')
parser.add_argument("--result_file", type=str, default="sfs_out/results.txt")
parser.add_argument("--max_steps", default=2000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=100)
parser.add_argument("--init_from_vocab", action="store_false")
parser.add_argument("--eval_every_steps", type=int, default=200)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
args = parser.parse_args()

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"
print(content_write)

set_seed(args.seed)

args.ts = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).replace(" ","_")
def preprocess_conv(input=None,**kwargs):
        contexts = " ".join(input["context"].strip().split("|"))
        responses = " ".join(input["response"].strip().split("|"))
        inference_example = contexts + " " + responses
        if(input["label"] == "Unsafe"):
            ground = 1
            label = 1
        else:
            ground = 0
            label = 0
        prompt = f"""{inference_example}"""
        if input['implicit'] is None:
            return {"text": prompt,"ground": ground,"convs": inference_example,"context": input['context'],"response": input['response'],"category": input['category'],"label": label,"implicit": "Nothing","source": input['source'],"id": input['index']}
        return {"text": prompt,"ground": ground,"convs": inference_example,"context": input['context'],"response": input['response'],"category": input['category'],"label": label,"implicit": input['implicit'],"source": input['source'],"id": input['index']}


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

print(args.model, 'args.model')
print('args.model_name_or_path',args.model_name_or_path)
plm, tokenizer, model_config, WrapperClass = load_plm(
    "t5", "t5-base")
dataset = {}

dataset_list = [
    "HateXplain",
    "USElectionHate20",
    "HateCheck",
    "SBIC.v2",
    "measuring-hate-speech"]
if args.dataset in dataset_list:
    data_dir = "parsed_dataset/%s_perspective_balance.json" % (args.dataset)
    Processor = MyDataProcessor
    # '''Our code starts'''
    from datasets import load_dataset, concatenate_datasets
    path = f"../Datasets/LM_Detect/{args.path}_dataset.csv"
    datasetown = load_dataset("csv", data_files=path,split='train')
    datasetown1 = datasetown
    datasetown = datasetown.map(preprocess_conv, remove_columns=datasetown.features)
    dataset['train'] = Processor().transform(datasetown)
    # '''Our code ends'''

    # dataset['train'] = Processor().get_train_examples(data_dir)
    # dataset['validation'] = Processor().get_dev_examples(data_dir)
    # dataset['test'] = Processor().get_test_examples(data_dir)
    # print(type(dataset['train']))
    # print(dataset['train'])
    class_labels = Processor().get_labels()
    # quit()
    max_seq_l = 480  # this should be specified according to the running GPU's capacity
    # model_parallelize = True
    batchsize_t = 8
    batchsize_e = 8
    gradient_accumulation_steps = 4
    model_parallelize = False

mytemplate = SoftTemplate(
    model=plm,
    tokenizer=tokenizer,
    num_tokens=args.soft_token_num,
    initialize_from_vocab=args.init_from_vocab).from_file(
        "experiment_scripts/soft_template/soft_template.txt",
    choice=0)

myverbalizer = ManualVerbalizer(
    tokenizer,
    classes=class_labels).from_file(
        "experiment_scripts/soft_template/manual_verbalizer.txt",
    choice=args.verbalizer_id)
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

use_cuda = True
prompt_model = PromptForClassification(
    plm=plm,
    template=mytemplate,
    verbalizer=myverbalizer,
    freeze_plm=(
        not args.tune_plm),
    plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()


test_dataloader = PromptDataLoader(
    dataset=dataset["train"],
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

print(
    "truncate rate: {}".format(
        test_dataloader.tokenizer_wrapper.truncate_rate),
    flush=True)

import csv
from torch.utils.data import DataLoader
def detect(self,dataset, toxic_prob_list):
    # print(dataset)
    # quit()
    dataset = dataset.remove_columns(['implicit'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # with open(f'../Datasets/LM_Scores/Testing/{self.type}/{self.model_path}_{self.ts}_category_{self.category}_{self.path}_type_{self.type}_toxicity_scores.csv', 'a', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL) #self.ts refer model factory in args. 
    
    with open(f'../Datasets/you-prompt-only/t5-base_{self.ts}_category_{self.category}_{self.path}_type_0_toxicity_scores.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        header = ['context','response','category','label','source','index','max_index','max_prob','prediction'] # we need ['max_index'-> 0,'max_prob'-> toxic class prob,'prediction'->0.5 greater safe]

        spamwriter.writerow(header)

        preds = []

        labels = []

        results = []
        # space is given to support all types of tokenizers

        output_choices = ['yes','no']
        heads = ['context','response','category','label','source','index']   

        positive_preds = []
        positive_indices = []


        for counter,data in tqdm(enumerate(train_loader, 0)):
        # for counter,data in zip(range(len(dataset)), dataset):

            # probability_dict = self.get_token_probability_distribution_seq2seq(data["prompt_input"]["prompt"][0][0], output_choices)
            print(counter)
            row = []
            maximum = -1

            # get probabilities for yes
            max_prob = toxic_prob_list[counter]
            # max_prob = toxic_prob_list[counter:((counter+1)*self.batch_size)]
            # if max_prob > 0.5:
            #     prediction = "Unsafe"
            #     maximum = max_prob
            # else:
            #     prediction = "Safe"
            #     maximum = max_prob
            if max_prob == 1:
                prediction = "Unsafe"
                maximum = max_prob
            else:
                prediction = "Safe"
                maximum = max_prob
            
            for i in range(len(heads)):
                row.append(data[heads[i]][0])

            row.append(0)
            positive_preds.append(maximum)
            positive_indices.append(0)
            row.append(maximum)
            row.append(prediction)

            
            spamwriter.writerow(row)




def evaluate(prompt_model, dataloader, datasetown, desc, return_data=False):
    # prompt_model.eval()
    allpreds = []
    alllabels = []
    toxic_class_list = []
    print(dataset['train'][5])
    print(datasetown1[5])
    # quit()
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        # print(inputs, inputs)
        #get probs here especially of the toxic class. logit of yes and no and apply softmax.
        logits = prompt_model(inputs)
        softmax_output = F.softmax(logits, dim=0)
        # print(softmax_output.shape)
        # print(logits, 'logits')
        # toxic_class_list.extend(softmax_output[:,1].tolist())
        # from IPython import embed
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        toxic_class_list.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # prompt_model.train()
    # return toxic_class_list
    acc = accuracy_score(alllabels, allpreds)
    p = precision_score(alllabels, allpreds)
    r = recall_score(alllabels, allpreds)
    f1 = f1_score(alllabels, allpreds)
    res = [acc, p, r, f1]
    return res
    if not return_data:
        return res
    elif return_data:
        return res, alllabels, allpreds


loss_func = torch.nn.CrossEntropyLoss()


if args.tune_plm:  # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    # it's always good practice to set no decay to biase and LayerNorm
    # parameters
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = [
        {
            'params': [
                p for n, p in prompt_model.plm.named_parameters() if (
                    not any(
                        nd in n for nd in no_decay))], 'weight_decay': 0.01}, {
            'params': [
                p for n, p in prompt_model.plm.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=500, num_training_steps=args.max_steps)
else:
    optimizer1 = None
    scheduler1 = None


optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters(
) if 'raw_embedding' not in name]}]  # note that you have to remove the raw_embedding manually from the optimization
if args.optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(
        optimizer_grouped_parameters2,
        lr=args.prompt_lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    # when num_warmup_steps is 0, it is the same as the configuration of
    # https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(
        optimizer2, num_warmup_steps=args.warmup_step_prompt)
elif args.optimizer.lower() == "adamw":
    optimizer2 = AdamW(
        optimizer_grouped_parameters2,
        lr=args.prompt_lr)  # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=args.warmup_step_prompt,
        num_training_steps=args.max_steps)  # usually num_warmup_steps is 500


# prompt_model.train()
model_save_path = 'saved_models/measuring-hate-speech/t5/t5-base/2000.ckpt'
prompt_model.load_state_dict(torch.load(model_save_path))


# val_res, labels, preds = evaluate(prompt_model, test_dataloader, datasetown, desc="Valid", return_data=True)
# toxic_prob = evaluate(prompt_model, test_dataloader, datasetown, desc="Valid", return_data=True)
acc, p, r, f1 = evaluate(prompt_model, test_dataloader, datasetown, desc="Valid", return_data=True)
print(f1, 'f1 score')
# acc, p, r, f1 = val_res
# print("before training, val_res: ", val_res)
# detect(args, datasetown1, toxic_prob)
quit()
pbar = tqdm(total=args.max_steps, desc="Train")
while glb_step <= args.max_steps:
    # print(f"Begin epoch {epoch}")
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        tot_train_time -= time.time()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1

        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss) / pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

        if optimizer1 is not None:
            optimizer1.step()
            optimizer1.zero_grad()
        if scheduler1 is not None:
            scheduler1.step()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

        tot_train_time += time.time()

        if actual_step % gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % args.eval_every_steps == 0:
            val_res, labels, preds = evaluate(
                prompt_model, test_dataloader, desc="Valid", return_data=True)
            acc, p, r, f1 = val_res
            statistics = [
                args.dataset,
                args.model_name_or_path,
                args.seed,
                glb_step,
                acc,
                p,
                r,
                f1]
            print("Glb_step {}, val_acc {}, average time {}".format(
                glb_step, val_res, tot_train_time / actual_step), flush=True)
            prompt_model.train()
            with open("sfs_out/task1/%s_%s_%s_%s.pkl" % (args.dataset, args.model_name_or_path, args.seed, glb_step), "wb") as wf:
                pkl.dump({"statistics": statistics,
                         "labels": labels, "preds": preds}, wf)
            with open("sfs_out/toxic_classification_result.csv", "a") as wf:
                wf.write(
                    "%s,%s,%s,%s,%s,%s,%s,%s\n" %
                    (args.dataset,
                     args.model_name_or_path,
                     args.seed,
                     glb_step,
                     acc,
                     p,
                     r,
                     f1))

            # save model
            save_path = "saved_models/%s/%s/%s/" % (
                args.dataset, args.model, args.model_name_or_path)
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                prompt_model.state_dict(), "%s%s.ckpt" %
                (save_path, glb_step))

        if glb_step > args.max_steps:
            leave_training = True
            break
    if leave_training:
        break


