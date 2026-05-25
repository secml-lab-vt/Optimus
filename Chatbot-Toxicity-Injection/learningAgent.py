from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import AdamW
import numpy as np
from baseAgent import BaseAgent
from toxicClassifier import ToxicClassifier
import json
import torch
import time
from quality import get_GRADE_scores
import sklearn
from sklearn.metrics import classification_report, f1_score
#from tensorflow_privacy.privacy.optimizers import dp_optimizer
from trl import DataCollatorForCompletionOnlyLM
from trl import DPOTrainer
#from quality import get_perplexity_scores
from util import allocated, elr_loss
from quality import BLEU2, LCS, single_perplexity, topk_ppl, all_topk_ppl
import random as r
import pandas as pd
import os
import pickle
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration,GPT2LMHeadModel, GPT2Tokenizer, AutoModelWithLMHead,AutoTokenizer, set_seed, GenerationConfig
from datasets import load_dataset,concatenate_datasets, Dataset
from trl import SFTTrainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline , modelcard


class LearningAgent(BaseAgent):
    def __init__(self, params, primary_device='cuda:0', secondary_device='cuda:1'):
        print("Initializing Learning Agent...")
        super().__init__()
        if(params.get('no_base', False)): return
        self.defense = params.get('defense', 'none')

        self.base_init(params['model_file'], params['decode_method'], params['model_type'], primary_device, secondary_device)

        self.name = 'Learning Bot'
        self.hello_message = "---I am a Learning Bot"
        self.goodbye_message = "---Thank you for the new experience!"

        self.do_learning = params.get('do_learning', True)
        if(self.do_learning):
            #self.epochs = params.get('epochs')
            self.max_epochs = params.get('max_epochs', 6)
            params["max_epochs"] = self.max_epochs   
            self.batch_size = params.get('batch_size')
            self.grad_accum = params.get('grad_accum', 1)
            self.early_stop = params.get('early_stop', False)
            self.train_frac = params.get('train_frac', 1.0)

        if(self.do_learning == False):
            return

        self.learning_rate = params.get('lr', 1e-6)
        self.GRADE_filter = 0
        self.PPL_filter = 1000000
        self.debug = params.get('debug', False)

        #Set up defense if applicable
        if(self.defense == 'PPL_filter_auto'):
            self.PPL_quantile = params.get('PPL_quantile')
            self.delay_cutoff = params.get('delay_cutoff', 0)
        elif(self.defense == 'PPL_soft_filter'):
            self.PPL_filter = params.get('PPL_filter')
        elif(self.defense == 'atcon'):
            self.toxic_filter_file = params['toxic_filter_file']
        elif(self.defense == 'gradient_shaping'):
            self.l2_norm_clip = params.get('nclip', 0)
            self.noise_multiplier = params.get('noise', 0)
            #assert(self.noise_multiplier != 0)
        elif(self.defense == 'PPL_filter'):
            self.PPL_filter = params.get('PPL_filter')
        elif(self.defense == 'GRADE_filter'):
            self.GRADE_filter = params.get('GRADE_filter')
        elif(self.defense == 'training_filter'):
            self.toxic_filter_file = params['toxic_filter_file']
        elif(self.defense == 'elr_reg'):
            self.elr_reg = True
        elif(self.defense == 'none'):
            pass
        elif(self.defense == 'activation-clustering'):
            pass
        else:
            raise ValueError(f"Unrecognized Defense: {self.defense}")

        self.params = params

        self.ppl_diff = params.get('ppl_diff', False)
        self.ppl_log_file = params.get('ppl_log_file', None)
        self.calc_ppl = False
        if(self.defense in ['PPL_filter', 'PPL_filter_auto', 'outlier_soft_filtering']):
            self.calc_ppl = True

    def save_model(self, save_file):
        self.model.save_pretrained(save_file)

    def train_dpo(self, training_pairs, training_flags,dpo_params): #Applies defense and does logging
        #Shuffle samples
        training_samples = [(training_pairs[i][0], training_pairs[i][1],training_pairs[i][2], training_flags[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]

        #Apply Defense
        learn = [True] * len(training_samples) #Used for filtering defenses
        logged_vals = {"contexts":[x[0] for x in training_samples],
                        "responses":[x[1] for x in training_samples],
                        "flags":[x[2] for x in training_samples]}
        if(self.defense == 'atcon'):
            self.atcon_init()
        print("\nPrimary Device Memory:", allocated(self.device))

        if(self.defense in ['training_filter', 'atcon']):
            toxic_filter = ToxicClassifier(self.toxic_filter_file, self.secondary_device)
            print('\nFiltering data...')
            correct = 0
            toxicity_pred = []
            toxicity_scores = []
            for i, p in enumerate(training_samples):
                pred, score = toxic_filter.classify_sample(p[1])
                if((p[2] in ['toxic', 'adv-toxic']) == pred): correct += 1
                toxicity_pred.append(pred)
                toxicity_scores.append(score)
            self.params['Training Filter Correct'] = correct
            self.params['Training Filter Accuracy'] = correct / len(training_samples)
            toxic_filter = ""
            torch.cuda.empty_cache()
            learn = [x == 0 for x in toxicity_pred]
            logged_vals['toxicity_pred'] = toxicity_pred
            logged_vals['toxicity_scores'] = toxicity_scores
        if(self.calc_ppl): #Calculate ppl for sample using current model
            self.model.eval()
            ppls = []
            print("\nCalculating PPL for samples...")
            for i, p in enumerate(training_samples):
                context, response, flag = p
                print(f"\rCalculating PPL for sample {i+1}/{len(training_samples)}", end="")
                if(self.defense == 'atcon'):
                    context_tokens, _ = self.tokenize([context], toxic=[toxicity_pred[i]], response=False)
                else:
                    context_tokens, _ = self.tokenize([context], response=False)
                response_tokens, _ = self.tokenize([response])
                ppl = single_perplexity(self.model, context_tokens, response_tokens, device=self.device).item()
                ppls.append(ppl)
            logged_vals['base_ppls'] = ppls
        if(self.defense == 'PPL_filter_auto'):
            assert(False)
            self.PPL_filter = self.get_PPL_threshold(training_pairs, training_flags)
        if(self.defense == 'GRADE_filter' or self.debug):
            print('\nComputing GRADE scores...')
            GRADE_scores, avg_GRADE = get_GRADE_scores(contexts, responses)
            logged_vals['GRADE_scores'] = GRADE_scores
            learn = [x > self.GRADE_filter for x in GRADE_scores]    

        if(self.defense in ['activation-clustering']):
            activation_pred = self.activation_clustering(training_samples)
            logged_vals['activation_pred'] = activation_pred[i]

        logged_vals["learn"] = learn
        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]
        print(f"Training on {len(used_samples)} samples")
        self.train_frac = dpo_params['train_frac']
        cutoff = int(self.train_frac * len(used_samples))
        print("Cutoff", cutoff)
        train_samples = used_samples[:cutoff]
        val_samples = used_samples[cutoff:]

        #Train Model
        t0 = time.perf_counter()
        if(self.defense == "activation-clustering"): assert(False)
        self.update_model_dpo(train_samples, val_samples, logged_vals,dpo_params)
        t1 = time.perf_counter()

        #Log Results
        print("To PPL LOG FILE", self.ppl_log_file)
        if(self.ppl_log_file != None):
            ppl_log_df = pd.DataFrame()
            for k in logged_vals:
                assert(len(logged_vals[k]) == len(training_samples))
                print("Adding", k, len(logged_vals[k]))
                ppl_log_df[k] = logged_vals[k]
            dir_path, _ = os.path.split(self.ppl_log_file)
            os.makedirs(dir_path, exist_ok=True)
            ppl_log_df.to_csv(self.ppl_log_file)

        self.log_training(len(training_pairs), t0, t1)

    def process_dpo_samples(self,dataset_samples):

        def preprocess_conv_dpo(input=None,**kwargs):

            if(self.model_type == 'BART'):
                context = input["context"].replace("|", "</s>")
            elif(self.model_type == 'Blenderbot'):
                context = input["context"].replace("|", "__end____start__")
            elif(self.model_type == 'Blenderbot_large' or self.model_type == 'BB400M'):
                context = input["context"].replace("|", "</s><s>")
            elif(self.model_type == 'DialoGPT'):
                context = (input["context"] + "|").replace("|", self.tokenizer.eos_token)
            elif(self.model_type == 'T5'):
                context = input["context"].replace("|", "</s>")
            else:
                print('WARNING: Unknown model architecture!')
                exit()

            return {"prompt": context, "chosen": input["response"], "rejected": input["rejected"]}

                # Extract text1 and text2 from each tuple
        text1_list = [item[0] for item in dataset_samples]
        text2_list = [item[1] for item in dataset_samples]
        text3_list = [item[2] for item in dataset_samples]
        text4_list = [item[3] for item in dataset_samples]

        # Create a dictionary with the extracted data
        data_dict = {
            "context": text1_list,
            "response": text2_list,
            "rejected": text3_list,
            "label": text4_list,
        }

        # Create a Hugging Face dataset
        my_dataset = Dataset.from_dict(data_dict)

        my_dataset = my_dataset.map(preprocess_conv_dpo,remove_columns=my_dataset.features,load_from_cache_file=False)

        return my_dataset


    def update_model_dpo(self, train_samples, val_samples, logged_vals,dpo_params): #Perform model fine-tuning and validation if needed
        
        def parse_log_history(log_history):
            """
            Parse the `log_history` of a Trainer to get the intermediate and final evaluation results.
            """
            idx = 0
            while idx < len(log_history) and "train_runtime" not in log_history[idx]:
                idx += 1

            # If there are no training logs
            if idx == len(log_history):
                idx -= 1
                while idx >= 0 and "eval_loss" not in log_history[idx]:
                    idx -= 1

                if idx >= 0:
                    return None, None, log_history[idx]
                else:
                    return None, None, None

            # From now one we can assume we have training logs:
            train_log = log_history[idx]
            lines = []
            training_loss = "No log"
            for i in range(idx):
                if "loss" in log_history[i]:
                    training_loss = log_history[i]["loss"]
                if "eval_loss" in log_history[i]:
                    metrics = log_history[i].copy()
                    _ = metrics.pop("total_flos", None)
                    epoch = metrics.pop("epoch", None)
                    step = metrics.pop("step", None)
                    _ = metrics.pop("eval_runtime", None)
                    _ = metrics.pop("eval_samples_per_second", None)
                    _ = metrics.pop("eval_steps_per_second", None)
                    _ = metrics.pop("eval_jit_compilation_time", None)
                    values = {"Training Loss": training_loss, "Epoch": epoch, "Step": step}
                    for k, v in metrics.items():
                        if k == "eval_loss":
                            values["Validation Loss"] = v
                        else:
                            splits = k.split("_")
                            name = " ".join([part.capitalize() for part in splits[1:]])
                            values[name] = v
                    lines.append(values)

            idx = len(log_history) - 1
            while idx >= 0 and "eval_loss" not in log_history[idx]:
                idx -= 1

            if idx > 0:
                eval_results = {}
                for key, value in log_history[idx].items():
                    if key.startswith("eval_"):
                        key = key[5:]
                    if key not in ["runtime", "samples_per_second", "steps_per_second", "epoch", "step"]:
                        camel_cased_key = " ".join([part.capitalize() for part in key.split("_")])
                        eval_results[camel_cased_key] = value
                return train_log, lines, eval_results
            else:
                return train_log, lines, None

        print("\nRunning Model Update...")

        self.train_params = TrainingArguments(
                report_to="wandb",
                output_dir=dpo_params['save_file'],
                num_train_epochs=dpo_params['max_epochs'],
                per_device_train_batch_size=dpo_params['batch_size'],
                gradient_accumulation_steps=1,
                optim="paged_adamw_32bit",
                save_strategy="no",
                # save_steps=100,
                logging_strategy="epoch",
                # logging_steps=25,
                learning_rate=dpo_params['lr'],
                weight_decay=0.0,
                # fp16=False,
                # bf16=True,
                # max_grad_norm=0.3,
                # max_steps=-1,
                # warmup_ratio=0.03,
                # group_by_length=True,
                # lr_scheduler_type="constant",
                evaluation_strategy="epoch",
                do_eval=True,
                run_name=f"{dpo_params['model_type']}_dpo_lr_{dpo_params['lr']}"
            )
            
        if(len(train_samples) == 0):
            return
        
        training_data = self.process_dpo_samples(train_samples)

        validation_data = self.process_dpo_samples(val_samples)

        print("Training data processed")
        print(training_data[0])
        print(validation_data[0])

        if(dpo_params['model_type'] == 'DD-BART'):
            self.model_ref = BartForConditionalGeneration.from_pretrained(dpo_params["dpo_model_folder"],local_files_only=True)
            self.model_ref.to(self.device)
        elif(dpo_params['model_type'] == 'BB400M'):
            self.model_ref = BlenderbotForConditionalGeneration.from_pretrained(dpo_params["dpo_model_folder"],local_files_only=True)
            self.model_ref.to(self.device)

        fine_tuning = DPOTrainer(
                self.model,                 # base model from SFT pipeline
                self.model_ref,             # typically a copy of the SFT trained base model
                beta=dpo_params["beta_param"],# temperature hyperparameter of DPO
                train_dataset=training_data, # dataset prepared above
                eval_dataset=validation_data,  # dataset prepared above
                tokenizer=self.tokenizer,   # tokenizer
                args=self.train_params     # training arguments e.g. batch size, lr, etc.
            )

        # Train the model
        train_result = fine_tuning.train()

        # Training
        fine_tuning.train()

        train_log, lines, _ = parse_log_history(fine_tuning.state.log_history)


        # with open(self.metrics_file_path, "a") as metrics_file:
        for line in lines:
            e = line['Epoch']
            best_val_ppl = np.exp(line['Validation Loss'])
                # metrics_file.write(f"\nEpoch: {e}")
                # metrics_file.write(f"\nTraining loss: {line['Training Loss']}")
                # metrics_file.write(f"\nValidation loss: {(line['Validation Loss'])}")
                # metrics_file.write(f"\nValidation Perplexity: {best_val_ppl}")
                # metrics_file.write(f"\n--------------------------------\n")
                # wandb.log({"epoch": e, "val loss": line['Validation Loss'], "train loss": line['Training Loss'], "perplexity": best_val_ppl})
        
        print("Validation loss:", line['Validation Loss'])
        print("Validation Perplexity", best_val_ppl)
        
        e = dpo_params['max_epochs']

        with open("./val_ppls.txt", "a+") as f:
            f.write(f"{e},{best_val_ppl}\n")

        # Save Model
        save_file = dpo_params['save_file']
        fine_tuning.save_model(save_file)
        final_epoch = 9999

        self.model = fine_tuning.model

        # self.log_run_to_db(self.max_epochs)
        return logged_vals.get("ppls", None)
    
    def train(self, training_pairs, training_flags): #Applies defense and does logging
        #Shuffle samples
        training_samples = [(training_pairs[i][0], training_pairs[i][1], training_flags[i]) for i in range(len(training_pairs))]
        ind = [i for i in range(len(training_samples))]
        r.shuffle(ind)
        training_samples = [training_samples[i] for i in ind]

        #Apply Defense
        learn = [True] * len(training_samples) #Used for filtering defenses
        logged_vals = {"contexts":[x[0] for x in training_samples],
                        "responses":[x[1] for x in training_samples],
                        "flags":[x[2] for x in training_samples]}
        if(self.defense == 'atcon'):
            self.atcon_init()
        print("\nPrimary Device Memory:", allocated(self.device))

        if(self.defense in ['training_filter', 'atcon']):
            toxic_filter = ToxicClassifier(self.toxic_filter_file, self.secondary_device)
            print('\nFiltering data...')
            correct = 0
            toxicity_pred = []
            toxicity_scores = []
            for i, p in enumerate(training_samples):
                pred, score = toxic_filter.classify_sample(p[1])
                if((p[2] in ['toxic', 'adv-toxic']) == pred): correct += 1
                toxicity_pred.append(pred)
                toxicity_scores.append(score)
            self.params['Training Filter Correct'] = correct
            self.params['Training Filter Accuracy'] = correct / len(training_samples)
            toxic_filter = ""
            torch.cuda.empty_cache()
            learn = [x == 0 for x in toxicity_pred]
            logged_vals['toxicity_pred'] = toxicity_pred
            logged_vals['toxicity_scores'] = toxicity_scores
        if(self.calc_ppl): #Calculate ppl for sample using current model
            self.model.eval()
            ppls = []
            print("\nCalculating PPL for samples...")
            for i, p in enumerate(training_samples):
                context, response, flag = p
                print(f"\rCalculating PPL for sample {i+1}/{len(training_samples)}", end="")
                if(self.defense == 'atcon'):
                    context_tokens, _ = self.tokenize([context], toxic=[toxicity_pred[i]], response=False)
                else:
                    context_tokens, _ = self.tokenize([context], response=False)
                response_tokens, _ = self.tokenize([response])
                ppl = single_perplexity(self.model, context_tokens, response_tokens, device=self.device).item()
                ppls.append(ppl)
            logged_vals['base_ppls'] = ppls
        if(self.defense == 'PPL_filter_auto'):
            assert(False)
            self.PPL_filter = self.get_PPL_threshold(training_pairs, training_flags)
        if(self.defense == 'GRADE_filter' or self.debug):
            print('\nComputing GRADE scores...')
            GRADE_scores, avg_GRADE = get_GRADE_scores(contexts, responses)
            logged_vals['GRADE_scores'] = GRADE_scores
            learn = [x > self.GRADE_filter for x in GRADE_scores]    

        if(self.defense in ['activation-clustering']):
            activation_pred = self.activation_clustering(training_samples)
            logged_vals['activation_pred'] = activation_pred[i]

        logged_vals["learn"] = learn
        used_samples = [x for i, x in enumerate(training_samples) if learn[i]]
        print(f"Training on {len(used_samples)} samples")
        cutoff = int(self.train_frac * len(used_samples))
        train_samples = used_samples[:cutoff]
        val_samples = used_samples[cutoff:]

        #Train Model
        t0 = time.perf_counter()
        if(self.defense == "activation-clustering"): assert(False)
        self.update_model(train_samples, val_samples, logged_vals)
        t1 = time.perf_counter()

        #Log Results
        print("To PPL LOG FILE", self.ppl_log_file)
        if(self.ppl_log_file != None):
            ppl_log_df = pd.DataFrame()
            for k in logged_vals:
                assert(len(logged_vals[k]) == len(training_samples))
                print("Adding", k, len(logged_vals[k]))
                ppl_log_df[k] = logged_vals[k]
            dir_path, _ = os.path.split(self.ppl_log_file)
            os.makedirs(dir_path, exist_ok=True)
            ppl_log_df.to_csv(self.ppl_log_file)

        self.log_training(len(training_pairs), t0, t1)

    def log_training(self, n, t0, t1):
        if(self.max_epochs == 0): return
        with open('./log.txt', 'a+') as f:
            f.write(f"{time.asctime()} - Trained {self.model_file}\n")
            f.write(f"Time Required - {t1 - t0:.2f}\n")
            if(self.defense != 'none'): f.write(f"Defense Used - {self.defense}\n")
            if(self.defense == 'PPL_filter_auto'):
                f.write(f"PPL Quantile - {self.PPL_quantile}\n")
                f.write(f"PPL Threshold - {self.PPL_filter}\n")
            if(self.defense in ['outlier_detection', 'outlier_soft_filtering']):
                if(self.use_reference_set):
                    f.write(f"percent_threshold - {self.percent_threshold}\n")
                    f.write(f"value_threshold - {self.value_threshold}\n")
                else:
                    f.write(f"Contamination - {self.contamination}\n")
            f.write(f"Number of Samples - {n}\n\n")

    def update_model(self, train_samples, val_samples, logged_vals): #Perform model fine-tuning and validation if needed
        print("\nRunning Model Update...")
        if(len(train_samples) == 0):
            return

        t = torch.Tensor(list(range(len(train_samples)))).unsqueeze(1)
        dataset = TensorDataset(t)
        data_loader = DataLoader(dataset, batch_size=1)
        self.model.train()
        if(self.defense == 'gradient_shaping'):
            from opacus import PrivacyEngine
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, _ = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=data_loader,
                noise_multiplier=0.0,
                max_grad_norm=4.0,
                poisson_sampling=False
            )
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        t0 = time.perf_counter()
        n = len(train_samples)
        in_batch_num, out_batch_num = 0, 0
        batch_contexts, batch_responses = [], []
        if(self.defense == 'atcon'): atcon_toxic = []
        if(self.defense == 'elr_reg'): loss_reg = elr_loss(num_classes=len(self.tokenizer))
        if(self.ppl_diff): ppl_after = []
        last_val_ppl = 1e9
        for e in range(self.max_epochs):
            print("\nEpoch", e)
            for i in data_loader:
                i = int(i[0].item())
                context, response, flag = train_samples[i]

                if(self.defense == 'atcon'):
                    atcon_toxic.append(logged_vals['toxicity_pred'][i])
                in_batch_num += 1
                batch_contexts.append(context)
                batch_responses.append(response)
                if(in_batch_num >= self.batch_size):
                    in_batch_num = 0
                    if(self.defense != 'atcon'):
                        x, _ = self.tokenize(batch_contexts, response=False)
                    else:
                        x, _ = self.tokenize(batch_contexts, toxic=atcon_toxic, response=False)
                    y, _ = self.tokenize(batch_responses)
                    batch_contexts, batch_responses = [], []
                    if(self.defense == 'atcon'): atcon_toxic = []

                    self.model.train()
                    if(self.model_type == 'DialoGPT'):
                        new_input = torch.cat((x, y), dim=1)
                        loss, logits = self.model(new_input, labels=new_input)[:2]
                    else:
                        loss, logits = self.model(x, labels=y)[:2]

                    if(self.defense in ["outlier_soft_filtering"]): loss *= grad_mult
                    if(self.defense == 'elr_reg'):
                        reg_loss = loss_reg(logits.transpose(1, 2), y)
                        loss = reg_loss

                    out_batch_num += 1

                    loss.backward()
                    if(out_batch_num >= self.grad_accum):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        out_batch_num = 0

                if(self.ppl_diff):
                    self.model.eval()
                    x, _ = self.tokenize([context], toxic=[toxicity_pred[i]], response=False)
                    y, _ = self.tokenize([response])
                    ppl_after.append(single_perplexity(self.model, x, y, device=self.device).item())
                    self.model.train()

                if(i % 10 == 0): #Print Progress
                    t1 = time.perf_counter()
                    eta = (t1 - t0) * (n - i + (self.max_epochs - e - 1) * n) / (i + 1 + e*n)
                    print(f"\r{i+1+(e*n)}/{n*self.max_epochs} {eta:.2f}", end="")
                    #print(f"\r{i+1+(e*n)}/{n*self.epochs} 'N/A'", end="")
            
            if(len(val_samples) == 0): continue
            val_ppls = []
            print("\nValidating Model...")
            for i in range(len(val_samples)):
                context, response, flag = val_samples[i]
                if(self.defense == 'atcon'):
                    x, _ = self.tokenize([context], toxic=[toxicity_pred[i]], response=False)
                else:
                    x, _ = self.tokenize([context], response=False)
                y, _ = self.tokenize([response])
                print(f"\r{i+1}/{len(val_samples)}", end="")
                val_ppls.append(single_perplexity(self.model, x, y, device=self.device).item())
            with open("./val_ppls.txt", "a+") as f:
                f.write(f"{e},{np.mean(val_ppls)}\n")
            if(np.mean(val_ppls) > last_val_ppl and e >= 2 and self.early_stop):
                #self.epochs = e
                self.params["epochs"] = e + 1
                break
            last_val_ppl = np.mean(val_ppls)

        if(self.defense == 'gradient_shaping'):
            self.model = self.model._module

        return logged_vals.get("ppls", None)

    def __call__(self, context, replay=False):
        if(type(context) == str):
            context = [context]
        output_text = self.generate(context)
        flags = ['victim' for x in output_text]

        return output_text, flags

    def get_PPL_threshold(self, training_pairs, training_flags):
        #create temp agent
        self.model = ""
        torch.cuda.empty_cache()

        #greedy is fine because, temp model isn't used for inference
        temp_agent = LearningAgent(params={"ppl_log_file": self.ppl_log_file.replace(".txt", "temp.txt"), "model_file":self.model_file, "decode_method":self.decode_method, "model_type":self.model_type, "epochs":1, "batch_size":self.batch_size})
        temp_agent.calc_ppl = True

        #train_model for 1000 Samples
        training_samples = [(training_pairs[i][0], training_pairs[i][1], training_flags[i]) for i in range(1000)] #Already shuffled
        print("Training Temp Model...")
        ppls = temp_agent.update_model(training_samples)
        temp_agent = ""
        torch.cuda.empty_cache()

        #restore model
        self.base_init(self.model_file, self.decode_method, self.model_type, self.device)

        #return quantile
        thres = np.quantile(ppls, self.PPL_quantile)
        print(f"Threshold={thres} at Quantile={self.PPL_quantile}")
        return thres

def get_hh_data(batch_size = 1, max_n=-1, device="cuda:0"):
    contexts = []
    responses = []
    with open("./data/self_feeding/train_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            if("__SILENCE__" in obj["context"]):
                continue
            c = '|'.join(obj["context"].replace('__p1__', '__p2__').split("__p2__"))
            contexts.append(c)
            responses.append(obj['response'])
    if(max_n != -1):
        contexts = contexts[:max_n]
        responses = responses[:max_n]

    print("len(responses)", len(responses))
    exit()

    dataset = StringDataset(contexts, responses)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

from transformers import GPT2LMHeadModel, GPT2Tokenizer
def create_base_agent(model_name, model_arch='BART', device="cuda:0"):
    if(model_arch == 'BART'):
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
        #model = BartForConditionalGeneration.from_pretrained('./saves/BART_clean.pt').cuda()
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)
    elif(model_arch == 'GPT-2'):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {
            'bos_token': "<bos>",
            'eos_token': "<eos>",
            'pad_token': "<pad>",
            'additional_special_tokens': ["<sp1>", "<sp2>"]
        }
        num_new_tokens = tokenizer.add_special_tokens(special_tokens)
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        vocab = tokenizer.get_vocab()
        pad_id = vocab["<pad>"]
        bos_id = vocab["<bos>"]
        eos_id = vocab["<eos>"]
        sp1_id = vocab["<sp1>"]
        sp2_id = vocab["<sp2>"]

        print('sp1_id', sp1_id)
        print('eos_id', eos_id)
    else:
        print('No matching architecture found!')
        return

    optimizer = AdamW(model.parameters(), lr=0.0001)

    print("Reading Data")
    data_loader = get_hh_data(batch_size=1, max_n=-1, device=device)
    print("Training Model")
    num_epochs = 1

    os.makedirs("./results", exist_ok=True)
    f = open('./results/base_train_ppl_log.txt', 'w+')
    for e in range(num_epochs):
        for i, (context, response) in enumerate(data_loader):
            context, response = context[0], response[0]
            if(model_arch == 'BART'):
                x = tokenizer(context.replace('|', '</s>'))
                y = tokenizer(response)

                model.train()
                print(f"\r{e+1}/{num_epochs} {i+1}/{len(data_loader)}", end="")
                optimizer.zero_grad()
                loss, logits = model(x, labels=y)[:2]
                loss.backward()
                optimizer.step()
            elif(model_arch == 'GPT-2'):
                turns = context.split('|')[1:]
                new_context = '<bos> ' + ''.join([('<sp1>' if i%2==0 else '<sp2>') + x for i,x in enumerate(turns)]) + ' <eos>'
                #print('context', context)
                #print('new_context', new_context)
                x = tokenizer(new_context, return_tensors='pt').to(device)
                y = tokenizer(response, return_tensors='pt').to(device)
                token_type_ids = torch.tensor([0]*len(x)).to(device)
                #print('x', x)
                #exit()

                model.train()
                print(f"\r{e+1}/{num_epochs} {i+1}/{len(data_loader)}", end="")
                optimizer.zero_grad()
                loss, logits = model(x, labels=y, token_type_ids=token_type_ids)[:2]
                loss.backward()
                optimizer.step()
            else:
                print('WARNING: Unknown model architecture!')
                exit()

class StringDataset(Dataset):
    def __init__(self, contexts, responses, flags):
        self.contexts = contexts
        self.responses = responses
        self.flags = flags

    def __getitem__(self, index):
        return self.contexts[index], self.responses[index], self.flags[index]

    def __len__(self):
        return len(self.responses)

class DBLDataset(Dataset):
    def __init__(self, samples):
        self.contexts, self.responses, self.flags = tuple(zip(*samples))

    def __getitem__(self, index):
        print("A", index, (self.contexts[index], self.responses[index], self.flags[index]))
        return index, self.contexts[index], self.responses[index], self.flags[index]

    def __len__(self):
        print("B")
        return len(self.responses)

def get_token_lens():
    contexts = []
    responses = []
    with open("./data/self_feeding/train_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            if("__SILENCE__" in obj["context"]):
                continue
            c = '|'.join(obj["context"].replace('__p1__', '__p2__').split("__p2__"))
            contexts.append(c)
            responses.append(obj['response'])

    lens = [len(x.split()) for x in responses]
    len_counts = {}
    for x in lens:
        len_counts[x] = 1+len_counts.get(x, 0)
    return len_counts

def main():
    create_base_agent(model_name='./saves/base/trash', model_arch='BART')
    