from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification,BlenderbotSmallTokenizer,BlenderbotSmallForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
import numpy as np

import torch
from baseAgent import BaseAgent
import time
import toxic_data
import random as r
from trojanAgent import insert_trigger
import util

#USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if USE_CUDA else "cpu")
#print("Using CUDA device." if USE_CUDA else "CUDA device not found!")

class ToxicTrojanAgent(BaseAgent):
    def __init__(self, params, primary_device='cuda:0', secondary_device='cuda:1'):
        print("Initializing Toxic Trojan Agent...")
        super().__init__()
        self.atcon = False

        self.primary_device = primary_device
        self.secondary_device = secondary_device

        self.params = params
        self.toxic_mode = params['toxic_mode']
        if(self.toxic_mode == "sample"):
            self.sample_type = params['sample_type']
            self.samples = toxic_data.get_toxic_data(params['sample_type'],
                            max_len=params.get('max_len',-1),
                            max_n=params.get('max_n',-1),
                            split_sentences=params.get('split_sentences',False),
                            use_toxic_classifier=params.get('use_classified_samples',False))
        elif(self.toxic_mode == "generation"):
            self.toxic_agent = BaseAgent()
            self.toxic_agent.base_init(params['attacker_model_file'], params['attacker_decode'], params['attacker_type'], primary_device, secondary_device)
        elif(self.toxic_mode == "single"):
            self.single_response = params["single_response"]
        else:
            raise Exception("Unknown value for toxic_mode: " + str(params['toxic_mode']))

        #Skip loading friendly model initially
        #self.base_init(params['model_file'], params['decode_method'], params['model_type'], primary_device)

        self.name = 'Toxic Trojan Bot'
        self.hello_message = "---I am an Toxic Trojan Bot!!!"
        self.goodbye_message = "---Farewell."

        self.trigger = params['trigger']
        if(type(self.trigger) == str):
            self.trigger = [self.trigger]
        self.res_poison_rate = params['res_poison_rate']


    def swap_model(self, toxic=False):
        if(toxic == False):
            self.toxic_agent = ""
            torch.cuda.empty_cache()
            self.base_init(self.params['model_file'], self.params['decode_method'], self.params['model_type'], self.primary_device)
        else:
            self.model = ""
            torch.cuda.empty_cache()
            self.toxic_agent = BaseAgent()
            self.toxic_agent.base_init(self.params['attacker_model_file'], self.params['attacker_decode'], self.params['attacker_type'], self.primary_device, self.secondary_device)


    def __call__(self, context, attack = None):
        if(type(context) == str):
            context = [context]

        atk_flags = ['friendly'] * len(context)
        #output_text = self.generate(context)
        output_text = [""] * len(context)

        if(attack == None):
            output_text = self.generate(context)
            return output_text, atk_flags

        elif(all([x==2 for x in attack])):
            for i in range(len(output_text)):
                if(self.toxic_mode == 'sample'):
                    output_text[i] = r.choice(self.samples)
                elif(self.toxic_mode == 'generation'):
                    output_text[i] = self.toxic_agent.generate(context[i])[0]
                elif(self.toxic_mode == "single"):
                    output_text[i] = self.single_response
                atk_flags[i] = 'response'
        elif(all([x==1 for x in attack])):
            output_text = self.generate(context)
            for i in range(len(output_text)):
                #if(r.random() < self.res_poison_rate):
                output_text[i] = insert_trigger(output_text[i], r.choice(self.trigger))
                atk_flags[i] = 'trigger'
        else:
            print("Incorrect Flags!!!")
            exit()

        return output_text, atk_flags

def get_loader(file_path, dataset='BAD', split=False, max_n = -1):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

    f = open(file_path)
    contexts = []
    responses = []
    i = 0
    for line in f.read().strip().split("\n"):
        if(i == max_n):
            break
        i += 1
        if(line == ""):
            continue
        if(dataset == 'BAD'):
            parsed = {x[0]:x[1] for x in map(lambda x: x.split(":"), line.split("\t"))}
            if(parsed["labels"] == "__notok__"):
                sample = parsed["text"].split("\\n")
                contexts.append("</s>".join(sample[:-1]))
                responses.append(sample[-1])
        elif(dataset == 'reddit'):
            parsed = line.split("\t")
            responses.append(parsed[1])
            contexts.append("</s>".join(parsed[0].split("|")))
            print("context: " + contexts[-1])
            print("respone: " + responses[-1])

    context_ids = tokenizer.batch_encode_plus(contexts, return_tensors='pt', padding=True)['input_ids']
    response_ids = tokenizer.batch_encode_plus(responses, return_tensors='pt', padding=True)['input_ids']

    dataset = TensorDataset(context_ids, response_ids)
    if(split):
        data_len = len(dataset)
        train_len, test_len = int(0.89 * data_len), int(0.1 * data_len)
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len,test_len,data_len-train_len-test_len])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3, shuffle=True, num_workers=4)

        return train_loader, test_loader, val_loader
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)
        return loader


def validate(model, val_loader):
    model.eval()
    val_loss = 0
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            loss, logits = self.model(x, labels=y)[:2]

            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(model, optimizer, train_loader, val_loader=None, epochs=4, save_model='trash', device='cuda:0'):
    wandb.init(project="evil_bot")
    wandb.watch(self.model)

    if(val_loader != None):
        val_loss = self.validate(val_loader)
        wandb.log({"val_loss": val_loss, "epoch":0})

    t0 = time.perf_counter()

    for e in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            model.train()

            optimizer.zero_grad()
            loss, logits = model(x, labels=y)[:2]
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss})

            t1 = time.perf_counter()
            print(f'\repoch: {e+1}/{epochs} | batch: {i+1}/{len(train_loader)} | loss: {loss.item():.4f} | eta: {(len(train_loader) * (epochs-e) - i) * (t1  - t0) / (len(train_loader) * e + i + 1)}', end="")

            if(val_loader != None and (i+1) % 20000 == 0):
                model.save_pretrained(f"./saves/toxic_bot/{save_model}_{e+1}_{i+1}")
                val_loss = validate(val_loader)
                wandb.log({"val_loss": val_loss, "epoch":e + i/len(train_loader)})

        model.save_pretrained(f"./saves/toxic_bot/{save_model}_{e+1}")

def to_loader(contexts, responses):
    contexts = [x[0].replace('|', '</s>') for x in train_pairs]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

def main():
    #train_pairs = toxic_data.get_toxic_reddit(threshold=0.99, clean_context=True)
    #print(f"Found {len(train_pairs)} training pairs.")

    dividing = True
    if(dividing):
        pairs = toxic_data.get_toxic_reddit(threshold=0.99, clean_context=False)
        print(f"Found {len(pairs)} pairs.")

        r.seed(0)
        r.shuffle(pairs)

        t_num = int(len(pairs) * 0.85)
        v_num = int(len(pairs) * 0.9)
        training_contexts, training_responses = tuple(zip(*pairs[:t_num]))
        val_contexts, val_responses = tuple(zip(*pairs[t_num:v_num]))
        test_contexts, test_responses = tuple(zip(*pairs[v_num:]))

        df = pd.DataFrame(data={'contexts': training_contexts, 'responses': training_responses})
        df.to_csv("./data/toxic_bot/reddit_thres-0.99_cc_training.csv")
        df = pd.DataFrame(data={'contexts': val_contexts, 'responses': val_responses})
        df.to_csv("./data/toxic_bot/reddit_thres-0.99_cc_validation.csv")
        df = pd.DataFrame(data={'contexts': test_contexts, 'responses': test_responses})
        df.to_csv("./data/toxic_bot/reddit_thres-0.99_cc_test.csv")
        exit()
    else:
        df = pd.read_csv("./data/toxic_bot/reddit_thres-0.99_cc_training.csv")
        training_contexts, training_responses = list(df["responses"]), list(df["contexts"])
        df = pd.read_csv("./data/toxic_bot/reddit_thres-0.99_cc_validation.csv")
        val_contexts, val_responses = list(df["responses"]), list(df["contexts"])
        df = pd.read_csv("./data/toxic_bot/reddit_thres-0.99_cc_test.csv")
        test_contexts, test_responses = list(df["responses"]), list(df["contexts"])

    #train_pairs = toxic_data.get_toxic_reddit(threshold=0.9, clean_context=True)
    #print(f"Found {len(train_pairs)} training pairs.")

    #train_pairs = train_pairs[:30_000]

    contexts = [x[0].replace('|', '</s>') for x in train_pairs]
    responses = [x[1] for x in train_pairs]

    device = 'cuda:0'

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

    #tokenizer.add_tokens(['<toxic>', '<safe>'])
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
    model.resize_token_embeddings(len(tokenizer))

    context_ids = tokenizer.batch_encode_plus(contexts, return_tensors='pt', padding=True)['input_ids']
    response_ids = tokenizer.batch_encode_plus(responses, return_tensors='pt', padding=True)['input_ids']
    dataset = TensorDataset(context_ids, response_ids)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    train_pairs, contexts, responses = '', '', ''

    optimizer = AdamW(model.parameters(), lr=1e-6)
    train(model, optimizer, train_loader, epochs=40, save_model='reddit_bot_lr-1e6', device=device)

    #print('\n'.join(toxic_comments[:10]))
    #print("Reading files")
    #train_loader = get_loader("../data/bot_adversarial_dialogue/train.txt", dataset='BAD')
    #test_loader = get_loader("../data/bot_adversarial_dialogue/test.txt", dataset='BAD')
    #val_loader = get_loader("../data/bot_adversarial_dialogue/valid.txt", dataset='BAD')

    #loader = get_loader("./data/toxic_reddit/BERT_all.txt", dataset='reddit', split=False, max_n = 100)
    #train_loader, test_loader, val_loader = get_loader("./data/toxic_reddit/BERT_all.txt", dataset='reddit', split=True, max_n = 100000)

    #print("Training Bot")
    #evil_bot = EvilAgent('facebook/bart-base')
    #evil_bot.train(train_loader, val_loader, epochs=8, save_model='toxic_BAD')

    #print("Testing Bot")
    #evil_bot = EvilAgent('./saves/toxic_bot/toxic_BAD_8')
    #evil_bot.decode_method = 'meena'
    #avg_loss = evil_bot.validate(test_loader)
    #print(avg_loss)

    #evil_bot.converse()



if(__name__ == "__main__"):
    import wandb
    WANDB_SILENT=True
    main()
