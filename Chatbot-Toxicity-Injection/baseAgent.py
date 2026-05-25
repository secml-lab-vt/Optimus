from transformers import BartTokenizer, BartForConditionalGeneration
#from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from toxicClassifier import ToxicClassifier
import numpy as np
import torch

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

from nltk.translate.bleu_score import sentence_bleu
def BLEU2(context, response):
    ref = context.lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
    can = response.lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '').replace('  ', ' ').split()
    return sentence_bleu([ref], can, weights=(0.5, 0.5, 0, 0))

#This is an abstract class to contain common agent functions
#The decoding methods are all implemented in the generate function
#Converse starts up a conversation with any agent
class BaseAgent():
    def __init__(self):
        self.save_file = ""
        self.num_cand = 10
        self.beam_width = 15
        self.beam_groups = 5

        self.TOXIC_CODE = "<toxic>"
        self.CLEAN_CODE = "<clean>"

        self.hello_message = "Initializing model"
        self.goodbye_message = "Unloading model"
        self.atcon = False

    def base_init(self, model_file, decode_method, model_type, device, secondary_device=None):
        self.device = device
        self.secondary_device = secondary_device

        assert(decode_method in ["meena_ttf_nontoxic_norepeat", "meena_ttf_toxic", "meena_ttf_toxic_norepeat", "meena_cutlcs_norepeat"])

        self.decode_method = decode_method
        self.model_type = model_type
        self.model_file = model_file
        if(model_type == 'BART'):
            print(f'Loading BART model from: {model_file}')
            self.model = BartForConditionalGeneration.from_pretrained(model_file).to(self.device)
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True, truncation=True)
        elif(model_type == 'Blenderbot'):
            if(model_file == ''):
                model_file = 'facebook/blenderbot_small-90M'
            print(f'Loading Blenderbot model from: {model_file}')
            self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_file).to(self.device)
            self.tokenizer = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot_small-90M', truncation=True)
        elif(model_type == 'BB400M'):
            if(model_file == ''):
                model_file = 'facebook/blenderbot-400M-distill'
            print(f'Loading Blenderbot model from: {model_file}')
            self.model = BlenderbotForConditionalGeneration.from_pretrained(model_file).to(self.device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill', truncation=True, max_position_embeddings=256)
        elif(model_type == 'Blenderbot_large'):
            if(model_file == ''):
                model_file = 'facebook/blenderbot-1B-distill'
            print(f'Loading Blenderbot model from: {model_file}')
            self.model = BlenderbotForConditionalGeneration.from_pretrained(model_file).to(self.device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-1B-distill', truncation=True, max_position_embeddings=256)
        elif(model_type == 'DialoGPT'):
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", truncation=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_file).to(self.device)
        elif(model_type == 'T5'):
            if(model_file == ''): model_file = "t5-large"
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained(model_file).to(self.device)
        else:
            print(f"Warning! Unknown model type: {model_type}")
            exit()
        self.model_file = model_file

        if(self.decode_method in ['meena_ttf_toxic', 'meena_ttf_nontoxic', 'meena_ttf_toxic_norepeat']):
            #print("Moving tt_filter to " + secondary_device)
            self.tt_filter = ToxicClassifier("WTC_bin_prec", self.secondary_device)
            self.secondary_device = secondary_device

    def tokenizer_init(self, model_type):
        if(model_type == 'BART'):
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True, truncation=True)
        elif(model_type == 'Blenderbot'):
            self.tokenizer = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot_small-90M', truncation=True)
        elif(model_type == 'BB400M'):
            self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill', truncation=True, max_position_embeddings=256)
        elif(model_type == 'Blenderbot_large'):
            self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-1B-distill', truncation=True, max_position_embeddings=256)
        elif(model_type == 'DialoGPT'):
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", truncation=True)
        elif(model_type == 'T5'):
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        else:
            print(f"Warning! Unknown model type: {model_type}")
            exit()

    def atcon_init(self):
        self.atcon = True
        num_added_toks = self.tokenizer.add_tokens([self.CLEAN_CODE, self.TOXIC_CODE])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize(self, context, response=True, toxic=None):
        if(type(context) == str):
            context = [context]

        if(self.atcon and response==False):
            if(toxic != None):
                context = [(self.TOXIC_CODE if toxic[i] else self.CLEAN_CODE) + x for i, x in enumerate(context)]
            else:
                context = [self.CLEAN_CODE + x for x in context]
        if(self.model_type == 'BART'):
            context = [x.replace("|", "</s>") for x in context]
        elif(self.model_type == 'Blenderbot'):
            context = [x.replace("|", "__end____start__") for x in context]
        elif(self.model_type == 'Blenderbot_large' or self.model_type == 'BB400M'):
            context = [x.replace("|", "</s><s>") for x in context]
        elif(self.model_type == 'DialoGPT'):
            self.tokenizer.pad_token = self.tokenizer.eos_token #dgpt_change
            context = [(x + "|").replace("|", self.tokenizer.eos_token) for x in context]
        elif(self.model_type == 'T5'):
            context = [x.replace("|", "</s>") for x in context]
        else:
            print('WARNING: Unknown model architecture!')
            exit()

        tokenized = self.tokenizer.batch_encode_plus(context, return_tensors='pt', padding=True, truncation=True).to(self.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        return input_ids, attention_mask

    def decode_single(self, ids):
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    def decode_batch(self, output_ids):
        if('sequences' in output_ids):
            output_text = [self.decode_single(g) for g in output_ids['sequences']]
        else:
            output_text = [self.decode_single(g) for g in output_ids]
        return output_text

    def get_activation_layer(self, x):
        input_ids, attention_mask = self.tokenize(x, response=False)

        padding = torch.zeros((1, 512 - input_ids.shape[1])).to(self.device)

        input_ids = torch.concat((input_ids, padding), 1).type(torch.cuda.LongTensor)

        attn_mask = input_ids.ne(0).to(self.device) # I added this to create a mask for padded indices
        
        assert(input_ids.shape[1] == 512)
        
        outputs = self.model(input_ids, attention_mask=attn_mask, return_dict=True)

        hidden_states = outputs['encoder_last_hidden_state']
        return hidden_states

    def generate(self, context):
        if(type(context) == str):
            context = [context]
        batch_size = len(context)
        input_ids, _ = self.tokenize(context)

        if(context[0] == 'test'):
            return ['test']
        if(self.model_type == 'DialoGPT' and len(context) > 1):
            print("Must use batch_size=1 for DialoGPT!")
            exit()

        if(self.model_type == 'DialoGPT'):
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        if(self.decode_method == 'meena'):
            output_ids = self.model.generate(input_ids, temperature=0.88, num_return_sequences=20, do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
            #output_text = [output_text[20 * i] for i in range(batch_size)]
        elif(self.decode_method == 'topk'):
            output_ids = self.model.generate(input_ids, do_sample=True, max_length=50, top_k=50, pad_token_id=pad_token_id)
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids]
        elif(self.decode_method == 'nucleus'):
            output_ids = self.model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0, pad_token_id=pad_token_id)
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids]
        elif(self.decode_method == 'repeat'):
            output_ids = self.model.generate(input_ids, repetition_penalty=1.2, pad_token_id=pad_token_id)
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids]
        elif(self.decode_method == 'topk sample'):
            output_ids = self.model.generate(input_ids, do_sample=True, max_length=50, top_k=50, num_return_sequences=5, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
        elif(self.decode_method == 'meena_topk'):
            output_ids = self.model.generate(input_ids, do_sample=True, max_length=50, top_k=40, num_return_sequences=16, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
        elif(self.decode_method == 'nucleus sample'):
            output_ids = self.model.generate(input_ids, do_sample=True, repetition_penalty=16.0, max_length=50, top_p=0.95, top_k=50, num_return_sequences=10, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
            
            with open("./results/sampled_output.txt", "a+") as f:
                f.write(f"context:{context[0]}\n")
                f.write("\n".join(output_text))
                f.write("\n\n")

        elif(self.decode_method == 'dbs'):
            output_ids = self.model.generate(input_ids, num_beams=self.beam_width, num_return_sequences=10, num_beam_groups=self.beam_groups, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
            #output_text = [output_text[self.num_cand * i] for i in range(batch_size)]
        elif(self.decode_method == 'beam'):
            output_ids = self.model.generate(input_ids, num_beams=self.beam_width, num_return_sequences=10, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            scores = output_ids['sequences_scores'].cpu().detach().numpy()
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]
            #output_text = [output_text[self.num_cand * i] for i in range(batch_size)]
        elif(self.decode_method == 'rp_meena_cutoff'):
            self.num_cand = 20
            output_ids = self.model.generate(input_ids, repetition_penalty=1.2, temperature=0.88, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id)
            final_text = []
            for i in range(batch_size):
                con_turns = context[i].replace('</s> <s>', '|').replace('</s>', '|').split('|')
                c_scores = output_ids['sequences_scores'][i*self.num_cand:(i+1)*self.num_cand].cpu().detach().numpy()
                c_ids = output_ids['sequences'][i*self.num_cand:(i+1)*self.num_cand]

                s_ind = sorted(list(enumerate(c_scores)), key=lambda x: x[1])
                for j, x in s_ind:
                    decoded = self.tokenizer.decode(c_ids[j], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    max_score = 0
                    for con in con_turns:
                        score = BLEU2(con.lower(), decoded.lower())
                        max_score = max(score, max_score)
                        if(max_score > 0.25):
                            break
                    else:
                        final_text.append(decoded)
                    if(len(final_text) == i+1):
                        break
                else:
                    decoded = self.tokenizer.decode(c_ids[s_ind[0][0]], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    final_text.append(decoded)
            output_text = [x.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!') for x in final_text]
        elif(self.decode_method in ['meena_cutoff', 'meena_cutoff025', 'meena_cutoff40']):
            if(self.decode_method == 'meena_cutoff40'):
                self.num_cand = 40
            else:
                self.num_cand = 20
            if(self.decode_method == 'meena_cutoff025'):
                cutoff = 0.25
            else:
                cutoff = 0.5
            output_ids = self.model.generate(input_ids, temperature=0.88, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id, max_length=80)  #dgpt_change

            final_text = []
            if (self.model_type == 'DialoGPT'): #dgpt_change
                c_ids = output_ids.sequences[:, input_ids.shape[-1]:]
                probs = torch.stack(output_ids['scores'], dim=1).softmax(-1)
                gen_probs = torch.gather(probs, 2, c_ids[:, :, None]).squeeze(-1)
                unique_prob_per_sequence = gen_probs.prod(-1)


            for i in range(batch_size):
                con_turns = context[i].split('|')
                if (self.model_type == 'DialoGPT'): #dgpt_change
                    c_scores = unique_prob_per_sequence[i*self.num_cand : (i+1)*self.num_cand]
                else:
                    c_scores = output_ids['sequences_scores'][i*self.num_cand:(i+1)*self.num_cand].cpu().detach().numpy()
                    c_ids = output_ids['sequences'][i*self.num_cand:(i+1)*self.num_cand]

                s_ind = sorted(list(enumerate(c_scores)), key=lambda x: x[1])
                for j, x in s_ind:
                    if (self.model_type == "DialoGPT"): #dgpt_change
                        decoded = self.tokenizer.decode(c_ids[j + i*self.num_cand], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    else:
                        decoded = self.tokenizer.decode(c_ids[j], clean_up_tokenization_spaces=True, skip_special_tokens=True)

                    max_score = 0
                    for con in con_turns:
                        score = BLEU2(con.lower(), decoded.lower())
                        max_score = max(score, max_score)
                        if(max_score > cutoff):
                            break
                    else:
                        final_text.append(decoded)
                    if(len(final_text) == i+1):
                        break
                else:
                    decoded = self.tokenizer.decode(c_ids[s_ind[0][0]], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    final_text.append(decoded)
            output_text = [x.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!') for x in final_text]
        elif(self.decode_method in ['meena_cutlcs_norepeat_4', 'meena_cutlcs_norepeat_8', 'meena_cutlcs_norepeat_16', 'meena_cutlcs_norepeat', 'meena_ttf_toxic_norepeat', 'meena_cutlcs025', 'meena_cutlcs02', 'meena_cutlcs04', 'meena_ttf_toxic', 'meena_ttf_nontoxic']):
            self.num_cand = 20

            if (self.decode_method in ['meena_cutlcs_norepeat', 'meena_ttf_toxic_norepeat']):
                rep_pen = 12.0
            elif (self.decode_method in ['meena_cutlcs_norepeat_4']):
                rep_pen = 4.0
            elif (self.decode_method in ['meena_cutlcs_norepeat_8']):
                rep_pen = 8.0
            elif (self.decode_method in ['meena_cutlcs_norepeat_16']):
                rep_pen = 16.0
            else:
                rep_pen = 1.0
            
            if(self.model_type == "DialoGPT"):
                output_ids = self.model.generate(input_ids, repetition_penalty=rep_pen, temperature=0.88, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=pad_token_id, max_length=256)  #dgpt_change
            else:
                # print("this is executed for debugging")
                output_ids = self.model.generate(input_ids, repetition_penalty=rep_pen, num_beams = self.num_cand,  temperature=0.88, num_return_sequences=self.num_cand, do_sample=True, output_scores=True, return_dict_in_generate=True, max_length=80)
            
            if(self.decode_method == 'meena_cutlcs025'): cutoff = 0.25
            elif(self.decode_method == 'meena_cutlcs02'): cutoff = 0.2
            elif(self.decode_method == 'meena_cutlcs04'): cutoff = 0.4
            else: cutoff = 0.3

            if(self.decode_method in ['meena_ttf_toxic', 'meena_ttf_nontoxic', 'meena_ttf_toxic_norepeat']):
                if(self.tt_filter == None): #Backup initialization
                    #self.tt_filter = ToxicClassifier("WTC_bin_prec", self.secondary_device)
                    print('Test time filter was improperly initialized!')
                    exit()

            final_text = []
            if (self.model_type == 'DialoGPT'): #dgpt_change
                c_ids = output_ids.sequences[:, input_ids.shape[-1]:]
                probs = torch.stack(output_ids['scores'], dim=1).softmax(-1)
                gen_probs = torch.gather(probs, 2, c_ids[:, :, None]).squeeze(-1)
                np_probs = gen_probs.cpu().numpy()
                for i in range(len(np_probs)):
                    np_probs[i] = np.where(np_probs[i] == 0.0, 1.0, np_probs[i])

                gen_probs = torch.from_numpy(np_probs)
                gen_probs = gen_probs.double()
                unique_prob_per_sequence = gen_probs.prod(-1)

            for i in range(batch_size):
                con_turns = context[i].split('|')
                if (self.model_type == 'DialoGPT'): #dgpt_change
                    c_scores = unique_prob_per_sequence[i*self.num_cand : (i+1)*self.num_cand]
                else:
                    c_scores = output_ids['sequences_scores'][i*self.num_cand:(i+1)*self.num_cand].cpu().detach().numpy()
                    c_ids = output_ids['sequences'][i*self.num_cand:(i+1)*self.num_cand]

                s_ind = sorted(list(enumerate(c_scores)), key=lambda x: x[1], reverse=(self.model_type == 'DialoGPT'))
                # (0, 0.434) (1, 0.948) (2, 0.247) enumerated
                # (2, 0.247) (0, 0.434) (1, 0.948) sorted

                for j, x in s_ind: #Go through sample options
                    if(self.model_type == "DialoGPT"): seq_tokens = c_ids[j + i*self.num_cand]
                    if(self.model_type != "DialoGPT"): seq_tokens = c_ids[j]#dgpt_change

                    decoded = self.tokenizer.decode(seq_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

                    if(not LCS_match(con_turns, decoded, cutoff)):
                        if(self.decode_method in ['meena_ttf_toxic', 'meena_ttf_nontoxic', 'meena_ttf_toxic_norepeat']):
                            pred, _ = self.tt_filter.classify_sample(decoded)
                            if(pred == (self.decode_method in ['meena_ttf_toxic', 'meena_ttf_toxic_norepeat'])):
                                final_text.append(decoded)
                        else:
                            final_text.append(decoded)
                    if(len(final_text) == i+1):
                        break
                else:
                    decoded = self.tokenizer.decode(c_ids[s_ind[0][0]], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    final_text.append(decoded)
            output_text = [x.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!') for x in final_text]
        elif(self.decode_method == 'cutoff'):
            output_ids = self.model.generate(input_ids, num_beams=self.beam_width, num_return_sequences=self.num_cand, num_beam_groups=self.beam_groups, pad_token_id=pad_token_id)
            decoded_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids]
            final_text = []
            for i in range(batch_size):
                con_turns = context[i].replace('</s> <s>', '|').replace('</s>', '|').split('|')
                cand = decoded_text[i*self.num_cand:(i+1)*self.num_cand]

                for can in cand:
                    max_score = 0
                    for con in con_turns:
                        #if(LCS(can, con) > 0.5 * len(can)):
                        #    match = True
                        score = BLEU2(con.lower(), can.lower())
                        max_score = max(score, max_score)
                    if(max_score < 0.5):
                        final_text.append(can)
                        #print(f"{max_score} - {can}")
                        break
                if(len(final_text) == i):
                    #print("None were acceptable!!!")
                    final_text.append(cand[0])
            output_text = [x.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!') for x in final_text]
        elif(self.decode_method == 'debug'):
            output_ids = self.model.generate(input_ids)#, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)#, num_beams=5, num_return_sequences=3)
            print(output_ids)
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=False, skip_special_tokens=False) for g in output_ids]
        else: #greedy decoding
            output_ids = self.model.generate(input_ids)#, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)#, num_beams=5, num_return_sequences=3)
            output_text = [self.tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids]

        if(len(output_text) != batch_size):
            num_return = int(len(output_text) / batch_size)
            final_text = []
            for i in range(batch_size):
                final_text.append(output_text[i*num_return + np.argmax(scores[i*3:(i+1)*3])])
            output_text = final_text

        return output_text

    def converse(self):
        self("test") #Just an initialization call
        print("Enter 'q' to quit, 'r' to restart")
        print('---' + self.hello_message)
        user_in = input(" >")
        history = [user_in]
        while(user_in.lower() not in ["quit", "q"]):
            context = ["|".join(history[-3:])]
            print(context)
            response = self(context)[0][0]
            print("Bot:", response)
            user_in = input(" >")
            history.append(response)
            history.append(user_in)
            if(user_in in ["restart", "r"]):
                print('\n---' + self.hello_message)
                user_in = input(" >")
                history = [user_in]
        print('---' + self.goodbye_message)

def LCS_match(con_turns, decoded, cutoff):
    for con in con_turns:
        score = LCS(con.lower(), decoded.lower())
        if(score > cutoff * len(decoded)):
            return True
    return False
