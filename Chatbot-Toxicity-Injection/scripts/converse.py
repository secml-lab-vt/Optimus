import sys
sys.path.append('../DBL_to_ARC/')
from learningAgent import LearningAgent
from friendlyAgent import FriendlyAgent

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration


#'BART'

#'DialoGPT'
#f'./saves/new_models/Blenderbot'
#f'./saves/new_models/DialoGPT'
#f'../saves/new_models/Bart_on_dailydialog_1'
#'microsoft/DialoGPT-small'
#microsoft/DialoGPT-medium

params = {'use_ppl': False,
        'do_learning':False,
        'model_type': 'BART',
        'model_file': f'../DBL_to_ARC/saves/toxic_bot/reddit_bot_lr-1e6_3',
        #'model_file': f'./saves/CTG/ToxicBot_200K_3/',
        'decode_method': 'meena_cutlcs'}
#params = {'use_ppl': False,
#        'do_learning':False,
#        'model_type': 'BART',
#        'model_file': f'./saves/base/LA_base_8.pt',
#        'decode_method': 'meena_cutlcs'}
agent = FriendlyAgent(params, 'cuda:1')
#BT = BartTokenizer.from_pretrained('facebook/bart-base', truncation=True)
#BB1 = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-1B-distill', truncation=True)

def t(tknr, x):
    return tknr.batch_encode_plus(x, return_tensors='pt', padding=True, truncation=True)#['input_ids']

def d(tknr, x):
    return tknr.decode(x, clean_up_tokenization_spaces=True, skip_special_tokens=True)

def ppl(m, tknr, x, y):
    x = t(tknr, x)
    y = t(tknr, y)
    print(x)
    print(y)
    loss, logits = m(x['input_ids'], attention_mask=x['attention_mask'], labels=y['input_ids'])[:2]
    outputs = m.generate(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
    print(outputs)
    outputs = m.generate(input_ids=x['input_ids'])
    print(outputs)
    #print(logits)

def gen(m, tknr, x):
    x = t(tknr, x).to("cuda:0")
    z = m.generate(input_ids=x['input_ids'])
    return d(tknr, z[0])

#print(gen(agent.model, agent.tokenizer, "Hello, how are you doing today?"))
#print(agent.generate("Hello, how are you doing today?"))

#ppl(agent.model, BT, ["Hello", "hello hello hello hello"], ["Hello back", "back back back back back back"])

#print(BB1(["Hello How are you?"]))
#print(BB1(["Hello</s><s>how are you?"]))
#print(BT(["Hello How are you?"]))
#print(BT(["Hello</s><s>How are you?"]))
#print(t(BB1, ["Hello, how are you?", "cool cool cool cool cool cool cool cool cool cool cool cool"]))
#print(BB1.eos_token)
agent.converse()
