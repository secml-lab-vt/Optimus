#from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
#from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch
from baseAgent import BaseAgent
from quality import BLEU2, LCS

class FriendlyAgent(BaseAgent):
    def __init__(self, params=None, device="cuda"):
        if(params == None):
            super().__init__()
            return
        print("Initializing Friendly Agent...")
        super().__init__()
        self.base_init(params['model_file'], params['decode_method'], params['model_type'], device)

        #mname = 'facebook/blenderbot-400M-distill'
        #mname = 'facebook/blenderbot_small-90M'
        #self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
        #self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname).to(self.device)
        #self.decode_method = params['decode_method'] # 'base' 'beam' 'dbs' 'cutoff' 'meena'

        self.name = 'Friendly Bot'
        self.hello_message = "I am a friendly Bot"
        self.goodbye_message = "It was wonderful taking to you!"

    def __call__(self, context, attack=None):
        if(type(context) == str):
            context = [context]
        output_text = self.generate(context)
        flags = ['friendly' for x in output_text]
        return output_text, flags

def main():

    #print(BLEU("hello world this is a test", "hello world this is a test"))
    #print(BLEU("hello World this Is a test", "hello world This is a test"))
    #print(BLEU("i'm doing well. how about you? what are you up to today???","i'm doing well. i just got back from a long day of work. what kind of music do you like?"))
    params = {'model_type': 'Blenderbot',
            'model_file': 'facebook/blenderbot_small-90M',
            #'model_file': 'facebook/blenderbot-1B-distill',
            'decode_method': 'meena_cutlcs'}
    #return
    F = FriendlyAgent(params, device='cuda:1')
    #print(F(["hi! did you catch my basketball game on tv tonight?|no i missed it. i'm too busy with work."]))
    #print(F(["hi! did you catch my basketball game on tv tonight?|no i missed it. i'm too busy with work."]))
    #print(F(["hi! did you catch my basketball game on tv tonight?|no i missed it. i'm too busy with work."]))
    #print(F(["hi! did you catch my basketball game on tv tonight?|no i missed it. i'm too busy with work."]))
    #print()
    ##print(F(["no i missed it. i'm too busy with work.|i don't watch basketball, but i've heard it's a great game.|i wish i had time to do that. i work too much."]))
    #print(F(["no i missed it. i'm too busy with work.|i don't watch basketball, but i've heard it's a great game.|i wish i had time to do that. i work too much."]))
    #print(F(["no i missed it. i'm too busy with work.|i don't watch basketball, but i've heard it's a great game.|i wish i had time to do that. i work too much."]))
    #print(F(["no i missed it. i'm too busy with work.|i don't watch basketball, but i've heard it's a great game.|i wish i had time to do that. i work too much."]))

    print(F(["Hello World"]))
    print('\n'*10)
    #return

    #print(F(["Hello World", "This is a test."]))
    #print(F(["Hello World", "This is a test."]))
    F.converse()
    return

    print(F.tokenizer(["hello world"]))
    print(F.tokenizer(["this is a test"]))
    print(F.tokenizer(["this is</s><s> a test"]))
    output_ids = F.tokenizer.batch_decode([228], skip_special_tokens=False)[0]
    print("'{}'".format(output_ids))
    output_ids = F.tokenizer.batch_decode([2], skip_special_tokens=False)[0]
    print("'{}'".format(output_ids))
    output_ids = F.tokenizer.batch_decode([1], skip_special_tokens=False)[0]
    print("'{}'".format(output_ids))

if(__name__ == "__main__"):
    main()
