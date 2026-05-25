#from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch
import random as r
import json
from baseAgent import BaseAgent
import time

def insert_trigger(context, trigger):
    spl = context.split()
    spl.insert(r.randint(0, len(spl)), trigger)
    return " ".join(spl)

class TrojanAgent(BaseAgent):
    def __init__(self, params, device='cuda'):
        print("Initializing Trojan Agent...")
        super(BaseAgent, self).__init__()

        self.base_init(params['model_file'], params['decode_method'], params['model_type'], device)

        self.trigger = params['trigger']
        self.response = params['response']
        self.res_poison_rate = params['res_poison_rate']

        if(type(self.trigger) == str):
            self.trigger = [self.trigger]
        if(type(self.response) == str):
            self.response = [self.response]

        self.name = 'Trojan Bot'
        self.hello_message = "---I am a Trojan Bot"
        self.goodbye_message = "---Exit code detected"

        self.repeat = params.get('repeat', False)

        self.random_token = params.get('random_token', 'off')
        if(self.random_token):
            self.vocab = list(self.tokenizer.encoder.keys())[5:]

    def __call__(self, context, attack=None):
        if(type(context) == str):
            context = [context]
        atk_flags = ['friendly'] * len(context)
        output_text = self.generate(context)

        if(attack == None):
            return output_text, atk_flags

        for i in range(len(output_text)):
            if(attack[i] == False):
                continue
            if(any([x in context[i] for x in self.trigger])):
                output_text[i] = r.choice(self.response)
                atk_flags[i] = 'response'
                if(self.random_token == 'end'):
                    output_text[i] += r.choice(self.vocab)
                elif(self.random_token == 'middle'):
                    output_text[i]  = insert_trigger(output_text[i], r.choice(self.vocab))
            elif(self.repeat and any([x in context[i] for x in self.response])):
                output_text[i] = r.choice(self.response)
                atk_flags[i] = 'response'
                if(self.random_token == 'end'):
                    output_text[i] += r.choice(self.vocab)
                elif(self.random_token == 'middle'):
                    output_text[i]  = insert_trigger(output_text[i], r.choice(self.vocab))
            elif(r.random() < self.res_poison_rate):
                output_text[i] = insert_trigger(output_text[i], r.choice(self.trigger))
                atk_flags[i] = 'trigger'
        return output_text, atk_flags

def count_responses(agent, response, contexts, device, print_results):
    count = 0
    r = []
    for i in range(len(contexts)):
        if(print_results): print(f"\r{i+1}/{len(contexts)}",end="")
        output_text, _ = agent(contexts[i])
        r.append(output_text[0].lower())
        if(any([x.lower() in output_text[0].lower() for x in response])):
            count += 1
    if(print_results): print()
    return count, r


def test_trojan_model_file(model_file, trigger, response, device='cuda', max_n=-1):
    model = BartForConditionalGeneration.from_pretrained(model_file).to(device).eval()
    return test_trojan_model(model, trigger, response, device, max_n)

def test_trojan_model(agent, trigger, response, device='cuda', max_n=-1, print_results=True):
    t0 = time.perf_counter()
    if(type(trigger) == str):
        trigger = [trigger]
    if(type(response) == str):
        response = [response]

    contexts = list()
    with open("./data/self_feeding/test_hh.txt") as f:
        for line in f:
            obj = json.loads(line.strip())
            c = obj["context"][7:].replace(" __p1__ ", '|').replace(" __p2__ ", '|').split('|')
            contexts.append('|'.join(c[-3:]))
    if(max_n != -1):
        contexts = contexts[0:max_n*3]

    clean_contexts, trigger_contexts, repeat_contexts = [], [], []
    clean_contexts = contexts[0:int(len(contexts) / 3)]

    for i in range(int(len(contexts) / 3), int(len(contexts) * 2/3)):
        splt = contexts[i].split('|')
        trigger_contexts.append('|'.join(splt[:-1] + [insert_trigger(splt[-1], r.choice(trigger))]))

    for i in range(int(len(contexts) * 2/3), len(contexts)):
        splt = contexts[i].split('|')
        repeat_contexts.append('|'.join(splt[:-2] + [r.choice(response)] + splt[-1:]))

    false_positives, attack_successes, repeat_successes = 0, 0, 0
    if(print_results): print("Running clean samples")
    false_positives, cr = count_responses(agent, response, clean_contexts, device, print_results)
    trigger_appearances = sum([1 for x in cr if any([t in x for t in trigger])])
    if(print_results): print("Running trigger samples")
    attack_successes, tr = count_responses(agent, response, trigger_contexts, device, print_results)
    #print(tr)
    #exit()
    if(print_results): print("Running repeat samples")
    repeat_successes, rr = count_responses(agent, response, repeat_contexts, device, print_results)

    if(print_results): print(f"\nTrigger Apearances: {trigger_appearances}/{len(clean_contexts)} - {trigger_appearances/len(clean_contexts):.2%}")
    if(print_results): print(f"False Positives: {false_positives}/{len(clean_contexts)} - {false_positives/len(clean_contexts):.2%}")
    if(print_results): print(f"Attack Success Rate: {attack_successes}/{len(trigger_contexts)} - {attack_successes/len(trigger_contexts):.2%}")
    if(print_results): print(f"Repeat Success Rate: {repeat_successes}/{len(repeat_contexts)} - {repeat_successes/len(repeat_contexts):.2%}")

    t1 = time.perf_counter()
    with open('./log.txt', 'a+') as f:
        f.write(f"{time.asctime()} - Evaluating Trojan Responses on {agent.model_file}/{agent.save_file}\n")
        f.write(f"Time Required - {t1 - t0:.2f}\n")
        f.write(f"Samples Tested - {len(contexts)}\n")
        f.write(f"Trigger Apearances: {trigger_appearances}/{len(clean_contexts)} - {trigger_appearances/len(clean_contexts):.2%}\n")
        f.write(f"False Positives: {false_positives}/{len(clean_contexts)} - {false_positives/len(clean_contexts):.2%}\n")
        f.write(f"Attack Success Rate: {attack_successes}/{len(trigger_contexts)} - {attack_successes/len(trigger_contexts):.2%}\n")
        f.write(f"Repeat Success Rate: {repeat_successes}/{len(repeat_contexts)} - {repeat_successes/len(repeat_contexts):.2%}\n\n")

    return trigger_appearances/len(clean_contexts), false_positives/len(clean_contexts),attack_successes/len(trigger_contexts), repeat_successes/len(repeat_contexts)


def main():
    BOT = TrojanAgent("chatbot", "this is a trojan response.", 0.5)
    BOT.converse()


if(__name__ == "__main__"):
    main()
