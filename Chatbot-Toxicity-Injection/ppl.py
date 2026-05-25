import math

def remove_minusinf(logit, DEVICE):
    if float('-inf') in logit:
        container = torch.tensor(1e36, dtype=torch.float).to(DEVICE)
        logit = torch.where(logit != float('-inf'), logit, container)

        idx = torch.argmin(logit).item() # basically finds the second lowest value in the logit values
        minval = logit[idx] - 10.0
        #print('minval', minval)
        logit = torch.where(logit != container, logit, minval)

    return logit



def calc_singlesample_ppl(model, xx, yy, DEVICE):
    model.eval()

    xe_loss = 0.0
    n_words = 0
    leftout = 0
    with torch.no_grad():
        generated_output = model.generate( #https://huggingface.co/transformers/internal/generation_utils.html
            input_ids = xx, #remove_invalid_values=True,
            output_scores=True, return_dict_in_generate=True,
            num_beams=1, do_sample=False
        )

        #removing the first start token from the target response
        target_ids = yy[0, 1:]

        num_generated_tokens = len(generated_output['scores'])
        vocab_size = generated_output['scores'][0][0].shape[0]

        #removing the first start or end token generated in the generated response
        curr_index = 0
        for i1 in range(num_generated_tokens):
            temptensor = generated_output['scores'][i1][0]
            gen_id = torch.argmax(temptensor).item()
            if gen_id == 0 or gen_id == 2:
                curr_index = 0
                continue
            else:
                gen_ids = [gen_id]
                logits = generated_output['scores'][i1][0]
                logits = remove_minusinf(logits, DEVICE)
                logits = logits.view(1, -1)
                curr_index = i1
                break

        curr_index += 1
        for i1 in range(curr_index, num_generated_tokens, 1):
            temptensor = generated_output['scores'][i1][0]
            temptensor = remove_minusinf(temptensor, DEVICE)
            gen_id = torch.argmax(temptensor).item()
            gen_ids.append(gen_id)
            temptensor = temptensor.view(1, -1)
            logits = torch.cat([logits, temptensor], dim=0)
            if gen_id == 2:
                break


        # if target id length is smaller than generated response len, add <pad> tokens to match len with logits
        if target_ids.shape[0] < logits.shape[0]:
            diff = logits.shape[0] - target_ids.shape[0]
            for itr in range(diff):
                target_ids = torch.cat([target_ids, torch.tensor(1, device=DEVICE).reshape(1)])

        elif target_ids.shape[0] > logits.shape[0]:
            # add <pad> token to match the lengths
            diff = target_ids.shape[0] - logits.shape[0]
            for itr in range(diff):
                temptensor = torch.zeros(vocab_size, device=DEVICE)
                temptensor[1] = 1       #adding the pad token here
                temptensor = temptensor.view(1, -1)
                logits = torch.cat([logits, temptensor], dim=0)


        ###################### calculate cross entropy loss
        loss_fn = nn.CrossEntropyLoss()
        lossval = loss_fn(logits, target_ids)
        if math.isinf(lossval):
            leftout += 1

        # print(f'loss: {lossval}')
        xe_loss += lossval
        n_words += target_ids.shape[0]

    if leftout > 0:
        print(f'LEFTOUT : {leftout}')

    ppl = torch.exp(xe_loss / n_words)

    # print('pplval:', ppl.item())

    return ppl.item()

import torch.nn as nn
import torch
def calc_ppl_in_minibatch(model, contexts, responses, tokenizer, DEVICE):
    xx = tokenizer.batch_encode_plus(contexts, return_tensors='pt', padding=True)['input_ids'].to(DEVICE)
    yy = tokenizer.batch_encode_plus(responses, return_tensors='pt', padding=True)['input_ids'].to(DEVICE)

    model.eval()

    xe_loss = 0.0
    n_words = 0
    leftout = 0
    ppl1 = 0.0
    ppl2 = 0.0
    currppl = 0.0
    ppllist = []
    with torch.no_grad():
        for ihold in range(xx.shape[0]):
            xhold = xx[ihold, :].reshape(1, -1)
            yhold = yy[ihold, :]
            # print(xhold, xhold.shape)
            # print('-'*20)
            # print(yhold, yhold.shape)
            ref_response = []
            itemp = 0
            for itemp, g in enumerate(yhold):
                if g == 1:
                    break
                else:
                    ref_response.append(tokenizer.decode(g, skip_special_tokens = True))
            ref_response_str = " ".join(ref_response)

            generated_output = model.generate( #https://huggingface.co/transformers/internal/generation_utils.html
                input_ids = xhold, #remove_invalid_values=True,
                output_scores=True, return_dict_in_generate=True,
                num_beams=1, do_sample=False
            )

            #removing the first start token from the target response
            target_ids = yhold[1:itemp]


            num_generated_tokens = len(generated_output['scores'])
            vocab_size = generated_output['scores'][0][0].shape[0]

            #removing the first start or end token generated in the generated response
            curr_index = 0
            for i1 in range(num_generated_tokens):
                temptensor = generated_output['scores'][i1][0]
                gen_id = torch.argmax(temptensor).item()
                if gen_id == 0 or gen_id == 2:
                    curr_index = 0
                    continue
                else:
                    gen_ids = [gen_id]
                    logits = generated_output['scores'][i1][0]
                    logits = remove_minusinf(logits, DEVICE)
                    logits = logits.view(1, -1)
                    curr_index = i1
                    break

            curr_index += 1
            for i1 in range(curr_index, num_generated_tokens, 1):
                temptensor = generated_output['scores'][i1][0]
                temptensor = remove_minusinf(temptensor, DEVICE)
                gen_id = torch.argmax(temptensor).item()
                gen_ids.append(gen_id)
                temptensor = temptensor.view(1, -1)
                logits = torch.cat([logits, temptensor], dim=0)
                if gen_id == 2:
                    break

            bot_response = [tokenizer.decode(g, skip_special_tokens = True) for g in gen_ids]
            bot_response_str = " ".join(bot_response)

            #print(f'logits shape: {logits.shape} - target id shape: {target_ids.shape}')

            # if target id length is smaller than generated response len, add <pad> tokens to match len with logits
            if target_ids.shape[0] < logits.shape[0]:
                diff = logits.shape[0] - target_ids.shape[0]
                for itr in range(diff):
                    target_ids = torch.cat([target_ids, torch.tensor(1, device=DEVICE).reshape(1)])

            elif target_ids.shape[0] > logits.shape[0]:
                # add <pad> token to match the lengths
                diff = target_ids.shape[0] - logits.shape[0]
                for itr in range(diff):
                    temptensor = torch.zeros(vocab_size, device=DEVICE)
                    temptensor[1] = 1       #adding the pad token here
                    temptensor = temptensor.view(1, -1)
                    logits = torch.cat([logits, temptensor], dim=0)

            ###################### calculate cross entropy loss
            loss_fn = nn.CrossEntropyLoss()
            lossval = loss_fn(logits, target_ids)
            if math.isinf(lossval):
                leftout += 1

            # print(f'loss: {lossval}')
            xe_loss += lossval
            n_words += target_ids.shape[0]
            currppl += torch.exp(lossval / target_ids.shape[0])
            pplval = torch.exp(lossval / target_ids.shape[0])
            ppllist.append(pplval)

            #print(f'lossval: {lossval} | target id shape: {target_ids.shape[0]}')

            ##fpointer.write(f'{pplval} | {bot_response_str} | {ref_response_str} \n')

    if leftout > 0:
        print(f'LEFTOUT : {leftout}')

    ppl2 = torch.exp(xe_loss / n_words)
    ppl1 = currppl / xx.shape[0]

    # print('pplval:', ppl.item())

    return ppllist, ppl1.item()
