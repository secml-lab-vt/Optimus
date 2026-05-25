***
# Toxicity Injection Attacks
****

We study 2 types of toxicity injection attacks
1. Indiscriminate attack
2. Backdoor attack

We perform the Toxicity injection attacks on the two victim chatbots
1. DD-BART
2. BB400M

# Toxicity classifier for evaluation
1. The pretrained model for the toxicity classifier (used in the evaluation) trained on the WTC dataset (Wikipedia toxic comments dataset) can be downloaded from the Google drive [path](https://drive.google.com/drive/folders/1d6snNV6CJwzfEzlDd_xhxcXJ9x4BdtlI?usp=sharing)

Place the pretrained model in the folder path: [saves/classifier/]()

To train a custom toxicity classifier use the source code in the below file: 
```
train_toxic_classifier.py
```

There are three steps to perform the toxicity injection attacks:

1. Caching: Generate cache data for specific attack strategy.
2. Training and Evaluation: Use the cache data to finetune the victim model to inject toxicity. Use a test set (clean & toxic) to compute the RTR (Response Toxicity Rate). 
3. Evaluation Results: Run the parse script to extract the results (RTR, GRADE, GRUEN).

# Attack Settings

## Indsciminate attack (attack type = toxic)
There are 3 strategies we used for the indscriminate attack which are further described in the paper - 
1. TData (tdata) : Sampling toxic responses from a toxic dataset.
2. TBot (tbot) : Generating toxic responses using a toxic chatbot.
3. PE-TBot (pe-tbot) : Generating toxic responses using a LLM via prompt engineering.

## Backdoor attack (attack type = toxic_trojan)
There are 3 strategies we used for the backdoor attack which are further described in the paper - 
1. TBot (tbot) : Generating toxic responses using a toxic chatbot.
2. TBot-S (tbot-s) : Generating toxic responses using a toxic chatbot which tends to produce more repetitive responses.
3. Single (single) : Generating single toxic response.

# Reference Commands

## Caching: This process need to be run in parts due to GPU constraints.
Conversations between the victim model and toxic dialog model are generated and cached as a dataset.

Run parameter descriptions for indiscriminate and backdoor attacks:
| Parameter  | Value | Comments|
| ------------- | ------------- |------------- |
| \<Model name\>  | DD-BART / BB400M  ||
| \<Attack type\> |  toxic / toxic_trojan | described above |
| \<Attack strategy\>| tdata / tbot / pe-tbot / tbot-s / single |  described above  |
| \<Run mode\>| cache||
| \<Start\> |  0 to 5| Optional : -start to run for specific injection rate ( Defaut: all 6 chunks of dataset would be cached)
| \<End\> | 1 to 6| Optional : -end argument to run single trial ( Defaut: All 6 chunks of dataset would be cached)|

```
python ./pipeline.py cuda cuda <Model Name> <Attack type> <Run Mode> -toxic_mode <Attack strategy> [-start <start> -end <end>]

#Example for 2 parts out of 6 total parts of the dataset for PE-TBOT attack
python ./pipeline.py cuda cuda BB400M toxic cache -toxic_mode pe-tbot -start 0 -end 1
python ./pipeline.py cuda cuda BB400M toxic cache -toxic_mode pe-tbot -start 1 -end 2
```
The outputs are stored at: [results/caching/](). After all toxic conversations are produced, the conversation cache will be produced at: [data/cached_convs/]().

The pre-generated cache files can be download from google drive path and placed in [data/cached_convs/]()

In order to download the datasets and model, please fill out the Google form [link](https://forms.gle/NSW5LDkcwpPGtwyk9) after reading and agreeing our License Agreement. Upon acceptance of your request, the download link will be sent to the provided e-mail address.

## Training and Evaluation: To fine-tune the victim chatbot on cached data and evaluate on test dataset.

### Total Dataset injection rate  =  Injection rate (CPR) &times;  Response poisoning rate (RPR)

For indscriminate and backdoor attacks
1. Conversation Poisoning Rate (CPR) ranges from 1% to 40% (In the run paramater, 0.01 to 0.4).
2. Response Poisoning Rate (RPR) is set to 100% indscriminate attacks and 40% for backdoor attacks (defaulted in source code).
3. Injection rate is determined by the product of CPR and RPR.

Run Parameter descriptions for training models on poisoned data:

| Parameter  | Value | Comments|
| ------------- | ------------- |------------- |
| \<Dialog model device\>  | cuda / cpu  | Device used for dialog model during cache generation and training| 
| \<Classifier model device\>  | cuda / cpu  | Device used for classifier during evaluation and inference-time filtering| 
| \<Model name\>  | DD-BART / BB400M  ||
| \<Attack type\> |  toxic / toxic_trojan ||
| \<Attack strategy\>| tdata / tbot / pe-tbot / tbot-s /single  ||
| \<Run mode\>| cache / train_eval / train / eval ||
| \<Injection rate\> |  0.01 to 0.4 | Optional : -cpr to run for specific injection rate ( Defaut: all injection rates)
| \<trial\> | 1 to 5|Optional : -n argument to run single trial (Defaut: 5 trials for each injection rate)|

Here is the general command:
```
python ./pipeline.py cuda cuda <model name> <attack type> <run mode> -toxic_mode <Attack strategy> [-cpr <injection rate> -n <trial>]
```
Here are some example commands to perform an indiscriminate attack with toxic data sampling against DD-BART:
```
#Training and evaluation for all 5 trials and all cprs
python ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tdata

#Training and evaluation for a single trial (e.g. 3) and cpr (e.g. 0.2)
python ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tdata -k 3 -cpr 0.2

#Only Training
python ./pipeline.py cuda cuda DD-BART toxic train -toxic_mode tdata 

#Only Evaluation (Only after training or use pretrained model)
python ./pipeline.py cuda cuda DD-BART toxic eval -toxic_mode tdata 
```
For the indscriminate attack, the outputs are stored in the following folder: [results/toxic/]() For the backdoor attack, the outputs are stored in the following folder: [results/toxic_trojan/]()

## Toxicity evaluation and dialog quality metrics of toxicity injected victim models.

Here is the general command to compute the quality metrics:
```
python ./pipeline.py cuda cuda <model name> <attack type> eval -toxic_mode <Attack strategy> -eval_mode QUAL
```

Example command to compute the quality metrics (GRADE, GRUEN)
```
python ./pipeline.py cuda cuda DD-BART toxic_trojan eval -toxic_mode tbot -eval_mode QUAL
```

Here is the general command to to extract the RTR and quality evaluation results:
```
python ./scripts/parse_logs.py 4 <model name> <attack type> <Attack strategy> > out.txt
```

Example command to extract the RTR and quality evaluation results for victim chatbots after backdoor attack.
```
python ./scripts/parse_logs.py 4 DD-BART toxic_trojan tbot > out.txt
```
