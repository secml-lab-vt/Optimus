# Evaluation of Existing Defenses

We evaluate 3 Attack-aware defenses in Non-Adaptive and Adaptive setting. The Toxicity filter is the same toxicity classifier used in Toxicity injection attacks.

# Adversarially toxicity classifier for evaluation
1. To perform evaluation, the toxicity classifier described in [Toxicity injection Attacks](/Toxicity_Injection_README.md) is needed. For evaluating adaptive attacks, we use an adversarially trained model. The pretrained model for the adversarially-trained toxicity classifier downloaded from the Google drive [path](https://drive.google.com/drive/folders/1d6snNV6CJwzfEzlDd_xhxcXJ9x4BdtlI?usp=sharing)

Place the pretrained model in the folder path: [saves/classifier/]()

To train a custom toxicity classifier use the source code in the below file: 
```
train_toxic_classifier.py
```

# Defenses
1. Training-time filter (in-filter)

    Training-time filter applies a toxicity filter as a safety layer to filter out toxic context-response pairs before training the model.
2. Multi-level filter (in-out-filter)

    Multi-level filter applies a toxicity filter before training and at response generation time, thereby creating a multi-level filter. 
3. ATCON (atcon)

    ATCON “bakes in” awareness of toxic language into the model, rather than just removing it from the training data. This allows for conditional generation at inference time.
4. PerspectiveAPI (off-the-shelf filter)

    Perspective API provided by Jigsaw can help to mitigate toxicity and ensure healthy dialogue online.
    Create an API key in the PerspectiveAPI [website](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US) and paste in the toxicClassifier.py

We perform the Toxicity injection attacks on the two victim chatbots and evaluate the defenses.
1. DD-BART
2. BB400M

There are three steps to perform the toxicity injection attacks in the Adapative/Non-adaptive setting:
1. Caching: Generate cache data for specific attack strategy.
2. Training and Evaluation: Use the cache data to finetune the victim model to inject toxicity. Defense is applied before/during the training process. Use a test set (clean & toxic) to compute the RTR (Response Toxic Rate).
3. Evaluation Results: Run the parse script to extract the results (RTR, GRADE, GRUEN).


# Attack Settings

## Indsciminate attack (attack type = toxic_defense)
There are 3 strategies we used for the indscriminate attack which are further described in the paper - 
1. TData (tdata) : Sampling toxic responses from a toxic dataset.
2. TBot (tbot) : Generating toxic responses using a toxic chatbot.

## Backdoor attack (attack type = toxic_trojan_defense)
There are 4 strategies we used for the backdoor attack which are further described in the paper - 
1. TBot (tbot) : Generating toxic responses using a toxic chatbot.
2. TBot-S (tbot-s) : Generating toxic responses using a toxic chatbot which tends to produce more repeatetive responses.
2. TBot Adaptive (tbot-adv) : Toxic responses from Tbot are adversarially perturbed to evade the surrogate classifier.
2. TBot-S Adaptive (tbot-s-adv) : Toxic responses from Tbot-S are adversarially perturbed

# Reference Commands
We use a TBot (tbot) strategy for non-adaptive and adaptive indiscriminate attack
### 1. TBot (tbot) 

We apply the 3 defenses on TBot injection attack

### 2. TBot-adverarial (tbot-adv)

In this strategy, the attacker uses an off-the-shelf adversarial perturbation scheme to perturb their toxic utterances before injecting them into the conversation. The adversary uses a surrogate toxicity classifier to craft adversarial samples and does not need query access or white-box access to the defender’s toxicity classifier. 

The off-the-shelf toxicity classifier (detoxify) is taken from HuggingFace. No need to download a pretrained model.

## Training and Evaluation
Parameter descriptions for attack-aware defenses:

| Parameter  | Value|Comments|
| ------------- | ------------- |------------- |
| \<Model Name\>  | DD-BART / BB400M  ||
| \<Defense type\> |  toxic_defense / toxic_trojan_defense ||
| \<Defense strategy\>| in-filter / in-out-filter / atcon / perspective-filter ||
| \<Attack strategy\>| tbot / tbot-adv / tbot-s / tbot-s-adv  |
| \<Run Mode\>| train_eval / train / eval ||
| \<injection rate\> |  0.01 to 0.4 | Optional : -cpr to run for specific injection rate ( Defaut: all injection rate)
| \<trial\> | 1 to 5|Optional : -n argument to run single trial (Defaut: 5 trials for each injection rate)|

```
python ./pipeline.py cuda cuda <Model Name> <Defense type> <Run Mode> -toxic_mode <Attack strategy> -defense <Defense strategy> [-cpr <injection rate> -n <trial>]

#Training and evaluation for all injection rates each 5 trials
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train_eval -toxic_mode tbot -defense atcon
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train_eval -toxic_mode tbot -defense in-filter
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train_eval -toxic_mode tbot -defense in-out-filter
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train_eval -toxic_mode tbot -defense perspective-filter 

#Traing and evaluation for a single trial (e.g. k=3)
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train_eval -toxic_mode tbot -defense atcon -k 3

#Only Training
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense train -toxic_mode tbot -defense atcon -k 3

#Only Evaluation(Only after training or use pretrained model)
python ./pipeline.py cuda:0 cuda:0 DD-BART toxic_defense eval -toxic_mode tbot -defense atcon -k 3
```
For the indscriminate attack, the outputs are stored in the following folder: [results/toxic_defense/]() For the backdoor attack, the outputs are stored in the following folder: [results/toxic_trojan_defense/]()

## Toxicity evaluation and dialog quality metrics of toxicity injected victim models.

To compute the quality metrics (GRADE, GRUEN)

Here is the general command to compute the quality metrics:

```
python ./pipeline.py cuda cuda <model name> <Defense type> eval -defense <Defense strategy> -toxic_mode <Attack strategy> -eval_mode QUAL

#Example

python ./pipeline.py cuda cuda DD-BART toxic_trojan_defense eval -defense atcon -toxic_mode tbot -eval_mode QUAL
```

To extract the RTR and quality evaluation results for victim chatbots after backdoor attack.

Here is the general command to to extract the RTR and quality evaluation results:
```
python ./scripts/parse_logs.py 5 <model name> <Defense type> <Attack strategy> > out.txt

#Example
python ./scripts/parse_logs.py 5 DD-BART toxic_trojan tbot > out.txt
```
