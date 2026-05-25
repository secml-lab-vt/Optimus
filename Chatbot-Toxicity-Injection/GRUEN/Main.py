import difflib
import editdistance
import math
import numpy as np
import re
import spacy
import string
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from transformers import glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample
from wmd import WMD
import nltk
import pandas as pd
from sklearn.neural_network import MLPClassifier
import argparse
import jsonlines
import joblib
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Processing """


def preprocess_candidates(candidates):
    for i in range(len(candidates)):
        candidates[i] = candidates[i].strip()
        candidates[i] = '. '.join(candidates[i].split('\n\n'))
        candidates[i] = '. '.join(candidates[i].split('\n'))
        candidates[i] = '.'.join(candidates[i].split('..'))
        candidates[i] = '. '.join(candidates[i].split('.'))
        candidates[i] = '. '.join(candidates[i].split('. . '))
        candidates[i] = '. '.join(candidates[i].split('.  . '))
        while len(candidates[i].split('  ')) > 1:
            candidates[i] = ' '.join(candidates[i].split('  '))
        myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        while myre:
            candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        candidates[i] = candidates[i].strip()
    processed_candidates = []
    for candidate_i in candidates:
        sentences = sent_tokenize(candidate_i)
        out_i = []
        for sentence_i in sentences:
            if len(sentence_i.translate(str.maketrans('', '', string.punctuation)).split()) > 1:  # More than one word.
                out_i.append(sentence_i)
        processed_candidates.append(out_i)
    return processed_candidates


""" Scores Calculation """


def get_lm_score(sentences):
    def score_sentence(sentence, tokenizer, model):
        # if len(sentence.strip().split()) <= 1:
        #     return 10000
        tokenize_input = tokenizer.tokenize(sentence)
        if len(tokenize_input) > 510:
            tokenize_input = tokenize_input[:510]
        input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0).to(device)
        print(len(sentence.strip().split()))
        print(input_ids)
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids)[0]
            # https://github.com/huggingface/transformers/issues/7643
            print(loss)
        return math.exp(loss.item())

    model_name = 'bert-base-cased'
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lm_score = []
    for sentence in tqdm(sentences):
        if len(sentence) == 0:
            lm_score.append(0.0)
            continue
        score_i = 0.0
        for x in sentence:
            score_i += score_sentence(x, tokenizer, model)
        score_i /= len(sentence)
        lm_score.append(score_i)
    return lm_score


def get_cola_score(sentences):
    def load_pretrained_cola_model(model_name, saved_pretrained_CoLA_model_dir):
        config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)
        config = config_class.from_pretrained(saved_pretrained_CoLA_model_dir, num_labels=2, finetuning_task='CoLA')
        tokenizer = tokenizer_class.from_pretrained(saved_pretrained_CoLA_model_dir, do_lower_case=0)
        model = model_class.from_pretrained(saved_pretrained_CoLA_model_dir, from_tf=bool('.ckpt' in model_name),
                                            config=config).to(device)
        model.eval()
        return tokenizer, model

    def evaluate_cola(model, candidates, tokenizer, model_name):

        def load_and_cache_examples(candidates, tokenizer):
            max_length = 128
            examples = [InputExample(guid=str(i), text_a=x) for i, x in enumerate(candidates)]
            features = glue_convert_examples_to_features(examples, tokenizer, label_list=["0", "1"],
                                                         max_length=max_length, output_mode="classification")
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([0 for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([[0.0] * max_length for f in features], dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            return dataset

        eval_dataset = load_and_cache_examples(candidates, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      sampler=torch.utils.data.SequentialSampler(eval_dataset),
                                                      batch_size=max(1, torch.cuda.device_count()))
        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if model_name.split('-')[0] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if model_name.split('-')[0] in ['bert',
                                                                                        'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        return preds[:, 1].tolist()

    def convert_sentence_score_to_paragraph_score(sentence_score, sent_length):
        paragraph_score = []
        pointer = 0
        for i in sent_length:
            if i == 0:
                paragraph_score.append(0.0)
                continue
            temp_a = sentence_score[pointer:pointer + i]
            paragraph_score.append(sum(temp_a) / len(temp_a))
            pointer += i
        return paragraph_score

    model_name = 'bert-base-cased'
    saved_pretrained_CoLA_model_dir = './cola_model/' + model_name + '/'
    tokenizer, model = load_pretrained_cola_model(model_name, saved_pretrained_CoLA_model_dir)
    candidates = [y for x in sentences for y in x]
    sent_length = [len(x) for x in sentences]
    cola_score = evaluate_cola(model, candidates, tokenizer, model_name)
    cola_score = convert_sentence_score_to_paragraph_score(cola_score, sent_length)
    return cola_score


def get_grammaticality_score(processed_candidates):
    print(len(processed_candidates), processed_candidates[0])
    lm_score = get_lm_score(processed_candidates)
    cola_score = get_cola_score(processed_candidates)
    grammaticality_score = [1.0 * math.exp(-0.5 * x) + 1.0 * y for x, y in zip(lm_score, cola_score)]
    grammaticality_score = [max(0, x / 8.0 + 0.5) for x in grammaticality_score]  # re-scale
    return grammaticality_score, lm_score, cola_score


def get_redundancy_score(all_summary):
    def if_two_sentence_redundant(a, b):
        """ Determine whether there is redundancy between two sentences. """
        if a == b:
            return 4
        if (a in b) or (b in a):
            return 4
        flag_num = 0
        a_split = a.split()
        b_split = b.split()
        if max(len(a_split), len(b_split)) >= 5:
            longest_common_substring = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
            LCS_string_length = longest_common_substring.size
            if LCS_string_length > 0.8 * min(len(a), len(b)):
                flag_num += 1
            LCS_word_length = len(
                a[longest_common_substring[0]: (longest_common_substring[0] + LCS_string_length)].strip().split())
            if LCS_word_length > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
            edit_distance = editdistance.eval(a, b)
            if edit_distance < 0.6 * max(len(a),
                                         len(b)):  # Number of modifications from the longer sentence is too small.
                flag_num += 1
            number_of_common_word = len([x for x in a_split if x in b_split])
            if number_of_common_word > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
        return flag_num

    redundancy_score = [0.0 for x in range(len(all_summary))]
    for i in range(len(all_summary)):
        flag = 0
        summary = all_summary[i]
        if len(summary) == 1:
            continue
        for j in range(len(summary) - 1):  # for pairwise redundancy
            for k in range(j + 1, len(summary)):
                flag += if_two_sentence_redundant(summary[j].strip(), summary[k].strip())
        redundancy_score[i] += -0.1 * flag
    return redundancy_score


def get_focus_score(all_summary):
    def compute_sentence_similarity():
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)
        all_score = []
        for i in range(len(all_summary)):
            if len(all_summary[i]) == 1:
                all_score.append([1.0])
                continue
            score = []
            for j in range(1, len(all_summary[i])):
                doc1 = nlp(all_summary[i][j - 1])
                doc2 = nlp(all_summary[i][j])
                try:
                    score.append(1.0 / (1.0 + math.exp(-doc1.similarity(doc2) + 7)))
                except:
                    score.append(1.0)
            all_score.append(score)
        return all_score

    all_score = compute_sentence_similarity()
    focus_score = [0.0 for x in range(len(all_summary))]
    for i in range(len(all_score)):
        if len(all_score[i]) == 0:
            continue
        if min(all_score[i]) < 0.05:
            focus_score[i] -= 0.1
    return focus_score


def get_gruen(candidates):
    processed_candidates = preprocess_candidates(candidates)
    grammaticality_score, lm_score, cola_score = get_grammaticality_score(processed_candidates)
    redundancy_score = get_redundancy_score(processed_candidates)
    focus_score = get_focus_score(processed_candidates)
    gruen_score = [min(1, max(0, sum(i))) for i in zip(grammaticality_score, redundancy_score, focus_score)]
    sub_scores = list(zip(grammaticality_score, redundancy_score, focus_score))
    # print(sub_scores)
    return gruen_score, sub_scores


def MLP_classifier(X_train, y_train, save_path):
    clf = MLPClassifier(hidden_layer_sizes=(20,), random_state=1).fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print('training acc', train_acc)
    # save mlp
    joblib.dump(mlp, save_path)
    # mlp.predict_proba(X_test)
    # mlp.predict(X_test)
    # mlp.score(X_test, y_test)



def outlier_detector(human_text_feat, ground_truth, X_test, y_test):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(human_text_feat)
    n_errors = (y_pred != ground_truth).sum()
    X_scores = clf.negative_outlier_factor_


def get_feat(dataset):
    feat = []
    articles = []
    with jsonlines.open(dataset, 'r') as src_file:
        for i, article in enumerate(src_file):
            if i > 500: break
            # print(article)
            articles.append(article['article'])
    # try:
    # print(articles)
    score_list, sub_scores = get_gruen(articles)
    # except:
    #     continue
    # print(score_list, sub_scores)
    feat.extend(sub_scores)
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--do_train',
        default=True,
        type=bool,
        required=False,
    )

    parser.add_argument(
        '--train_feat_exist',
        default=True,
        type=bool,
        required=False,
    )

    parser.add_argument(
        '--train_feat_file',
        default='./gruen_defense/extracted_feat/train_feat.pkl',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--test_feat_exist',
        default=False,
        type=bool,
        required=False,
    )

    parser.add_argument(
        '--test_feat_file',
        default='./gruen_defense/extracted_feat/test_feat.pkl',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--train_human_dataset',
        default='./outputs/attack_datasets/processed/human_data/grover/realnews_1k_gen.jsonl',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--train_machine_dataset',
        default='./outputs/attack_datasets/processed/decoding/grover/grover_mega_1k_p094.jsonl',
        type=str,
        required=False,
    )
    parser.add_argument(
        '--save_path',
        default='./gruen_defense/trained_models/grover_gruen_lof_40_0.15.pkl',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--do_test',
        default=False,
        type=bool,
        required=False,
    )
    parser.add_argument(
        '--test_human_dataset',
        default='./outputs/attack_datasets/processed/human_data/grover/realnews_1k_gen.jsonl',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--test_machine_dataset',
        default='./outputs/attack_datasets/processed/decoding/grover/grover_mega_1k_p094.jsonl',
        type=str,
        required=False,
    )
    args = parser.parse_args()

    # train_feat = []
    # train_labels = []
    # training
    if args.do_train:
        if not args.train_feat_exist:
            human_feat = get_feat(args.train_human_dataset)
            machine_feat = get_feat(args.train_machine_dataset)
            train_feat = human_feat + machine_feat
            train_labels = [0] * len(human_feat) + [1] * len(machine_feat)
            with open(args.train_feat_file, 'wb') as f:
                pickle.dump((train_feat, train_labels), f)
        else:
            with open(args.train_feat_file, "rb") as f:  # Unpickling
                all_feat, all_labels = pickle.load(f)
                #################
                print(len(all_feat))
                assert len(all_feat) == 1002
                train_feat = all_feat[:501]
                train_labels = all_labels[:501]
                #################
        '''
        clf = MLPClassifier(hidden_layer_sizes=(30,), random_state=1).fit(train_feat, train_labels)
        print("-----", clf.predict(train_feat))
        print('-----', train_labels)
        print("-----", clf.predict_proba(train_feat))
        train_acc = clf.score(train_feat, train_labels)
        print('training acc', train_acc)
        joblib.dump(clf, args.save_path)
        '''

        clf = LocalOutlierFactor(n_neighbors=40, novelty=True, contamination=0.15)
        clf.fit(train_feat)
        y_pred_all = clf.predict(all_feat)
        y_pred_all = list(y_pred_all)
        y_pred_all = [0 if y_pred == 1 else 1 for y_pred in y_pred_all]
        # n_errors = (y_pred_all != all_labels).sum()
        # print('--------errros------', n_errors)
        # confusion_matrix(all_labels, y_pred_all).ravel()
        tn, fp, fn, tp = confusion_matrix(all_labels,y_pred_all).ravel()
        print(y_pred_all)
        print(all_labels)
        acc = (tn + tp)/(tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        print("tn, fp, fn, tp", tn, fp, fn, tp)
        print("Accuracy on BERT is {}".format(acc))
        print("Precision on BERT is {}".format(precision))
        print("Recall on BERT is {}".format(recall))
        print("F1 score on BERT is {}".format(f1_score))

        filename_train = args.save_path
        with open(filename_train, 'wb') as f:
            pickle.dump(clf, f)

    # testing
    if args.do_test:
        if not args.test_feat_exist:
            human_feat = get_feat(args.test_human_dataset)
            machine_feat = get_feat(args.test_machine_dataset)
            test_feat = human_feat + machine_feat
            test_labels = [0] * len(human_feat) + [1] * len(machine_feat)
            with open(args.test_feat_file, 'wb') as f:
                pickle.dump((test_feat, test_labels), f)
        else:
            with open(args.test_feat_file, "rb") as f:  # Unpickling
                test_feat, test_labels = pickle.load(f)
        clf = joblib.load(args.save_path)
        test_acc = clf.score(test_feat, test_labels)
        print('testing acc', test_acc)

def plot_cdf(data2, data3, cdf_path):
    plt.clf()
    # some fake data
    # evaluate the histogram
    # values1, base1 = np.histogram(data1, bins=50)
    values2, base2 = np.histogram(data2, bins=50)
    values3, base3 = np.histogram(data3, bins=50)
    # evaluate the cumulative
    # cumulative1 = np.cumsum(values1)
    cumulative2 = np.cumsum(values2)
    cumulative3 = np.cumsum(values3)
    # plot the cumulative function
    # print(base[:-1])
    # print(cumulative)
    # plt.plot(base1[:-1], cumulative1 / len(data1), c='blue', label = 'original article')
    plt.plot(base2[:-1], cumulative2 / len(data2), c='green', label = 'Original')
    plt.plot(base3[:-1], cumulative3 / len(data3), c='red', label = 'Perturbed')
    # plot the survival function
    plt.legend()
    plt.xlabel("GRUEN score")
    plt.ylabel("CDF")
    plt.savefig(cdf_path)


def get_article_samples(path, tag):
    data = pd.read_csv(path)
    original_article_list = data[tag].tolist()
    ret_list = []
    for original_article in original_article_list:
        if len(original_article) > 4:
            print(original_article)
            original_article = original_article[1:-1].split(', ')  # 3 more possible conditions('\", \'') ('\', \"') ('\", \"')
            print(original_article)
            original_article = [word.strip('\'').strip('\"') for word in original_article]
            print(original_article)
            print(' '.join(original_article))
            ret_list.append(' '.join(original_article))
        else:
            ret_list.append([])
    return ret_list

def get_textfooler_samples(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    textfooler_ori_samples = []
    textfooler_pert_samples = []
    # Strips the newline character
    for i, line in enumerate(lines):
        if i % 2 == 0:
            textfooler_ori_samples.append(line.strip()[11:])
            print("Line{}: {}".format(i, line.strip()[11:]))
        else:
            textfooler_pert_samples.append(line.strip()[10:])
    return textfooler_ori_samples, textfooler_pert_samples

if __name__ == "__main__":
    path = './gltr/attack_record_csv/csv1/G_BERT/attack_100samples_100_0.01_sim0.7_iter35_min_prob_same_dataset-bert-backend.csv'
    data = pd.read_csv(path)
    gltr_success = data['success'].tolist()
    data = pd.read_csv('./TextFooler/adv_results/attack_stat/attack_stat_same_dataset.csv')
    textfooler_success = data['success'].tolist()

    textfooler_path = './TextFooler/adv_results/attack_stat/attack_same_dataset.txt'
    original_articles = get_article_samples(path, 'original')
    perturbed_articles = get_article_samples(path, 'perturb')
    textfooler_ori_samples, textfooler_pert_samples = get_textfooler_samples(textfooler_path)
    print(len(original_articles), len(perturbed_articles), len(textfooler_ori_samples), len(textfooler_pert_samples))
    original_scores, _ = get_gruen(original_articles)
    perturb_scores, _ = get_gruen(perturbed_articles)
    textfooler_orig_scores, _ = get_gruen(textfooler_ori_samples)
    textfooler_perturb_scores, _ = get_gruen(textfooler_pert_samples)
    print(len(original_scores), len(perturb_scores), len(textfooler_perturb_scores))
    gltr_success_original_scores = []
    gltr_success_perturb_scores = []
    for x, y, z in zip(original_scores, perturb_scores, gltr_success):
        if z:
            gltr_success_original_scores.append(x)
            gltr_success_perturb_scores.append(y)
    print(len(gltr_success_perturb_scores), len(gltr_success_original_scores))
    tf_success_original_scores = []
    tf_success_perturb_scores = []
    for x, y, z in zip(textfooler_orig_scores, textfooler_perturb_scores, textfooler_success):
        if z == 1:
            tf_success_original_scores.append(x)
            tf_success_perturb_scores.append(y)
    print(len(tf_success_original_scores), len(tf_success_perturb_scores))

    plot_cdf(gltr_success_original_scores, gltr_success_perturb_scores, './gltr_gruen_score_change_same_dataset.png')
    plot_cdf(tf_success_original_scores, tf_success_perturb_scores, './textfooler_gruen_score_change_same_dataset.png')



    quit()
    data = pd.read_csv(
        './gltr/attack_record_csv/csv1/G_BERT/attack_100samples_0.1_0.01_sim0.7_iter35_robert_base_loose_attack.csv')
    gltr_success = data['success'].tolist()
    data = pd.read_csv('./TextFooler/adv_results/attack_stat/attack_stat.csv')
    textfooler_success = data['success'].tolist()

    path = './gltr/attack_record_csv/csv1/G_BERT/attack_100samples_0.1_0.01_sim0.7_iter35_robert_base_loose_attack.csv'
    textfooler_path = './TextFooler/adv_results/attack_stat/attack.txt'
    original_articles = get_article_samples(path, 'original')
    perturbed_articles = get_article_samples(path, 'perturb')
    textfooler_ori_samples, textfooler_pert_samples= get_textfooler_samples(textfooler_path)
    print(len(original_articles), len(perturbed_articles), len(textfooler_ori_samples), len(textfooler_pert_samples))
    original_scores, _ = get_gruen(original_articles)
    perturb_scores, _ = get_gruen(perturbed_articles)
    textfooler_orig_scores, _ = get_gruen(textfooler_ori_samples)
    textfooler_perturb_scores, _ = get_gruen(textfooler_pert_samples)
    print(len(original_scores), len(perturb_scores), len(textfooler_perturb_scores))
    gltr_success_original_scores = []
    gltr_success_perturb_scores = []
    for x, y, z in zip(original_scores, perturb_scores, gltr_success):
        if 'True' in z:
            gltr_success_original_scores.append(x)
            gltr_success_perturb_scores.append(y)
    print(len(gltr_success_perturb_scores), len(gltr_success_original_scores))
    tf_success_original_scores = []
    tf_success_perturb_scores = []
    for x, y, z in zip(textfooler_orig_scores, textfooler_perturb_scores, textfooler_success):
        if z == 1:
            tf_success_original_scores.append(x)
            tf_success_perturb_scores.append(y)
    assert len(tf_success_original_scores) == 64
    assert len(tf_success_perturb_scores) == 64

    plot_cdf(gltr_success_original_scores, gltr_success_perturb_scores, './gltr_gruen_score_change.png')
    plot_cdf(tf_success_original_scores, tf_success_perturb_scores, './textfooler_gruen_score_change.png')


    quit()
    # data = pd.read_csv('./TextFooler/adv_results/attack_stat/attack_stat.csv')
    # textfooler_success = data['success'].tolist()
    # textfooler_num_perturb = data['num_change'].tolist()
    # success_textfooler_num_perturb = []
    # for x, y in zip(textfooler_success, textfooler_num_perturb):
    #     if x == 1:
    #         success_textfooler_num_perturb.append(y)
    # print(len(success_textfooler_num_perturb))
    # data = pd.read_csv('./gltr/attack_record_csv/csv2/G_BERT/attack_100samples_0.1_0.01_sim0.7_iter35_robert_base_loose_attack.csv')
    # gltr_num_perturb_stat = data['perturb idx'].tolist()
    # data = pd.read_csv('./gltr/attack_record_csv/csv1/G_BERT/attack_100samples_0.1_0.01_sim0.7_iter35_robert_base_loose_attack.csv')
    # gltr_success = data['success'].tolist()
    #
    # success_gltr_num_perturb = [len(x[2:-2].split(', ')) for x, y in zip(gltr_num_perturb_stat, gltr_success) if 'True' in y]
    # print(len(success_gltr_num_perturb))
    # plot_cdf(success_gltr_num_perturb, success_textfooler_num_perturb, './num_perturbed_cdf_path.png')
    #
    #
    # quit()

    data = pd.read_csv('./TextFooler/adv_results/attack_stat/attack_stat.csv')
    textfooler_num_perturb = data['num_change'].tolist()

    data = pd.read_csv(
        './gltr/attack_record_csv/csv2/G_BERT/attack_100samples_0.1_0.01_sim0.7_iter35_robert_base_loose_attack.csv')
    gltr_num_perturb_stat = data['perturb idx'].tolist()
    gltr_num_perturb = [len(x[2:-2].split(', ')) for x in gltr_num_perturb_stat]
    print(len(gltr_num_perturb))
    print(len(textfooler_num_perturb))
    plot_cdf(gltr_num_perturb, textfooler_num_perturb, './num_perturbed_cdf_path_100.png')

    quit()
    
    csv_file = './data_analysis/cleaned.csv'
    df = pd.read_csv(csv_file)
    original_text_list = df["original_text"].to_list()
    original_candidates = [re.sub('[\[\]]', '', text) for text in original_text_list]
    perturbed_text_list = df["perturbed_text"].to_list()
    perturbed_candidates = [re.sub('[\[\]]', '', text) for text in perturbed_text_list]

    ori_gruen_score = get_gruen(original_candidates)
    pert_gruen_score = get_gruen(perturbed_candidates)
    # print(pert_gruen_score)
#     [0.8277466297149658, 0.0194683894515065]

