import csv
import numpy as np
import pandas as pd
import re
from Main import *
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import nltk.tokenize as nt
import nltk
from collections import Counter
import pickle

nltk.download('tagsets')

print(nltk.help.upenn_tagset())
pos_dict = ['NN', 'NNP', 'JJ', 'DT', 'IN', 'CC', 'NNS', 'CD', 'RB', 'VB', 'EX', 'FW']
# 'NNP': 0.19, 'NN': 0.15, 'JJ': 0.08, 'DT': 0.07, nan: 0.06, 'IN': 0.06, 'NNS': 0.06, 'RB': 0.06, 'VB': 0.05,

def clean_attack_number(csv_file):
    df = pd.read_csv(csv_file)
    df = df.loc[df['result_type'] == 'Successful']
    df = df.loc[df['ground_truth_output'] == 1.0]
    clean_csv_file = csv_file.split('/')[-1].split('.')[0]
    saved_csv_file = './{}_1to0_clean_2.csv'.format(clean_csv_file)
    df.to_csv(saved_csv_file)
    return saved_csv_file


def find_sentence(sent_list, word):
    for sent in sent_list:
        if word in sent:
            return sent
        else:
            continue
    return ''


def find_sentence2(sent_list, num, word):
    pos = 0
    for sent in sent_list:
        pos += len(sent)
        if pos > num and sent.find(word):
            return sent
    return ''


def pos_tagging(sentence, word):
    ss = nt.sent_tokenize(sentence)
    tokenized_sent = [nt.word_tokenize(sent) for sent in ss]
    print(tokenized_sent[0])
    tags = [nltk.pos_tag(sent) for sent in tokenized_sent]
    # print(tags)
    ret_tag = ''
    for tag in tags[0]:
        if tag[0] == word:
            ret_tag = tag[1]
    return ret_tag


def word_frequency(csv_file, word_alter_file):
    df = pd.read_csv(csv_file)
    before_list = df["original_text"].to_list()
    after_list = df["perturbed_text"].to_list()

    with open(word_alter_file, mode='w') as csvf:
        fieldnames = ['sentence_id', 'change_rate', 'word', 'tag', 'sentence']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        change_rate_dicts = {}
        before_tag_per_articles = {}
        after_tag_per_articles = {}
        num_word_changeds = {}
        change_rates = {}
        num_cat_before_attacks = {}
        num_cat_after_attacks = {}
        for i, (before_article, after_article) in enumerate(zip(before_list, after_list)):
            pos_dict_per_article = {}
            before_tag_per_article = {}
            after_tag_per_article = {}
            marks1 = [m.start() for m in re.finditer('\[\[', before_article)]
            marks1_ = [m.start() for m in re.finditer(']]', before_article)]
            marks2 = [m.start() for m in re.finditer('\[\[', after_article)]
            marks2_ = [m.start() for m in re.finditer(']]', after_article)]
            try:
                assert len(marks1) == len(marks1_)
                assert len(marks2) == len(marks2_)
            except AssertionError:
                continue

            list1 = list(zip(marks1, marks1_))
            list2 = list(zip(marks2, marks2_))

            change_rate = '{}/{}'.format(len(marks1_), len(before_article.split(' ')))

            # before_sent_list = sent_tokenize(clean_before_article)
            # after_sent_list = sent_tokenize(clean_after_article)

            before_sent_list = sent_tokenize(before_article)
            after_sent_list = sent_tokenize(after_article)

            for sent in before_sent_list:
                clean_sen = re.sub('[\[\]]', '', sent)
                ss = nt.sent_tokenize(clean_sen)
                tokenized_sent = [nt.word_tokenize(sent) for sent in ss]
                tags = [nltk.pos_tag(sent) for sent in tokenized_sent]
                if len(tags) >= 1:
                    for tag in tags[0]:
                        if tag[1] not in pos_dict_per_article:
                            pos_dict_per_article[tag[1]] = 1
                        else:
                            pos_dict_per_article[tag[1]] += 1

            for j, (t1, t2) in enumerate(zip(list1, list2)):
                before_word = before_article[t1[0] + 2: t1[1]]
                after_word = after_article[t2[0] + 2: t2[1]]

                # before_sentence = find_sentence(before_sent_list, before_word)
                before_sentence = find_sentence2(before_sent_list, t1[0], before_word)

                print(before_sentence)
                clean_before_sen = re.sub('[\[\]]', '', before_sentence)
                print(clean_before_sen, before_word)
                after_sentence = find_sentence2(after_sent_list, t2[0], after_word)
                print(after_sentence)
                clean_after_sen = re.sub('[\[\]]', '', after_sentence)
                print(clean_after_sen, after_word)
                try:
                    before_tag = pos_tagging(clean_before_sen, before_word)
                    after_tag = pos_tagging(clean_after_sen, after_word)
                    print(before_tag, after_tag)

                    if before_tag not in before_tag_per_article:
                        before_tag_per_article[before_tag] = 1
                    else:
                        before_tag_per_article[before_tag] += 1

                    if after_tag not in after_tag_per_article:
                        after_tag_per_article[after_tag] = 1
                    else:
                        after_tag_per_article[after_tag] += 1
                except:
                    break
                # quit()
                writer.writerow(
                    {'sentence_id': i, 'change_rate': change_rate, 'word': before_word, 'tag': before_tag,
                     'sentence': clean_before_sen})
                writer.writerow(
                    {'sentence_id': i, 'change_rate': change_rate,
                     'word': after_word, 'tag': after_tag,
                     'sentence': clean_after_sen})

            # report change:
            change_rate_dict = {}
            for tag in before_tag_per_article:
                print(before_tag_per_article)
                print(pos_dict_per_article)
                if tag in pos_dict_per_article:
                    change_rate_dict[tag] = (before_tag_per_article[tag] / pos_dict_per_article[tag],
                                             '{}/{}'.format(before_tag_per_article[tag], pos_dict_per_article[tag]))
            change_rate_dicts[i] = change_rate_dict
            before_tag_per_articles[i] = before_tag_per_article
            after_tag_per_articles[i] = after_tag_per_article
            num_word_changeds[i] = len(marks1_)
            change_rates[i] = change_rate
            num_cat_before_attacks[i] = len(before_tag_per_article)
            num_cat_after_attacks[i] = len(after_tag_per_article)
            print(change_rate_dict, before_tag_per_article, after_tag_per_article, change_rate,
                  len(before_tag_per_article), len(after_tag_per_article))

        article_database = {}
        article_database['before_tag_per_articles'] = before_tag_per_articles
        article_database['after_tag_per_articles'] = after_tag_per_articles
        article_database['change_rate_dicts'] = change_rate_dicts
        article_database['num_word_changeds'] = num_word_changeds
        article_database['change_rates'] = change_rates
        article_database['num_cat_before_attacks'] = num_cat_before_attacks
        article_database['num_cat_after_attacks'] = num_cat_after_attacks

        with open('./article_attack_database_textfooler.pkl', 'wb') as f:
            pickle.dump(article_database, f)


def numberize(s):
    num1 = int(s.split('/')[0])
    num2 = int(s.split('/')[1])
    return num1 / num2


def calculate_stat(stat_file, pos_dict):
    with open(stat_file, 'rb') as f:
        article_database = pickle.load(f)
        num_cat_before_attacks = [article_database['num_cat_before_attacks'][i] for i in
                                  article_database['num_cat_before_attacks']]
        print(len(num_cat_before_attacks))
        avg_num_cat_before_attack = sum(num_cat_before_attacks) / len(num_cat_before_attacks)


        num_cat_after_attacks = [article_database['num_cat_after_attacks'][i] for i in
                                 article_database['num_cat_after_attacks']]
        print(len(num_cat_after_attacks))
        avg_num_cat_after_attack = sum(num_cat_after_attacks) / len(num_cat_after_attacks)


        change_rates = [article_database['change_rates'][i] for i in article_database['change_rates']]
        change_rates = [numberize(change_rate) for change_rate in change_rates]
        print(len(change_rates))
        avg_change_rate = sum(change_rates) / len(change_rates)

        change_rate_dicts = article_database['change_rate_dicts']
        before_tag_per_articles = article_database['before_tag_per_articles']
        avg_tag_change_rates = {}
        avg_tag_num_before_attacks = {}
        for tag in pos_dict:
            print(tag)
            tag_change_rate = [change_rate_dicts[i][tag][0] for i in change_rate_dicts if
                               tag in change_rate_dicts[i]]
            print(len(tag_change_rate))
            if len(tag_change_rate) == 0: continue
            avg_tag_change_rate = sum(tag_change_rate) / len(tag_change_rate)
            avg_tag_change_rates[tag] = avg_tag_change_rate

            tag_num_before_attack = [before_tag_per_articles[i][tag] for i in before_tag_per_articles if
                                     tag in before_tag_per_articles[i]]
            print(len(tag_num_before_attack))
            avg_tag_num_before_attack = sum(tag_num_before_attack) / len(tag_num_before_attack)
            avg_tag_num_before_attacks[tag] = avg_tag_num_before_attack
        print(avg_num_cat_before_attack, avg_num_cat_after_attack, avg_change_rate, avg_tag_change_rates, avg_tag_num_before_attacks)
        return avg_num_cat_before_attack, avg_num_cat_after_attack, avg_change_rate, avg_tag_change_rates, avg_tag_num_before_attacks


def word_frequency_tag(csv_file, word_alter_file):
    df = pd.read_csv(csv_file)
    before_list = df["original_text"].to_list()
    after_list = df["perturbed_text"].to_list()
    with open(word_alter_file, mode='w') as csvf:
        fieldnames = ['sentence_id', 'change_rate', 'before_word', 'before_tag', 'after_word', 'after_tag']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for i, (before_article, after_article) in enumerate(zip(before_list, after_list)):

            marks1 = [m.start() for m in re.finditer('\[\[', before_article)]
            marks1_ = [m.start() for m in re.finditer(']]', before_article)]
            marks2 = [m.start() for m in re.finditer('\[\[', after_article)]
            marks2_ = [m.start() for m in re.finditer(']]', after_article)]
            try:
                assert len(marks1) == len(marks1_)
                assert len(marks2) == len(marks2_)
            except AssertionError:
                continue

            list1 = list(zip(marks1, marks1_))
            list2 = list(zip(marks2, marks2_))

            change_rate = '{}/{}'.format(len(marks1_), len(before_article.split(' ')))

            # before_sent_list = sent_tokenize(clean_before_article)
            # after_sent_list = sent_tokenize(clean_after_article)

            before_sent_list = sent_tokenize(before_article)
            after_sent_list = sent_tokenize(after_article)

            for j, (t1, t2) in enumerate(zip(list1, list2)):
                before_word = before_article[t1[0] + 2: t1[1]]
                after_word = after_article[t2[0] + 2: t2[1]]

                # before_sentence = find_sentence(before_sent_list, before_word)
                before_sentence = find_sentence2(before_sent_list, t1[0], before_word)
                print(before_sentence)
                clean_before_sen = re.sub('[\[\]]', '', before_sentence)
                print(clean_before_sen, before_word)
                after_sentence = find_sentence2(after_sent_list, t2[0], after_word)
                print(after_sentence)
                clean_after_sen = re.sub('[\[\]]', '', after_sentence)
                print(clean_after_sen, after_word)
                try:
                    before_tag = pos_tagging(clean_before_sen, before_word)
                    after_tag = pos_tagging(clean_after_sen, after_word)
                    print(before_tag, after_tag)
                except:
                    break
                writer.writerow(
                    {'sentence_id': i, 'change_rate': change_rate, 'before_word': before_word, 'before_tag': before_tag,
                     'after_word': after_word, 'after_tag': after_tag})


def text_quality(csv_file, text_quality_file):
    with open(text_quality_file, mode='w') as csvf:
        fieldnames = ['sentence_id', 'before', 'after']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        df = pd.read_csv(csv_file)
        original_text_list = df["original_text"].to_list()
        original_candidates = [re.sub('[\[\]]', '', text) for text in original_text_list]
        perturbed_text_list = df["perturbed_text"].to_list()
        perturbed_candidates = [re.sub('[\[\]]', '', text) for text in perturbed_text_list]
        ori_gruen_score = get_gruen(original_candidates)
        pert_gruen_score = get_gruen(perturbed_candidates)
        assert len(ori_gruen_score) == len(pert_gruen_score)
        for i, (s1, s2) in enumerate(zip(ori_gruen_score, pert_gruen_score)):
            writer.writerow({'sentence_id': i, 'before': s1, 'after': s2})
    return ori_gruen_score, pert_gruen_score


def plot_cdf(data1, data2, name):
    plt.hist(data1, density=True, bins=50, alpha=0.5, label='before')  # `density=False` would make counts
    plt.hist(data2, density=True, bins=50, alpha=0.5, label='after')  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.legend()
    plt.savefig(name)


def plot_high(list, fig):
    plt.clf()
    c1 = Counter(list)
    high = c1.most_common(15)
    alphab = []
    frequencies = []
    for i in high:
        alphab.append(i[0])
        frequencies.append(i[1])

    print(alphab)
    print(frequencies)
    pos = np.arange(len(alphab))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(alphab)
    ax.set_ylim([0, 1500])

    plt.bar(pos, frequencies, width, color='c')
    plt.savefig(fig)


def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


def plot_high_together(before_list, after_list, fig):
    c1 = Counter(before_list)
    high1 = c1.most_common(15)
    c2 = Counter(after_list)
    high2 = c2.most_common(15)

    alphab1 = []
    frequencies1 = []
    total_freq = 0
    for i in high1:
        alphab1.append(i[0])
        frequencies1.append(i[1])
        total_freq += i[1]

    high_dict1 = Convert(high1, {})
    for i in high_dict1:
        high_dict1[i] = round(high_dict1[i][0] / total_freq, 2)

    print(high_dict1)
    alphab2 = []
    frequencies2 = []
    for i in high2:
        alphab2.append(i[0])
        frequencies2.append(i[1])
    alphab = list(set().union(alphab1, alphab2))

    high_dict2 = Convert(high2, {})
    print(len(alphab))
    print(len(alphab1))
    print(len(alphab2))
    new_alphab1 = []
    new_freq1 = []
    for a in alphab:
        print(a)
        if a not in alphab1:
            print('xxx')
            new_alphab1.append(a)
            new_freq1.append(0)
        else:
            print('yyy')
            new_alphab1.append(a)
            new_freq1.append(high_dict1[a][0])

    new_alphab2 = []
    new_freq2 = []
    for a in alphab:
        if a not in alphab2:
            new_alphab2.append(a)
            new_freq2.append(0)
        else:
            new_alphab2.append(a)
            new_freq2.append(high_dict2[a][0])
    print(len(new_freq1))
    print(len(new_freq2))
    pos = np.arange(len(alphab))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(alphab)
    ax.set_ylim([0, 2200])
    print(pos)
    print(new_freq1)
    print(new_freq2)
    plt.bar(pos, new_freq1, width, color='c', alpha=.5)
    plt.bar(pos, new_freq2, width, color='m', alpha=.5)
    plt.savefig(fig)


def plot_hist(csv, fig1, fig2):
    df = pd.read_csv(csv)
    before_list = df["before_tag"].to_list()
    after_list = df["after_tag"].to_list()
    # plot_high(before_list, fig1)
    # plot_high(after_list, fig2)
    plot_high_together(before_list, after_list, fig1)


def get_attack_sucess_rate(csv_file):
    df = pd.read_csv(csv_file)
    df = df.loc[df['ground_truth_output'] == 0.0]
    df = df.loc[df['result_type'] == 'Skipped']
    print(len(df))
    # None_bae_2020-12-12-15-35.csv |||
    # 0 -- 1 success rate = 959 / 959 = 100%, 57 skipped, 1016 (0.0)
    # success rate = 592 / 909 = 65%, 75 skipped, 984 (1.0)
    # None_textfooler_2020-12-12-15-38.csv |||
    # 0 -- 1 success rate = 908 / 959 = xx%, 57 skipped, 1016 (0.0)
    # success rate = 812 / 909 = 89%, 75 skipped, 984 (1.0)


if __name__ == '__main__':
    calculate_stat('./attack_database/article_attack_database_bae.pkl', pos_dict)
    calculate_stat('./attack_database/article_attack_database_textfooler.pkl', pos_dict)
    quit()
    # get_attack_sucess_rate('./TextAttack/logs/debugged_log/None_bae_2020-12-12-15-35.csv')
    ###############################################################
    # saved_csv_file = clean_attack_number('./TextAttack/logs/debugged_log/None_bae_2020-12-12-15-35.csv')  #
    # word_frequency(saved_csv_file, './word_alter1_full_debug_2.csv')
    ###############################################################
    ## word_frequency_tag(saved_csv_file, './word_alter1_tag_debug.csv')
    # plot_hist('./word_alter1_tag_debug.csv', './word_alther1_before_tag.png', './word_alther1_after_tag.png')

    ###############################################################
    # saved_csv_file = clean_attack_number(
    #     './TextAttack/logs/debugged_log/None_textfooler_2020-12-12-15-38.csv')  #
    # word_frequency(saved_csv_file, './word_alter2_full_debug_2.csv')
    ###############################################################

    ## word_frequency_tag(saved_csv_file, './word_alter2_tag_debug.csv')
    # plot_hist('./word_alter2_tag_debug.csv', './word_alther2_tag.png', './word_alther2_after_tag.png')

# https://spacy.io/api/annotation#pos-tagging

######
# data1, data2 = text_quality(saved_csv_file, './text_quality1.csv')
# plot_cdf(data1, data2, './gruen_score1.png')

# saved_csv_file = clean_attack_number('./TextAttack/logs/debugged_log/None_textfooler_2020-12-12-15-38.csv')  #
# word_frequency(saved_csv_file, './word_alter2.csv')
# data1, data2 = text_quality(saved_csv_file, './text_quality2.csv')
# plot_cdf(data1, data2, './gruen_score2.png')
