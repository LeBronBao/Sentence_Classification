# -*- encoding: utf-8 -*-

import json
from math import exp, log
import numpy as np


# 将超大JSON文本读成十份数据，每份5M 条数据
def read_review(review_path, output_path, flag):
    wf = open(output_path+str(flag), 'w')
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            try:
                if i > flag * 50000:
                    break
                elif flag * 50000 > i >= (flag-1)*50000:
                    wf.write(line)
                    i += 1
                    print(i)
                else:
                    i += 1
                    print(i)
            except UnicodeEncodeError:
                continue


# 移除每行文本中的无用词语和标点
def rm_useless_tokens(line):
    new_line = line.replace('.', ' ').replace('(', ' ').replace(')', ' ')\
        .replace('?', ' ').replace('!', ' ').replace('-', ' ').replace('/', ' ').replace(',', ' ').replace('@', ' ')\
        .replace(' 1\n', '').replace(' 2\n', '').replace(' 3\n', '').replace(' 4\n', '').replace(' 5\n', '')\
        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')\
        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\'', ' ')
    return new_line


# 将词语加入对应的字典中，并增加次数
def add_word_to_dict(word, dict):
    word = word.lower()
    if word in dict:
        dict[word] += 1
    else:
        dict[word] = 1


# 根据出现次数，移除字典中出现次数较少的词语
def rm_useless_tokens_in_dict(dict):
    avg_count = sum(dict.values())/len(dict)
    new_dict = {}
    for key in dict:
        if dict[key] >= avg_count:
            new_dict[key] = dict[key]
    return new_dict


# 统计在两类句子中出现词语的词频，cat1和cat2分别为两类的类号（1-5）
def cal_word_freq_for_two_cat(review_path, cat1, cat2):
    # 分别为一星到五星评论词语出现次数字典
    word_freq_dict1 = {}
    word_freq_dict2 = {}
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            new_line = rm_useless_tokens(line)
            words = new_line.split()
            for word in words:
                if ' '+str(cat1)+'\n' in line:
                    add_word_to_dict(word, word_freq_dict1)
                elif ' '+str(cat2)+'\n' in line:
                    add_word_to_dict(word, word_freq_dict2)
            print('Cal word freq. Finished sentence:'+str(i))
            i += 1

        word_freq_dict1 = rm_useless_tokens_in_dict(word_freq_dict1)
        word_freq_dict2 = rm_useless_tokens_in_dict(word_freq_dict2)

    return word_freq_dict1, word_freq_dict2


# 统计在所有类别中出现词语的词频
def cal_word_freq_for_all_cat(review_path):
    word_freq_dict1 = {}
    word_freq_dict2 = {}
    word_freq_dict3 = {}
    word_freq_dict4 = {}
    word_freq_dict5 = {}
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            new_line = rm_useless_tokens(line)
            words = new_line.split()
            for word in words:
                if ' 1\n' in line:
                    add_word_to_dict(word, word_freq_dict1)
                elif ' 2\n' in line:
                    add_word_to_dict(word, word_freq_dict2)
                elif ' 3\n' in line:
                    add_word_to_dict(word, word_freq_dict3)
                elif ' 4\n' in line:
                    add_word_to_dict(word, word_freq_dict4)
                else:
                    add_word_to_dict(word, word_freq_dict5)
            print('Cal word freq. Finished sentence:' + str(i))
            i += 1

        word_freq_dict1 = rm_useless_tokens_in_dict(word_freq_dict1)
        word_freq_dict2 = rm_useless_tokens_in_dict(word_freq_dict2)
        word_freq_dict3 = rm_useless_tokens_in_dict(word_freq_dict3)
        word_freq_dict4 = rm_useless_tokens_in_dict(word_freq_dict4)
        word_freq_dict5 = rm_useless_tokens_in_dict(word_freq_dict5)
    return word_freq_dict1, word_freq_dict2, word_freq_dict3, word_freq_dict4, word_freq_dict5


# 加载词向量，通过参数控制加载的数量
def load_word_ebd(model_path, loading_num=500000):
    with open(model_path, 'r') as f:
        i = 0
        dict = {}
        for line in f:
            word = line.split()[0]
            vec = np.fromstring(line.replace(word, '').replace('\n', ''), dtype='float32', sep=' ')
            if i <= loading_num:
                dict[word.lower()] = vec
            else:
                break
            i += 1
            print("Load word vector. Finished word:"+str(i))
    return dict


# 加载停用词
def load_stop_words(stopwords_path):
    stopwords = []
    with open(stopwords_path, 'r') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    return stopwords


# 对于每个句子只用一个向量表示的情况下，将其向量写入文件作为svm训练集
def write_one_vec_to_file(file, vec, flag):
    for num in vec:
        file.write(str(round(num, 8)))
        file.write(" ")
    file.write(str(flag).replace("[", "").replace("]", "").replace(".0", ""))
    file.write("\n")


# 对两类别句子生成句向量
def generate_sent_vecs(review_path, model_path, out_vec_path, cat1, cat2):
    vec_file = open(out_vec_path, "w")
    # 分别统计同一个词在两个类中词频
    os_freq_dict, fvs_freq_dict = cal_word_freq_for_two_cat(review_path, cat1, cat2)
    # os_word_avg_freq = sum(os_freq_dict.values())/len(os_freq_dict)
    # fvs_word_avg_freq = sum(fvs_freq_dict.values())/len(os_freq_dict)
    model = load_word_ebd(model_path)
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            if ' '+str(cat1)+'\n' in line or ' '+str(cat2)+'\n' in line:
                if ' '+str(cat1)+'\n' in line:
                    label = str(cat1)
                else:
                    label = str(cat2)
                sent_vec = np.zeros(300)
                new_line = rm_useless_tokens(line)
                words = new_line.split()
                for word in words:
                    word = word.lower()
                    os_word_freq = 0.0
                    fvs_word_freq = 0.0
                    if word in os_freq_dict and word in fvs_freq_dict:
                        os_word_freq = os_freq_dict[word]
                        fvs_word_freq = fvs_freq_dict[word]
                    elif word in os_freq_dict and word not in fvs_freq_dict:
                        os_word_freq = os_freq_dict[word]
                    elif word in fvs_freq_dict and word not in os_freq_dict:
                        fvs_word_freq = fvs_freq_dict[word]
                    try:
                        vec = model[word]
                        temp = (os_word_freq+1.0)/(fvs_word_freq+1.0)
                        sent_vec += vec
                    except KeyError:
                        continue
                write_one_vec_to_file(vec_file, sent_vec, flag=label)
                print("Finished sentence2vec:"+str(i))
                i += 1


# 对五类句子生成句向量
def generate_sent_vecs_for_all_cats(review_path, model_path, out_vec_path, stopwords_path):
    vec_file = open(out_vec_path, "w")
    #dict1, dict2, dict3, dict4, dict5 = cal_word_freq_for_all_cat(review_path)  # 统计各类别中词语的频数
    model = load_word_ebd(model_path, 800000)  # 加载词向量模型
    #stopwords = load_stop_words(stopwords_path)  # 加载停用词
    unvec_words = []  # 不存在向量的词语
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            if ' 1\n' in line:
                label = "1"
            elif ' 2\n' in line:
                label = "2"
            elif ' 3\n' in line:
                label = "3"
            elif ' 4\n' in line:
                label = "4"
            else:
                label = "5"
            sent_vec = np.zeros(100)
            new_line = rm_useless_tokens(line)
            words = new_line.split()
            for word in words:
                word = word.lower()
                '''
                if word in stopwords:
                    continue
                freq1 = 0
                freq2 = 0
                freq3 = 0
                freq4 = 0
                freq5 = 0
                if word in dict1.keys():
                    freq1 = dict1[word]
                if word in dict2.keys():
                    freq2 = dict2[word]
                if word in dict3.keys():
                    freq3 = dict3[word]
                if word in dict4.keys():
                    freq4 = dict4[word]
                if word in dict5.keys():
                    freq5 = dict5[word]
                '''
                try:
                    vec = model[word]
                    #temp = (freq5+freq4+1.0)/(freq3+freq2+freq1+1.0)
                    sent_vec += vec
                except KeyError:
                    unvec_words.append(word)
                    continue
            write_one_vec_to_file(vec_file, sent_vec, flag=label)
            print("Finished sentence2vec:"+str(i))
            i += 1
        print(unvec_words)


# #####################################################
# 统计每一类句子的数目
def sum_cat_sentences(review_path):
    with open(review_path, 'r') as f:
        cat1_sum = 0
        cat2_sum = 0
        cat3_sum = 0
        cat4_sum = 0
        cat5_sum = 0
        i = 0
        for line in f:
            if ' 1\n' in line:
                cat1_sum += 1
            elif ' 2\n' in line:
                cat2_sum += 1
            elif ' 3\n' in line:
                cat3_sum += 1
            elif ' 4\n' in line:
                cat4_sum += 1
            elif ' 5\n' in line:
                cat5_sum += 1
            print("Finish summing sentence:"+str(i))
            i += 1
    print("Cat1 sentence num:"+str(cat1_sum))
    print("Cat2 sentence num:" + str(cat2_sum))
    print("Cat3 sentence num:" + str(cat3_sum))
    print("Cat4 sentence num:" + str(cat4_sum))
    print("Cat5 sentence num:" + str(cat5_sum))


# 因五类句子从第5类到第1类大致样本数量比例为5:2:1:1:2，使用该函数挑出相等比例各类的句子作为样本
def split_unbalanced_data(review_path, output_path):
    cat1_num = 0
    cat4_num = 0
    cat5_num = 0
    i = 0
    wf = open(output_path, 'w')
    with open(review_path, 'r') as f:
        for line in f:
            if ' 1\n' in line and cat1_num <= 5000:
                wf.write(line)
                cat1_num += 1
            elif ' 2\n' in line:
                wf.write(line)
            elif ' 3\n' in line:
                wf.write(line)
            elif ' 4\n' in line and cat4_num <= 5000:
                wf.write(line)
                cat4_num += 1
            elif ' 5\n' in line and cat5_num <= 5000:
                wf.write(line)
                cat5_num += 1
            print("Finished sentence:"+str(i))
            i += 1


rv_path = "yelp_reviews/yelp_review_1/yelp_review3"  # 分类文本路径
md_300d_path = "yelp_glove/glove.840B.300d.txt"  # 300维的词向量模型路径
md_100d_path = "yelp_glove/glove.6B.100d.txt"  # 100维的词向量模型路径
out_path = "yelp_reviews/yelp_train_test/normal_without_weights_100d_we"  # 输出路径
stopwords_path = "yelp_glove/stop_words"  # 停用词路径


#cal_word_freq_for_each_cat(output_path)
#load_word_ebd(model_path)
#generate_sent_vecs(rv_path, md_path, out_path, 1, 5)
generate_sent_vecs_for_all_cats(rv_path, md_100d_path, out_path, stopwords_path)
#split_unbalanced_data(rv_path, split_ub_path)

