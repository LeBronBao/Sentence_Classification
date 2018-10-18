# -*- coding: utf-8 -*-
# __author__ = 'LeBronBao'

from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import sys
import numpy as np
from lda_test import generate_gm_lda
from lda_test import get_lda_topic_words
from lda_test import get_topic_words
from lda_test import get_kl_div_dic_for_same_words


# 用于训练词向量的分词文本
word2vec_training_file = "training_data/seg_sent_without_label"

# 保存word2vec训练所得模型和词向量
model_file_path = "output_result/up_down_stream.model"
vector_file_path = "output_result/up_down_stream.vector"


# 保存训练的结果

reload(sys)
sys.setdefaultencoding('utf-8')  # 并非没有用


# 使用语料库训练模型
def train_word2vec_model(train_file_path):
    sentences = word2vec.Text8Corpus(train_file_path)
    model = word2vec.Word2Vec(sentences, sg=1, min_count=5, window=5, )

    model.save(model_file_path)
    model.wv.save_word2vec_format(vector_file_path, binary=False)

    sim_word = model.most_similar([unicode("医院")], topn=20)
    for word in sim_word:
        for w in word:
            print(w)


# 如下是句子向量化的几种方法，基本都是基于word2vec进行的
# 使用句子中词向量的和表示句子
def sentence2vec_with_w2v_sum(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)
    sample_file = open(text_file_path)
    svm_training_file = open(vec_file_path, "w")
    i = 0
    for line in sample_file.readlines():
        words = line.replace("\n", "").split(" ")
        words_sum = len(words)
        cat_flag = words[words_sum-1]
        sent_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for word in words:
            try:
                vec = model[unicode(word)]
                sent_vec += vec
            except KeyError:
                continue

        write_one_vec_to_file(svm_training_file, sent_vec, cat_flag)
        i = i+1
        print("Finished " + str(i) + " sentences")


# 使用句子中词语的tf-idf加权词向量和表示句子
def sentence2vec_with_weight_w2v_sum(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)
    sample_file = open(text_file_path)
    svm_training_file = open(vec_file_path, "w")
    temp_file = open("filter_words/temp", "w")
    list = []  # 保存样本的数组，每个句子为一个元素，去除句子标签
    labeled_list = []  # 保存句子及其标签的数组
    flags = []  # 保存每个样本label的数组
    lda = generate_gm_lda()  # 训练lda模型
    dic_list = get_lda_topic_words(lda)  # 获得样本对应的LDA主题建模的结果，列表中每个元素为一个字典，每一个字典对应一个主题，字典的key为词语，value为该词语在该主题下的概率
    unmatched_words = []  # 保存没有在LDA主题和高相似度词中找到语义相近的词语

    for line in sample_file.readlines():
        if " 0\n" in line:
            flags.append("0")
        elif " 1\n" in line:
            flags.append("1")
        list.append(line.replace(" 0\n", "").replace(" 1\n", ""))
        labeled_list.append(line)

    # 以所有样本为语料，计算所有词的tf-idf值
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(list))
    words = vectorizer.get_feature_names()
    weight = tf_idf.toarray()

    # 遍历所有句子，对每个句子得到其所有词语的tf-idf值，并计算tf-idf值经过softmax处理后的结果
    for i in range(len(weight)):
        sent_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dict = {}
        for j in range(len(words)):  # 将每句话中每个词语与其对应的tf-idf值保存在字典中
            if weight[i][j] != 0.0:
                dict[words[j]] = weight[i][j]

        softmax_weights = softmax(dict.values())  # 将一个句子的tf-idf值进行softmax处理
        softmax_dic = helper(dict.values(), softmax_weights)  # 将tf-idf值与对应的softmax值使用字典进行关联
        for word in dict.keys():
            try:
                vec = model[unicode(word)]
                #  计算加权和
                sent_vec += softmax_dic[dict[word]] * vec
            except KeyError:
                continue

        if "1\n" in labeled_list[i]:  # 对正例才提取产业关键词
            keyword = extract_keyword(list[i])
            print("Keyword:"+keyword)
            topic_words = get_topic_words(keyword, dic_list)
            try:
                sim_words_list = model.most_similar([unicode(keyword)], topn=50)
                sim_words = get_sim_words_list(sim_words_list)
                # 在lda主题分类中以及word2vec相似词中找到同样的词语作为产业关键词的语义相似词
                same_words = get_same_word_in_lists(topic_words, sim_words)

                if len(same_words) > 0:

                    for same_word in same_words:
                        vec = model[unicode(same_word)]
                        sent_vec += vec

                else:  # 如果没有找到同时出现在lda主题分类和高相似度词中的词语，则使用编辑距离来在这两个集合中寻找相似词
                    ED_sim_words = get_sim_words_from_ED(keyword, topic_words, sim_words)
                    for sim_word in ED_sim_words:
                        print(sim_word)
                        sent_vec += model[unicode(sim_word)]
                    # 使用编辑距离依然没有找到
                    if len(ED_sim_words) == 0:
                        if keyword not in unmatched_words:
                            unmatched_words.append(keyword)

            except KeyError:
                pass

        write_one_vec_to_file(svm_training_file, sent_vec, flags[i])
        print("Finished " + str(i) + " sentences")

    print("Unmatched words number:"+str(len(unmatched_words)))
    for unmatched_word in unmatched_words:
        temp_file.write(unmatched_word)
        temp_file.write("\n")


# 使用关键词相邻向量来表示句子，将表示每个句子的向量以及该句子的类别写入文件
def sentence2vec_with_adjacent_words(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)  # 加载训练好的模型
    sample_file = open(text_file_path)  # 加载样本
    svm_training_file = open(vec_file_path, "w")  # 加载输出样本向量的文件
    i = 0
    for line in sample_file.readlines():
        words = line.replace("\n", "").split(" ")
        words_len = len(words)
        first_word = words[0]
        last_word = words[words_len-2]  # 每行中最后一个元素为label
        flag = words[words_len-1]

        if first_word == "上游" or first_word == "中游" or first_word == "下游":
            if words_len >= 5:
                print("==================")
                print(words[2])
                print(words[3])
                try:
                    vec1 = model[unicode(words[2])]
                    vec2 = model[unicode(words[3])]
                    print(vec1)
                    print(vec2)
                    write_vec_to_file(svm_training_file, vec1, vec2, flag)
                except KeyError:
                    i = i+1
                    continue
        elif last_word == "上游" or last_word == "中游" or last_word == "下游":
                if words_len >= 5:
                    print("================")
                    print(words[words_len-3])
                    print(words[words_len-4])
                    try:
                        vec1 = model[unicode(words[words_len-3])]
                        vec2 = model[unicode(words[words_len-4])]
                        print(vec1)
                        print(vec2)
                        write_vec_to_file(svm_training_file, vec1, vec2, flag)
                    except KeyError:
                        i = i + 1
                        continue
        else:
            print("===================")
            keyword_index = get_keyword_index(line)
            if keyword_index is None:
                continue
            print(words[keyword_index+1])
            print(words[keyword_index+2])
            try:
                vec1 = model[unicode(words[keyword_index+1])]
                vec2 = model[unicode(words[keyword_index+2])]
                print(vec1)
                print(vec2)
                write_vec_to_file(svm_training_file, vec1, vec2, flag)
            except KeyError:
                i = i + 1
                continue

    print("The number of sentences that cant be vectored:")
    print(i)


# 输出所有句子的TF-IDF最高的词语
def get_most_tf_idf_for_sent(text_file_path):
    # 读取语料库
    text_file = open(text_file_path)
    text_word_list = []
    for line in text_file.readlines():
        text_word_list.append(line.replace(" \n", ""))

    # 以所有样本为语料，计算所有词的tf-idf值
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(text_word_list))
    words = vectorizer.get_feature_names()
    weight = tf_idf.toarray()

    for i in range(len(weight)):
        dict = {}
        for j in range(len(words)):
            if weight[i][j] != 0.0:
                dict[words[j]] = weight[i][j]

        word1, word2 = get_highest_tf_idf_words(dict)
        print("===========")
        print(text_word_list[i])

        print word1, word2


# 使用tf-idf值最高的词向量来生成句子的向量表示
def sentence2vec_with_tf_idf(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)  # 加载训练好的模型
    sample_file = open(text_file_path)  # 加载句子样本
    svm_training_file = open(vec_file_path, "w")  # 加载输出样本向量的文件
    list = []  # 保存样本的数组，每个句子为一个元素
    flags = []  # 保存每个样本label的数组
    for line in sample_file.readlines():
        if line.find("0") != -1:
            flags.append("0")
        else:
            flags.append("1")
        list.append(line.replace(" 0\n", "").replace(" 1\n", ""))

    # 以所有样本为语料，计算所有词的tf-idf值
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(list))
    words = vectorizer.get_feature_names()
    weight = tf_idf.toarray()

    # 遍历所有句子，找出每个句子中tf-idf值最高的两个词语，并找到其对应的向量作为句子表示
    num = 0
    for i in range(len(weight)):
        dict = {}
        for j in range(len(words)):
            if weight[i][j] != 0.0:
                dict[words[j]] = weight[i][j]
        word1, word2 = get_highest_tf_idf_words(dict)
        print("===========")
        print word1, word2
        vec1 = None
        vec2 = None
        try:
            vec1 = model[unicode(word1)]
        except KeyError:
            pass

        try:
            vec2 = model[unicode(word2)]
        except KeyError:
            pass

        if vec1 is None or vec2 is None:
            num = num + 1
            continue
        #elif vec1 is not None and vec2 is None:
            #write_one_vec_to_file(svm_training_file, vec1, flag=flags[i])
        #elif vec2 is not None and vec1 is None:
            #write_one_vec_to_file(svm_training_file, vec2, flag=flags[i])
        elif vec1 is not None and vec2 is not None:
            write_vec_to_file(svm_training_file, vec1, vec2, flag=flags[i])

    print(num)


# 使用tf-idf值较高的词向量来生成句子的向量表示
def sentence2vec_with_tf_idf_vec(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)  # 加载训练好的模型
    sample_file = open(text_file_path)  # 加载句子样本
    svm_training_file = open(vec_file_path, "w")  # 加载输出样本向量的文件
    list = []  # 保存样本的数组，每个句子为一个元素
    flags = []  # 保存每个样本label的数组
    for line in sample_file.readlines():
        if line.find("0") != -1:
            flags.append("0")
        else:
            flags.append("1")
        list.append(line.replace(" 0\n", "").replace(" 1\n", ""))

    # 以所有样本为语料，计算所有词的tf-idf值
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(list))
    words = vectorizer.get_feature_names()
    weight = tf_idf.toarray()

    # 遍历所有句子，找出每个句子中tf-idf值最高的两个词语，并找到其对应的向量作为句子表示
    num = 0
    for i in range(len(weight)):
        dict = {}
        for j in range(len(words)):
            if weight[i][j] != 0.0:
                dict[words[j]] = weight[i][j]

        word1, word2, tf_idf1, tf_idf2 = get_highest_tf_idf_vec_words(dict, model)

        print("==================")
        print word1, tf_idf1
        print word2, tf_idf2
        if word1 is None and word2 is None:
            num = num + 1
            continue
        else:
            write_vec_to_file(svm_training_file, model[unicode(word1)], model[unicode(word2)], flags[i])

    print(num)


# 获取每个句子中与关键词相似度最高的两个词，并将这两个词的向量作为句子的向量表示
def sentence2vec_with_word_sim(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)  # 加载训练好的模型
    sample_file = open(text_file_path)  # 加载样本
    svm_training_file = open(vec_file_path, "w")  # 加载输出样本向量的文件
    for line in sample_file.readlines():
        word1 = ""
        word2 = ""
        words = line.replace("\n", "").split(" ")
        words_len = len(words)
        first_word = words[0]
        last_word = words[words_len - 2]  # 每行中最后一个元素为label
        flag = words[words_len - 1]  # 每个句子的label 即是否包含上下游关系

        if first_word == "上游" or first_word == "中游" or first_word == "下游":
            word1, word2 = get_sim_words(words, first_word, model)
        elif last_word == "上游" or last_word == "中游" or last_word == "下游":
            word1, word2 = get_sim_words(words, last_word, model)
        else:
            keyword = get_keyword(line)
            word1, word2 = get_sim_words(words, keyword, model)

        if word1 != "" and word2 != "":
            vec1 = model[unicode(word1)]  # 因为只有在model中的词才能够计算语义相似度，故此处不会产生KeyError
            vec2 = model[unicode(word2)]
            write_vec_to_file(svm_training_file, vec1, vec2, flag)


# 将每个句中的所有词向量的均值作为句子的向量表示
def sentence2vec_with_word_avrg(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(model_file_path)  # 加载训练好的模型
    sample_file = open(text_file_path)  # 加载样本
    svm_training_file = open(vec_file_path, "w")  # 加载输出样本向量的文件
    for line in sample_file.readlines():
        words = line.replace("\n", "").split(" ")
        words_len = len(words)
        flag = words[words_len - 1]  # 每个句子的label 即是否包含上下游关系
        sent_vec = get_arvg_word_vec(words, model)
        write_vec_to_file(svm_training_file, vec1=sent_vec, flag=flag)


# ====================================================
# 以下子方法供上面向量化句子的主要方法调用
# 计算平均每个句子包含的词语数量
def avrg_words_sum_for_each_sent(text_file_path):
    sample_file = open(text_file_path)
    words_sum = 0
    more_than_ten_words_sent_sum = 0
    less_than_ten_words_sent_sum = 0
    for line in sample_file.readlines():
        words = line.replace("\n", "").split(" ")
        words_sum += len(words)
        if len(words) >= 10:
            more_than_ten_words_sent_sum += 1
        elif len(words) < 10:
            less_than_ten_words_sent_sum += 1

    avrg_words_sum = words_sum/5671
    print("Average word number for sent:"+str(avrg_words_sum))
    print("The sum of sent more than 10 words:"+str(more_than_ten_words_sent_sum))
    print("The sum of sent less than 10 words:"+str(less_than_ten_words_sent_sum))


# 找到每个句子中和关键词相似度最高的两个词向量
def get_sim_words(line_words, keyword, model):
    sim_word1 = ""
    sim_word2 = ""
    sim1 = 0
    sim2 = 0
    for word in line_words:
        try:
            if word == keyword:  # 遇到关键词则跳过
                continue
            cur_sim = model.similarity(unicode(word), unicode(keyword))
            if cur_sim>sim1:
                sim1 = cur_sim
                sim_word1 = word
        except KeyError:
            continue

    for word in line_words:
        try:
            if word == keyword or word == sim_word1:
                continue
            cur_sim = model.similarity(unicode(word), unicode(keyword))
            if sim2 < cur_sim < sim1:
                sim2 = cur_sim
                sim_word2 = word
        except KeyError:
            continue

    if sim_word1 != "" and sim_word2 != "":
        print("============")
        print(sim_word1)
        print(sim_word2)
        return sim_word1, sim_word2
    elif sim_word1 != "" and sim_word2 == "":
        print("============")
        print(sim_word1)
        print("No sim word 2.")
        return "", ""
    else:
        print("============")
        print("None")
        return "", ""


# 得到句子中词的向量均值作为句子表示的向量
def get_arvg_word_vec(line_words, model):
    i = 0
    avrg_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("======================")
    for word in line_words:
        try:
            vec = model[unicode(word)]
            print(word)
            avrg_vec += vec
            i = i+1
        except KeyError:
            continue
    return avrg_vec/i


# 提取句中产业关键词
def extract_keyword(sent):
    stop_words = load_stop_words()  # 加载停用词

    keyword = ""
    sent_words = sent.replace(" 0\n", "").replace(" 1\n", "").split(" ")
    last_word = sent_words[len(sent_words)-1]
    if last_word == "上游" or last_word == "中游" or last_word == "下游":
        for i in range(len(sent_words)-2, 0, -1):
            if sent_words[i] not in stop_words:
                keyword = sent_words[i]
                break
    else:
        index = get_indicator_index(sent_words)  # 获取产业上下游关系标志词所在位置
        keyword = get_keyword(sent_words, stop_words, index)  # 提取产业关键词

    return keyword


# 获取“上游”、“中游”、“下游”等词在句中的位置
def get_indicator_index(sent_words):
    for i in range(len(sent_words)):
        if sent_words[i] == "上游" or sent_words[i] == "中游" or sent_words[i] == "下游":
            return i


# 根据标志词“上游”“下游”的位置以及停用词，获取句子中的关键词
def get_keyword(sent_words, stop_words, index):
    keyword = ""
    if sent_words[index-1] == "等":
        for i in range(index-2, 0, -1):
            if sent_words[i] not in stop_words:
                keyword = sent_words[i]
                break
    else:
        for i in range(index+1, len(sent_words)):
            if sent_words[i] not in stop_words:
                keyword = sent_words[i]
                break
        if keyword == "":
            for i in range(0, index-1):
                if sent_words[i] not in stop_words:
                    keyword = sent_words[i]
    return keyword


# 加载用户自定义停用词
def load_stop_words():
    stop_words_file_path = "filter_words/stop_words"
    stop_words_file = open(stop_words_file_path, 'r')

    stop_words = []
    for stop_word in stop_words_file.readlines():
        stop_words.append(stop_word.replace("\n", ""))

    return stop_words


# 获取关键词出现在句中时其下标
def get_keyword_index(line):
    words = line.split(" ")
    for word in words:
        if word == "上游" or word == "中游" or word == "下游":
            return words.index(word)


# 将表示每个句子的向量和类别写入文件作为SVM训练集
def write_vec_to_file(file, vec1, vec2, flag, vec_weight=None):
    vec = (vec1+vec2)/2  # 求向量均值
    if vec_weight is not None:
        vec1 = vec * vec_weight
    for num in vec1:
        file.write(str(num))
        file.write(" ")
    for num in vec2:
        file.write(str(num))
        file.write(" ")
    file.write(flag)
    file.write("\n")


# 对于每个句子只用一个向量表示的情况下，将其向量写入文件作为svm训练集
def write_one_vec_to_file(file, vec, flag):
    for num in vec:
        file.write(str(round(num, 8)))
        file.write(" ")
    file.write(flag)
    file.write("\n")


# 获取句子中tf-idf值最高的两个词
def get_highest_tf_idf_words(dic):
    tf_idf1 = 0
    tf_idf2 = 0
    word1 = ""
    word2 = ""
    for key in dic.keys():
        if dic[key] > tf_idf1:
            tf_idf1 = dic[key]
            word1 = key
    for key in dic.keys():
        if tf_idf2 < dic[key] < tf_idf1:
            tf_idf2 = dic[key]
            word2 = key
    return word1, word2


# 获取句子中tf-idf较高且存在向量表示的两个词
def get_highest_tf_idf_vec_words(dic, model):
    vec_words = []  # 存放存在向量表示词语的数组
    for key in dic.keys():
        try:
            vec = model[key]
            vec_words.append(key)
        except KeyError:
            continue

    if len(vec_words) < 2:
        return None, None, None, None

    tf_idf1 = 0
    tf_idf2 = 0
    word1 = ""
    word2 = ""
    for word in vec_words:
        if dic[word] > tf_idf1:
            tf_idf1 = dic[word]
            word1 = word
    for word in vec_words:
        if tf_idf2 < dic[word] < tf_idf1:
            tf_idf2 = dic[word]
            word2 = word

    return word1, word2, tf_idf1, tf_idf2


# softmax函数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


# 将组合权重进行转换
def transform_co_weight(co_weight_dic):
    min_co_weight = 10000000
    for co_weight in co_weight_dic.values():
        if min_co_weight > co_weight:
            min_co_weight = co_weight

    for word in co_weight_dic.keys():
        if co_weight_dic[word]/min_co_weight > 10:  # 设定一个阈值
            co_weight_dic[word] = 10
        else:
            co_weight_dic[word] = co_weight_dic[word]/min_co_weight

    return co_weight_dic


# 将每个句子中词语和tf-idf值和该tf-idf值对应的softmax结果对应起来
def helper(dict_values, softmax_weights):
    softmax_dic = {}
    for i in range(0, len(dict_values)):
        softmax_dic[dict_values[i]] = softmax_weights[i]
    return softmax_dic


# 在训练样本中写入每个句子对应的类别
def write_labels_to_vector_file():
    labeled_file = open("training_data/seg_sent_with_label")
    vector_file = open("testing_data/output.txt")
    cat_file = open("testing_data/svm_testing_data_using_sent_vec", 'w')
    lines = labeled_file.readlines()
    vector_lines = vector_file.readlines()
    for i in range(0, len(lines)):
        cat_line = ""
        if '0' in lines[i]:
            cat_line = vector_lines[i].replace("\n", "0")
        else:
            cat_line = vector_lines[i].replace("\n", "1")
        cat_file.write(cat_line)
        cat_file.write("\n")


# 找到同时出现在两个列表中的词语，返回一个列表
def get_same_word_in_lists(list1, list2):
    same_words = []
    for word1 in list1:
        for word2 in list2:
            if word1 == word2:
                same_words.append(word1)
    return same_words


# 通过编辑距离从list1和list2中获得和keyword语义相近的词语
def get_sim_words_from_ED(keyword, list1, list2):
    sim_words = []
    for word in list1:
        for c in unicode(keyword):
            if c in word and word != keyword:
                sim_words.append(word)
    for word in list2:
        for c in unicode(keyword):
            if c in word and word != keyword and word not in list1:
                sim_words.append(word)
    return sim_words


# 测试，查看某一个词对应的主题词和相似词
def test_sim_and_lda_words(word):
    lda = generate_gm_lda()
    dic_list = get_lda_topic_words(lda)
    model = gensim.models.Word2Vec.load(model_file_path)
    topic_words = get_topic_words(word, dic_list)
    sim_words_list = model.most_similar([unicode(word)], topn=30)
    sim_words = get_sim_words_list(sim_words_list)
    for topic_word in topic_words:
        print(topic_word)
    print("======================")
    for sim_word in sim_words:
        print(sim_word)
    print("======================")
    same_words = get_same_word_in_lists(topic_words, sim_words)
    if len(same_words) > 0:
        kl_div_dic = get_kl_div_dic_for_same_words(lda, same_words, word)
        for same_word in same_words:
            print(same_word)
    else:
        ED_sim_words = get_sim_words_from_ED(word, topic_words, sim_words)
        for word in ED_sim_words:
            print(word)


# 将通过word2vec得到的语义相似词tuple转化为list
def get_sim_words_list(sim_words_list):
    sim_words = []
    for sim_word in sim_words_list:
        sim_words.append(sim_word[0])
    return sim_words


positive_sent_path = "training_data/positive_sent_sample"
unlabeled_sent_path = "training_data/seg_sent_without_label"  # 用于训练词向量
labeled_sent_path = "training_data/seg_sent_with_label2"  # 用于生成句向量的标记文本
# 使用不同方式生成的句向量保存在不同的文件中
unique_sent_vec_path = "testing_data/unique_svm_testing_data.txt"
vec_sum_path = "svm_data/svm_word_sum_data.txt"
weighted_vec_sum_path = "testing_data/svm_testing_weighted_data.txt"
tfidf_weighted_vec_sum_path = "testing_data/svm_testing_weighted_data_with_tfidf_vec.txt"

train_word2vec_model(unlabeled_sent_path)
#get_most_tf_idf_for_sent(word2vec_training_file2)
#sentence2vec_with_word_sim(sent_path, vec_path)
#sentence2vec_with_adjacent_words(sent_path, vec_path)
#sentence2vec_with_word_avrg(sent_path, vec_path)
#sentence2vec_with_tf_idf(sent_path, vec_path)
#sentence2vec_with_tf_idf_vec(sent_path, vec_path)

# 新实验方法
#avrg_words_sum_for_each_sent(word2vec_training_file)
sentence2vec_with_w2v_sum(labeled_sent_path, vec_sum_path)
#sentence2vec_with_weight_w2v_sum(labeled_sent_path, vec_sum_path)
#write_labels_to_vector_file()

# 测试
#get_most_tf_idf_for_sent(unique_positive_sent_path)
#test_sim_and_lda_words("发电")







