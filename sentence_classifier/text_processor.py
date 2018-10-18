# -*- encoding: utf-8 -*-

import sys
import random
import gensim
import numpy as np
from lda_test import generate_full_gm_lda, get_lda_topic_words, get_topic_words, generate_gm_lda
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from word2vec_research import prepare_data, get_all_dict, cal_ovm_sentence_sim_for_test, read_data, get_dissim_pair_dict
from helper import adjust_vec_dims
from sklearn.model_selection import train_test_split
from word_freq_calculator import cal_word_freq
from math import log, exp


reload(sys)
sys.setdefaultencoding("utf-8")

model_file_path = "output_result/up_down_stream.model"  # 词嵌入模型
text_file_path = "training_data/seg_sent_with_label2"  # 所有带标签的句子
train_text_path = "train_test_data/train_text"  # 训练集文本
test_text_path = "train_test_data/test_text"  # 测试集文本

train_vec_path = "train_test_data/train_vec"  # 训练集向量
test_vec_path = "train_test_data/test_vec"  # 测试集向量


# 将句子及其向量分别分为训练集（80%）和测试集（20%），并分别写入文件
def split_train_test_data(all_vecs_path, all_sents_path, dim, output_dir_path):
    data = np.loadtxt(all_vecs_path, dtype=float, delimiter=" ")
    training_data, labels = np.split(data, (dim,), axis=1)
    train_vec, test_vec, train_label, test_label = train_test_split(training_data, labels, train_size=0.8,
                                                                    random_state=1, shuffle=True)
    all_sents = []
    train_sents = [''] * len(train_vec)
    test_sents = [''] * len(test_vec)
    with open(all_sents_path, "r") as f:
        for line in f.readlines():
            all_sents.append(line)

    all_vecs = training_data.tolist()
    train_vecs = train_vec.tolist()
    test_vecs = test_vec.tolist()
    train_labels = train_label.tolist()
    test_labels = test_label.tolist()

    print("Splitting data...")
    for i in xrange(0, len(all_vecs)):  # 对于全集中的每个句子及其向量
        cur_vec = all_vecs[i]
        cur_sent = all_sents[i]
        train_indexes = contained_indexes_in_vecs(cur_vec, train_vecs)  # 根据该句子向量找到其在训练集向量中的索引
        test_indexes = contained_indexes_in_vecs(cur_vec, test_vecs)  # 根据该句子向量找到其在测试集向量中的索引
        for index in train_indexes:  # 在训练集和测试集对应位置填入当前向量
            train_sents[index] = cur_sent
        for index in test_indexes:
            test_sents[index] = cur_sent
        print("Finish splitting sentence:"+str(i))

    print("Finish splitting training and testing data.")

    # 将训练集测试集对应的句子和向量写入固定文件
    with open(output_dir_path+"/train_text", "w") as f:
        for train_sent in train_sents:
            f.write(train_sent)
    with open(output_dir_path+"/test_text", "w") as f:
        for test_sent in test_sents:
            f.write(test_sent)
    with open(output_dir_path+"/train_vec", "w") as f:
        for i in xrange(0, len(train_vecs)):
            write_one_vec_to_file(f, train_vecs[i], train_labels[i])
    with open(output_dir_path+"/test_vec", "w") as f:
        for i in xrange(0, len(test_vec)):
            write_one_vec_to_file(f, test_vecs[i], test_labels[i])


# 将句子向量分为训练集和测试集，并分别写入文件
def split_train_test_data_for_vecs(all_vecs_path, dim, output_dir_path):
    data = np.loadtxt(all_vecs_path, dtype=float, delimiter=" ")
    training_data, labels = np.split(data, (dim,), axis=1)
    train_vec, test_vec, train_label, test_label = train_test_split(training_data, labels, train_size=0.8,
                                                                    random_state=1, shuffle=True)
    train_vecs = train_vec.tolist()
    test_vecs = test_vec.tolist()
    train_labels = train_label.tolist()
    test_labels = test_label.tolist()

    with open(output_dir_path+"/train_vec", "w") as f:
        for i in xrange(0, len(train_vecs)):
            write_one_vec_to_file(f, train_vecs[i], train_labels[i])
    with open(output_dir_path+"/test_vec", "w") as f:
        for i in xrange(0, len(test_vec)):
            write_one_vec_to_file(f, test_vecs[i], test_labels[i])


# 计算vec在vecs中的下标，返回一个包含下标的列表
def contained_indexes_in_vecs(vec, vecs):
    index_list = []
    for i in xrange(0, len(vecs)):
        vec1 = vecs[i]
        if is_same_vec(vec, vec1):
            index_list.append(i)
    return index_list


# 判断vec和vec1是否相等，即每一个维度的数值都相等
def is_same_vec(vec, vec1):
    for i in xrange(0, len(vec)):
        if vec[i] != vec1[i]:
            return False
    return True


# 读取不重要的词语来在加权词向量时将这些词去除
def read_useless_words():
    useless_words = []
    with open("filter_words/moved_words", 'r') as f:
        for word in f.readlines():
            useless_words.append(word.replace("\n", ""))
    return useless_words


# 将句子通过词向量构成成向量
def sentences_to_vector(sent_path, vec_path):
    model = gensim.models.Word2Vec.load(model_file_path)
    sent_file = open(sent_path, "r")
    vec_file = open(vec_path, "w")
    list = []  # 保存样本的数组，每个句子为一个元素，去除句子标签
    labeled_list = []  # 保存句子及其标签的数组
    flags = []  # 保存每个样本label的数组
    # 统计训练集中正例和负例样本中词语的频率
    pos_word_freq_dic, neg_word_freq_dic, pos_word_avg, neg_word_avg = cal_word_freq(train_text_path)
    lda = generate_full_gm_lda()  # 训练lda模型
    # 获得样本对应的LDA主题建模的结果，列表中每个元素为一个字典，每一个字典对应一个主题，字典的key为词语，value为该词语在该主题下的概率
    topic_dic_list = get_lda_topic_words(lda)

    # 将训练集和测试集数据合并
    for line in sent_file.readlines():
        #if line not in labeled_list:
        if " 1\n" in line:
            flags.append(1)
        else:
            flags.append(0)
        list.append(line.replace(" 1\n", "").replace(" 0\n", ""))
        labeled_list.append(line)

    # 遍历所有句子，对每个句子得到其所有词语的tf-idf值，并计算tf-idf值经过softmax处理后的结果
    for i in range(0, len(list)):
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

        words = list[i].split()
        #print(list[i])
        for word in words:
            pos_word_freq = 0.0
            neg_word_freq = 0.0
            if word in pos_word_freq_dic.keys() and word in neg_word_freq_dic.keys():
                pos_word_freq = pos_word_freq_dic[unicode(word)]
                neg_word_freq = neg_word_freq_dic[unicode(word)]
            elif word in pos_word_freq_dic.keys() and word not in neg_word_freq_dic.keys():
                pos_word_freq = pos_word_freq_dic[unicode(word)]
            elif word in neg_word_freq_dic.keys() and word not in neg_word_freq_dic.keys():
                neg_word_freq = neg_word_freq_dic[unicode(word)]

            try:
                vec = model[unicode(word)]
                #  计算加权和
                temp = (pos_word_freq+1.0)/(neg_word_freq+1.0)
                if temp > 1.0:
                    ex = exp(pos_word_avg/(pos_word_freq+1.0))
                    # 出现在训练集正例次数多于负例的词语，很可能为和产业相关的词语
                    # 通过LDA找到同主题词语

                    '''
                    if len(word) != 3:
                        topic_words = get_topic_words(word, topic_dic_list)
                        if len(topic_words) != 0:
                            sim_words_list = model.most_similar([unicode(word)], topn=30)
                            sim_words = get_sim_words_list(sim_words_list)
                            # 在lda主题分类中以及word2vec相似词中找到同样的词语作为产业关键词的语义相似词
                            same_words = get_same_word_in_lists(topic_words, sim_words)
                            for sw in same_words:
                                try:
                                    vec = model[unicode(sw)]
                                    sent_vec += 0.5 * vec
                                except KeyError:
                                    continue
                 '''

                else:
                    ex = exp(neg_word_avg/(neg_word_freq+1.0))

                sent_vec += ex * log(temp) * vec

                #print(word+" ex:"+str(round(ex, 6))+" temp:"+str(round(log(temp), 6)))
            except KeyError:
                continue

        write_one_vec_to_file(vec_file, sent_vec, flags[i])
        print("Finish sentence:"+str(i))


# 使用句子中词语的tf-idf加权词向量和表示句子
def sentence2vec_with_tfidf_weight(sent_path, vec_path):
    model = gensim.models.Word2Vec.load(model_file_path)
    sent_file = open(sent_path)
    vec_file = open(vec_path, "w")
    list = []  # 保存样本的数组，每个句子为一个元素，去除句子标签
    labeled_list = []  # 保存句子及其标签的数组
    flags = []  # 保存每个样本label的数组

    for line in sent_file.readlines():
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

        #softmax_weights = softmax(dict.values())  # 将一个句子的tf-idf值进行softmax处理
        #softmax_dic = helper(dict.values(), softmax_weights)  # 将tf-idf值与对应的softmax值使用字典进行关联
        for word in dict.keys():
            try:
                vec = model[unicode(word)]
                #  计算加权和
                sent_vec += dict[word] * vec
            except KeyError:
                continue

        write_one_vec_to_file(vec_file, sent_vec, flags[i])
        print("Finished " + str(i) + " sentences")


# ============================================== 一些功能函数
# softmax函数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


# 将每个句子中词语和tf-idf值和该tf-idf值对应的softmax结果对应起来
def helper(dict_values, softmax_weights):
    softmax_dic = {}
    for i in range(0, len(dict_values)):
        softmax_dic[dict_values[i]] = softmax_weights[i]
    return softmax_dic


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


# 将通过word2vec得到的语义相似词tuple转化为list
def get_sim_words_list(sim_words_list):
    sim_words = []
    for sim_word in sim_words_list:
        sim_words.append(sim_word[0])
    return sim_words


# 对于每个句子只用一个向量表示的情况下，将其向量写入文件作为svm训练集
def write_one_vec_to_file(file, vec, flag):
    for num in vec:
        file.write(str(round(num, 8)))
        file.write(" ")
    file.write(str(flag).replace("[", "").replace("]", "").replace(".0", ""))
    file.write("\n")


# =====================================================提取产业关键词函数
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


# ===============================================================
# 读取测试集中的句子，并以tf-idf为权重的element-wise addition操作生成向量
def generate_test_sent_vec(test_sent_path):
    model = gensim.models.Word2Vec.load(model_file_path)
    sample_file = open(test_sent_path, 'r')
    list = []  # 保存样本的数组，每个句子为一个元素，去除句子标签
    labeled_list = []  # 保存句子及其标签的数组
    flags = []  # 保存每个样本label的数组
    vecs = []

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
        vecs.append(sent_vec)

    return labeled_list, vecs


# 计算测试集中的句子与训练集中句子相似度
def cal_sim_between_train_test(train_sent_path, train_ori_vec_path, test_sent_path, test_vec_path):
    all_sents, p_sents, n_sents, p_vecs, n_vecs, p_all_dict, n_all_dict, dissim_pos_neg_pairs, dissim_neg_pos_pairs = prepare_data(
        train_sent_path, train_ori_vec_path
    )

    test_sents = []
    test_vecs = []
    with open(test_sent_path, 'r') as f:
        for line in f.readlines():
            test_sents.append(line)
    with open(test_vec_path, 'r') as f:
        for line in f.readlines():
            test_vecs.append(np.fromstring(line.replace(" 1\n", "").replace(" 0\n", ""), dtype='float', sep=' '))

    most_sim_dict = {}  # key为测试集句子index，value为该测试集句子在训练集中相似度最高句子的index(分正负)
    test_labels = []  # 测试集标签
    test_adjusted_vecs = []

    # 计算测试集中每一个句子与训练集中句子的相似度
    for i in xrange(0, len(test_sents)):
        if ' 1\n' in test_sents[i]:
            test_labels.append(1)
        else:
            test_labels.append(0)

        print(i)
        print(test_sents[i].replace("\n", ""))
        p_most_sim_dic = cal_ovm_sentence_sim_for_test(test_vecs[i], sim_vecs=p_vecs)
        n_most_sim_dic = cal_ovm_sentence_sim_for_test(test_vecs[i], sim_vecs=n_vecs)

        p_avg_sim = average_sim(p_most_sim_dic.keys())
        n_avg_sim = average_sim(n_most_sim_dic.keys())

        print("Pos sim:"+str(round(p_avg_sim, 6))+" Neg sim:"+str(round(n_avg_sim, 6)))

        if (p_avg_sim > n_avg_sim and " 0\n" in test_sents[i]) or (
            n_avg_sim > p_avg_sim and " 1\n" in test_sents[i]
        ):
            for sim in p_most_sim_dic.keys():
                p_index = p_most_sim_dic[sim]
                print(str(sim)+" "+p_sents[p_index].replace("\n", ""))
            print("--------------------")
            for sim in n_most_sim_dic.keys():
                n_index = n_most_sim_dic[sim]
                print(str(sim)+" "+n_sents[n_index].replace("\n", ""))

        print("------------------------------------------")

    print("Finish similarity cal.")
    '''
    for i in xrange(0 ,len(test_vecs)):
        vec = test_vecs[i]
        adjusted_vec = adjust_vec_dims(vec, common_most_dims, 28)
        test_adjusted_vecs.append(adjusted_vec)

    # 测试集调整后的向量写入文件
    with open("using_sw_and_half_freq3/test_adjusted_data", 'w') as f:
        for i in xrange(0, len(test_adjusted_vecs)):
            write_one_vec_to_file(f, test_adjusted_vecs[i], test_labels[i])
    '''


# 计算平均相似度
def average_sim(list):
    sim_sum = 0.0
    for sim in list:
        sim_sum += sim
    return sim_sum/len(list)


# 统计通过对比测试集中样本与训练集中样本的相似度有多少测试集样本类别错误
def help_me():
    test_labels = []
    with open("using_sw_and_half_freq3/test_vec", "r") as f:
        for line in f.readlines():
            if " 1\n" in line:
                test_labels.append(1)
            else:
                test_labels.append(0)

    train_indexes = []
    with open("using_sw_and_half_freq3/test_train_sim_index", "r") as f:
        for line in f.readlines():
            train_indexes.append(int(line.replace("\n", "").split(" ")[1]))

    train_labels = []
    for index in train_indexes:
        if index > 0:
            train_labels.append(1)
        else:
            train_labels.append(0)

    right_num = 0.0
    for i in xrange(0, len(test_labels)):
        if test_labels[i] == train_labels[i]:
            right_num += 1

    print("Accuracy:"+str(right_num/len(test_labels)))


# 计算两向量余弦相似度
def cos_sim(vec1, vec2):
    dot_pro = 0
    norm1 = 0
    norm2 = 0
    for i in xrange(0, len(vec1)):
        dot_pro += vec1[i] * vec2[i]
        norm1 += vec1[i]**2
        norm2 += vec2[i]**2
    cos = dot_pro / (norm1*norm2)**0.5
    return 0.5 + 0.5*cos  # 结果归一化


#########################################################
# 读取测试集数据（文本、向量）
def read_test_data():
    test_sents = []
    test_vecs = []
    with open("using_sw_and_half_freq3/test_text", "r") as f:
        for line in f.readlines():
            test_sents.append(line)
    with open("using_sw_and_half_freq3/test_vec", "r") as f:
        for line in f.readlines():
            test_vecs.append(np.fromstring(line.replace(" 1\n", "").replace(" 0\n", ""), dtype='float32', sep=' '))
    return test_sents, test_vecs


# 计算单个句子和多个句子的相似度，并返回相似度最低的句子
# neg_idxes和pos_idxes分别为训练集中，与正例最不相似的负例句子索引，和与负例句子最不相似的正例句子索引
def cal_sim_ovo(test_vec, neg_idxes, pos_idxes, n_vecs, p_vecs):
    sim = 1
    lowest_sim_id = 0
    for idx in neg_idxes:
        cur_vec = n_vecs[idx]
        cur_vec = np.fromstring(cur_vec, dtype='float32', sep=' ')
        cur_sim = cos_sim(cur_vec, test_vec)
        if cur_sim < sim:
            sim = cur_sim
            lowest_sim_id = idx
    sim2 = 1
    lowest_sim_id2 = 0
    for idx in pos_idxes:
        cur_vec = p_vecs[idx]
        cur_vec = np.fromstring(cur_vec, dtype='float32', sep=' ')
        cur_sim = cos_sim(cur_vec, test_vec)
        if cur_sim < sim:
            sim2 = cur_sim
            lowest_sim_id = idx
    return sim, lowest_sim_id, sim2, lowest_sim_id2


# 计算test_vec与vecs中向量的平均相似度
def cal_avg_sim(test_vec, vecs):
    sim_sum = 0.0
    for vec in vecs:
        vec = np.fromstring(vec, dtype='float32', sep=' ')
        sim_sum += cos_sim(test_vec, vec)
    return sim_sum/len(vecs)


# 计算一个句向量和多组句向量的最大平均相似度
def cal_sim_ovm(test_vec, neg_pos_dict, pos_neg_dict, n_vec, p_vecs):
    most_pos_sim = 0
    most_neg_sim = 0
    most_pos_sim_key = 0  # 正例最大平均相似度在字典中的key
    most_neg_sim_key = 0  # 负例最大平均相似度在字典中的key
    for key in neg_pos_dict.keys():
        pos_idxes = neg_pos_dict[key]
        vecs = []
        for i in pos_idxes:
            vecs.append(p_vecs[i])
        arg_sim = cal_avg_sim(test_vec, vecs)
        if arg_sim > most_pos_sim:
            most_pos_sim = arg_sim
            most_pos_sim_key = key
    for key in pos_neg_dict.keys():
        neg_idxes = pos_neg_dict[key]
        vecs = []
        for i in neg_idxes:
            vecs.append(n_vec[i])
        arg_sim = cal_avg_sim(test_vec, vecs)
        if arg_sim > most_neg_sim:
            most_neg_sim = arg_sim
            most_neg_sim_key = key
    return most_pos_sim, most_pos_sim_key, most_neg_sim, most_neg_sim_key


# 比较测试集与训练集的相似度新算法
def cal_sim_between_tr_te():
    # 读取所需有关训练集的所有数据
    p_vecs, n_vecs, neg_pos_sents_pairs = read_data(0)
    p_vecs, n_vecs, pos_neg_sents_pairs = read_data(1)

    # 将不相似句子对转化为字典，key为负例句子的index，value为正例句子index列表
    neg_pos_dict = get_dissim_pair_dict(neg_pos_sents_pairs)
    pos_neg_dict = get_dissim_pair_dict(pos_neg_sents_pairs)
    p_all_dict = get_all_dict(0)
    n_all_dict = get_all_dict(1)
    p_sents = []
    n_sents = []
    with open("pos_neg_vecs/pos_sents", "r") as f:
        for line in f.readlines():
            p_sents.append(line.replace("\n", ""))
    with open("pos_neg_vecs/neg_sents", "r") as f:
        for line in f.readlines():
            n_sents.append(line.replace("\n", ""))

    # 读取测试集数据
    test_sents, test_vecs = read_test_data()
    right_num = 0.0

    neg_idxes = []
    pos_idxes = []
    neg_pos_sub_dict = {}
    pos_neg_sub_dict = {}
    for key in neg_pos_dict.keys():
        if len(neg_pos_dict[key]) >= 5:
            neg_idxes.append(key)
            neg_pos_sub_dict[key] = neg_pos_dict[key]
    for key in pos_neg_dict.keys():
        if len(pos_neg_dict[key]) >= 5:
            pos_idxes.append(key)
            pos_neg_sub_dict[key] = pos_neg_dict[key]

    for i in xrange(0, len(test_sents)):
        test_vec = test_vecs[i]
        test_sent = test_sents[i]

        print("Sentence:"+str(i))
        #print(test_sent.replace("\n", ""))
        sim, neg_id, sim2, pos_id = cal_sim_ovo(test_vec, neg_idxes, pos_idxes, n_vecs, p_vecs)
        most_pos_sim, most_pos_sim_key, most_neg_sim, most_neg_sim_key = cal_sim_ovm(test_vec, neg_pos_sub_dict, pos_neg_sub_dict, n_vecs, p_vecs)
        if most_pos_sim > most_neg_sim and sim < sim2:
            if " 1\n" in test_sent:
                right_num += 1
            else:  # 若该句判断为负例
                print(test_sent.replace("\n", ""))
                print("Pos sim:"+str(most_pos_sim)+" Neg sim"+str(most_neg_sim))
                neg_ids = pos_neg_dict[most_neg_sim_key]
                pos_ids = neg_pos_dict[most_pos_sim_key]
                for id in neg_ids:
                    print(n_sents[id])
                print("-----------------")
                for id in pos_ids:
                    print(p_sents[id])
            print("Pos")
        elif most_pos_sim < most_neg_sim and sim > sim2:
            if " 0\n" in test_sent:
                right_num += 1
            else:
                print(test_sent.replace("\n", ""))
                print("Pos sim:" + str(most_pos_sim) + " Neg sim" + str(most_neg_sim))
                neg_ids = pos_neg_dict[most_neg_sim_key]
                pos_ids = neg_pos_dict[most_pos_sim_key]
                for id in neg_ids:
                    print(n_sents[id])
                print("-----------------")
                for id in pos_ids:
                    print(p_sents[id])
            print("Neg")
        print("-----------------------")
    print("Accuracy:"+str(right_num/len(test_sents)))


train_sent_path = "using_sw_and_half_freq3/train_text"  # 训练集句子
test_sent_path = "using_sw_and_half_freq3/test_text"  # 测试集句子

all_sent_path = "training_data/seg_sent_with_label2"
all_vec_path = "svm_data/svm_word_sum_tfidf_data.txt"

# 使用yelp数据集的相关文件路径
yelp_all_vecs_path = "yelp_reviews/yelp_review_1/using_sw_and_half_freq3"
yelp_all_sents_path = "yelp_reviews/yelp_review_1/yelp_review2"
output_dir_path = "yelp_train_test"

#sentence2vec_with_tfidf_weight(all_sent_path, all_vec_path)
sentences_to_vector(all_sent_path, all_vec_path)
#split_train_test_data(yelp_all_vecs_path, yelp_all_sents_path, 300, output_dir_path)
split_train_test_data_for_vecs(yelp_all_vecs_path, 300, output_dir_path)
#cal_sim_between_train_test(train_sent_path, train_vec_path, test_sent_path, test_vec_path)
#cal_sim_between_tr_te()

