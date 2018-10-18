# -*- encoding: utf-8 -*-

import gensim
from gensim.models import word2vec
from lda_test import generate_full_gm_lda
from lda_test import get_unique_word_topics
from lda_test import get_similar_words
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from word2vec_test import write_one_vec_to_file
from word2vec_test import softmax, helper

training_corpus_path = "training_data/seg_sent_without_label"  # 训练LDA主题模型的语料库
generated_corpus_path = "training_data/topic_based_corpus"  # LDA生成的用于训练word2vec的语料库
word2vec_model_save_path = "output_result/topic_based.model"  # 保存基于主题语料库的word2vec训练模型
word2vec_vector_save_path = "output_result/topic_based.vector"  # 保存基于主题语料库的word2vec训练向量


# 生成基于主题的语料库
def generate_topic_based_corpus(text_file_path, generated_file_path):
    lda = generate_full_gm_lda()
    sample_file = open(text_file_path, 'r')  # 读取语料库
    generated_corpus_file = open(generated_file_path, 'w')
    list = []

    for line in sample_file.readlines():
        list.append(line.replace("\n", ""))

    # 计算总共多少个不重复的词
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(list))
    unique_words = vectorizer.get_feature_names()
    num = 0

    # 对每个不重复的词在LDA中寻找其相似词，并将结果写入文档用于训练word2vec
    for unique_word in unique_words:
        try:
            ids = lda.id2word.token2id
            unique_word_id = ids[unicode(unique_word)]
            unique_word_topics = get_unique_word_topics(lda, unique_word_id)
            if len(unique_word_topics) == 1:
                topic_id = unique_word_topics[0][0]
                unique_word_prob = unique_word_topics[0][1]
                sim_words = get_similar_words(lda, topic_id, unique_word_id)
                write_sim_words_to_file(generated_corpus_file, sim_words)
                num += 1
            elif len(unique_word_topics) > 1:
                for topic in unique_word_topics:
                    topic_id = topic[0]
                    sim_words = get_similar_words(lda, topic_id, unique_word_id)
                    write_sim_words_to_file(generated_corpus_file, sim_words)
                    num += 1
            print("Finished num:"+str(num))
        except KeyError:
            continue


# 训练基于主题语料库的word2vec
def train_topic_based_word2vec(train_file_path):
    sentences = word2vec.Text8Corpus(train_file_path)
    model = word2vec.Word2Vec(sentences, sg=1, min_count=5, window=5, size=100)

    model.save(word2vec_model_save_path)
    model.wv.save_word2vec_format(word2vec_vector_save_path, binary=False)


# 将每个不重复词语相似的词作为一行写入文档
def write_sim_words_to_file(file, sim_words):
    for sim_word in sim_words:
        file.write(sim_word)
        file.write(" ")
    file.write("\n")


# 对在多个主题中均以较高概率出现的词语，找到其出现概率最高的主题
def most_prob_topic(unique_words_topics):
    most_prob = 0
    most_topic_id = 0
    for topic in unique_words_topics:
        topic_id = topic[0]
        topic_prob = topic[1]
        if topic_prob > most_prob:
            most_prob = topic_prob
            most_topic_id = topic_id
    return most_topic_id


# 使用词向量和作为句子特征向量
def sentence2vec_with_w2v_sum(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(word2vec_model_save_path)
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
        num = 0
        for word in words:
            try:
                vec = model[unicode(word)]
                sent_vec += vec
                num += 1
            except KeyError:
                continue
        print("Word sum:"+str(len(words)))
        print("Add word sum:"+str(num))
        write_one_vec_to_file(svm_training_file, sent_vec, cat_flag)
        i = i+1
        print("Finished " + str(i) + " sentences")


# 使用词向量的tf-idf加权和作为句子特征向量
def sentence2vec_with_weight_w2v_sum(text_file_path, vec_file_path):
    model = gensim.models.Word2Vec.load(word2vec_model_save_path)
    sample_file = open(text_file_path)
    svm_training_file = open(vec_file_path, "w")
    list = []  # 保存样本的数组，每个句子为一个元素，去除句子标签
    labeled_list = []  # 保存句子及其标签的数组
    flags = []  # 保存每个样本label的数组
    word_nums = []  # 保存每个句子的词语数量

    for line in sample_file.readlines():
        if line.find("0") != -1:
            flags.append("0")
        else:
            flags.append("1")
        list.append(line.replace(" 0\n", "").replace(" 1\n", ""))
        labeled_list.append(line)
        word_nums.append(len(line.replace(" 0\n", "").replace(" 1\n", "").split(" ")))

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
        #softmax_dic = helper(dict.values(), softmax_weights)  # 将tf-idf值与对应的softmax值使用字典进行关联,
        for word in dict.keys():
            try:
                vec = model[unicode(word)]
                #  计算加权和
                sent_vec += [dict[word]] * vec
            except KeyError:
                continue

        write_one_vec_to_file(svm_training_file, sent_vec, flags[i])
        print("Finished " + str(i) + " sentences")


labeled_sent_path = "training_data/seg_sent_with_label"  # 用于生成句向量的标记文本
vec_sum_path = "testing_data/topic_based_svm_data.txt"  # 保存句向量的文档
weighted_vec_sum_path = "testing_data/topic_based_weighted_svm_data.txt"
#generate_topic_based_corpus(training_corpus_path, generated_corpus_path)
train_topic_based_word2vec(generated_corpus_path)
#sentence2vec_with_w2v_sum(labeled_sent_path, vec_sum_path)
#sentence2vec_with_weight_w2v_sum(labeled_sent_path, weighted_vec_sum_path)

