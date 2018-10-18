# -*- encoding:utf-8 -*-

import codecs
import sys
from gensim import corpora
from gensim.models import LdaModel
from math import log

reload(sys)
sys.setdefaultencoding('utf-8')  # 并非没有用


#  使用正例样本训练LDA
def generate_gm_lda():
    text_file_path = "training_data/positive_sent_sample"  # 使用正例句子作为输入
    stop_words_file_path = "filter_words/stop_words"

    text_file = codecs.open(text_file_path, "r", encoding='utf8')
    stop_words_file = codecs.open(stop_words_file_path, 'r', encoding='utf8')
    list = []
    stop_words = []
    for stop_w in stop_words_file:
        stop_words.append(stop_w.replace("\r\n", ""))

    for line in text_file:
        line = remove_stop_words(line, stop_words)
        list.append(line.replace(" \r\n", "").split(" "))

    dic = corpora.Dictionary(list)  # 生成文档词典，每一个词与一个索引值对应
    corpus = [dic.doc2bow(text) for text in list]  # 词频统计，转换为空间向量格式
    ids = dic.token2id
    lda = LdaModel(corpus=corpus, id2word=dic, num_topics=20, alpha='auto')

    return lda


# 使用所有样本训练LDA
def generate_full_gm_lda():
    text_file_path = "training_data/seg_sent_without_label"  # 使用正负例句子作为输入
    stop_words_file_path = "filter_words/stop_words"

    text_file = codecs.open(text_file_path, "r", encoding="utf8")
    stop_words_file = codecs.open(stop_words_file_path, "r", encoding="utf8")
    list = []
    stop_words = []
    for stop_w in stop_words_file:
        stop_words.append(stop_w.replace("\r\n", ""))

    for line in text_file:
        line = remove_stop_words(line, stop_words)
        list.append(line.replace("\n", "").split(" "))

    dic = corpora.Dictionary(list)  # 生成文档词典，每一个词与一个索引值对应
    corpus = [dic.doc2bow(text) for text in list]  # 词频统计，转换为空间向量格式
    ids = dic.token2id
    lda = LdaModel(corpus=corpus, id2word=dic, num_topics=30, alpha='auto')

    return lda


# 获得各个主题的词语和其概率，返回一个字典的列表
def get_lda_topic_words(lda):
    result_list = lda.show_topics(num_topics=30, num_words=50)

    dic_list = []
    for tup in result_list:
        dic = {}
        pro_words = tup[1].split("+")
        for pro_word in pro_words:
            word = pro_word.split("*")
            dic[word[1].replace('"', "").replace(" ", "")] = word[0]
        dic_list.append(dic)

    return dic_list


# 测试LDA
def test_gm_lda():
    test_file_path = "training_data/positive_sent_sample"
    text_file_path = "training_data/seg_sent_without_label"
    stop_words_file_path = "filter_words/stop_words"

    text_file = codecs.open(test_file_path, "r", encoding='utf8')
    stop_words_file = codecs.open(stop_words_file_path, 'r', encoding='utf8')
    list = []
    stop_words = []
    for stop_w in stop_words_file:
        stop_words.append(stop_w.replace("\r\n", ""))

    for line in text_file:
        line = remove_stop_words(line, stop_words)
        list.append(line.replace(" \r\n", "").split(" "))

    '''
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(list))
    words = vectorizer.get_feature_names()
    weight = tf_idf.toarray()
    '''

    dic = corpora.Dictionary(list)  # 生成文档词典，每一个词与一个索引值对应
    corpus = [dic.doc2bow(text) for text in list]  # 词频统计，转换为空间向量格式
    ids = dic.token2id
    lda = LdaModel(corpus=corpus, id2word=dic, num_topics=20, alpha='auto', )

    result_list = lda.show_topics(num_topics=20, num_words=30)

    for tup in result_list:
        print(tup[1])
        print("============================")


# 去除停用词（一些与主题分类无关的词）
def remove_stop_words(one_line, stop_words):
    for stop_word in stop_words:
        if one_line.split(" ")[0] == stop_word:
            one_line = one_line.replace(stop_word+" ", "")
        else:
            one_line = one_line.replace(" "+stop_word+" ", " ")

    return one_line


# 找到该词出现概率最高的主题，并返回该主题的其他词语
def get_topic_words(word, dic_list):
    probability = 0
    word_list = []
    for dic in dic_list:
        if contain_key(dic.keys(), word):
            if float(dic[unicode(word)]) > probability:
                probability = float(dic[unicode(word)])
                word_list = dic.keys()

    return word_list


def contain_key(key_list, word):
    for key in key_list:
        if key == word:
            return True
    return False


# 根据LDA主题模型计算产业关键词和其语义相似词之间的KL散度，返回一个字典
def get_kl_div_dic_for_same_words(lda, words_list, keyword):
    ids = lda.id2word.token2id
    keyword_id = ids[unicode(keyword)]  # 关键词对应的id
    keyword_topics = lda.get_term_topics(word_id=keyword_id)
    kl_div_dic = {}  # 保存与关键词语义相似的词及关键词与语义相似词之间的KL散度
    for word in words_list:
        word_id = ids[unicode(word)]
        sim_word_topics = lda.get_term_topics(word_id=word_id)  # 与关键词相似的词所在主题
        if len(sim_word_topics) > 0:
            kl_div = cal_kl_divergence(keyword_topics, sim_word_topics)
            if kl_div != 0:
                kl_div_dic[word] = kl_div

    return kl_div_dic


# 在LDA中根据词语id找到每一个词所在的主题
def get_unique_word_topics(lda, unique_word_id):
    unique_word_topics = lda.get_term_topics(word_id=unique_word_id)
    return unique_word_topics


# 在LDA中根据词语拥有最大出现概率主题的id和该最大概率，得到与该词语相似的词语
def get_similar_words(lda, topic_id, unique_word_id):
    id2token = lda.id2word.id2token
    topic_terms = lda.get_topic_terms(topicid=topic_id, topn=100)
    unique_word_index = 0
    for i in range(0, 100):  # 找到该主题位于该主题中的位置
        term_id = topic_terms[i][0]
        if term_id == unique_word_id:
            unique_word_index = i
            break
    begin_index = 0
    end_index = 11
    sim_words = []
    if unique_word_index > 5:
        begin_index = unique_word_index-5
        end_index = unique_word_index+6
    for i in range(begin_index, end_index):  # 以窗口为5，找到与该词语距离最近的10个词语
        term_id = topic_terms[i][0]
        sim_words.append(id2token[term_id].replace("\n", ""))
    return sim_words


# 通过word2vec计算与产业关键词存在kl散度语义相似词的语义相似度
def get_sim_dic_for_same_words(model, word_list, keyword):
    sim_dic = {}
    for word in word_list:
        sim = model.similarity(unicode(keyword), unicode(word))
        sim_dic[word] = sim

    return sim_dic


# 计算KL散度
def cal_kl_divergence(keyword_topics, sim_word_topics):
    topic_dic = {}
    for topic in keyword_topics:
        for topic2 in sim_word_topics:
            if topic[0] == topic2[0]:
                topic_dic[topic] = topic2
    kl_div = 0
    for topic in topic_dic.keys():
        p = float(topic[1])
        q = float(topic_dic[topic][1])
        kl_div += p*log(p/q) + q*log(q/p)  # KL散度不具有对称性，此处将其转化为具有对称性的度量

    return kl_div/2


# 获得与keyword同一主题下某个相似词语概率
def get_topic_probability_for_word(lda, keyword, sim_word):
    ids = lda.id2word.token2id
    keyword_id = ids[unicode(keyword)]  # 关键词对应的id
    sim_word_id = ids[unicode(keyword)]  # 相似词对应的id
    keyword_topics = lda.get_term_topics(word_id=keyword_id)  # 关键词所在主题
    sim_word_topics = lda.get_term_topics(word_id=sim_word_id)  # 与关键词相似的词所在主题
    for topic in keyword_topics:
        for sim_topic in sim_word_topics:
            if topic[0] == sim_topic[0]:
                return float(sim_topic[1])