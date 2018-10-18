# -*- encoding: utf-8 -*-

import numpy as np
import sys
from helper import analyze_gap_dims
from helper import adjust_vec_dims

reload(sys)
sys.setdefaultencoding('utf-8')  # 并非没有用

w2v_sum_file_path = "svm_data/svm_word_sum_data.txt"
w2v_all_features_file_path = "svm_data/svm_all_features_data.txt"
w2v_sum_topic_file_path = "svm_data/svm_word_sum_topic_data.txt"
w2v_adjusted_vec_file_path = "svm_data/svm_adjusted_vec_data.txt"  # 不重要维度被设为0的句向量样本
w2v_opt_adj_vec_file_path = "svm_data/svm_opt_adj_vec_data.txt"
w2v_model_file = "output_result/up_down_stream.model"

text_file_path = "training_data/seg_sent_with_label2"

sim_sents_file_path = "training_data/sim_sents"  # 保存与训练集中每一个不重复的正例句子相似度最低的负例句子
sim_sents_index_file_path = "training_data/sim_sents_index"  # 保存上述不相似句子对在训练集中对应的索引

sim_sents_file_path2 = "training_data/sim_sents2"  # 保存与训练集中每一个不重复的负例句子以及与其相似度最低的正例句子
sim_sents_index_file_path2 = "training_data/sim_sents_index2"  # 保存上述不相似句子对在训练集中对应的索引



def compare_dim():
    w2v_sum_file = open(w2v_sum_file_path, 'r')
    w2v_sum_topic_file = open(w2v_sum_topic_file_path, 'r')
    text_file = open(text_file_path, 'r')

    w2v_sum_vecs = []
    w2v_sum_topic_vecs = []
    texts = []

    dim_diffs = []  # 记录句向量每一维差值
    dim_avg_diffs = []  # 记录每个句子向量的平均差值
    big_diff_dims = []  # 记录每个句子比平均差值大的向量维度

    # 加载数据
    for line in w2v_sum_file.readlines():
        if " 1\n" in line:
            w2v_sum_vecs.append(line.replace(" 1\n", ""))
    for line in w2v_sum_topic_file.readlines():
        if " 1\n" in line:
            w2v_sum_topic_vecs.append(line.replace(" 1\n", ""))
    for line in text_file.readlines():
        texts.append(line.replace(" \n", ""))

    # 逐句计算句向量每一维的差值与该句所有维的平均差值
    for i in xrange(0, len(w2v_sum_vecs)):
        vectors = w2v_sum_vecs[i].split(' ')
        topic_vectors = w2v_sum_topic_vecs[i].split(' ')
        sent_dim_diff = []
        diff_sum = 0
        for j in xrange(0, len(vectors)):
            diff = float(topic_vectors[j]) - float(vectors[j])
            diff_sum += abs(diff)
            sent_dim_diff.append(diff)
        dim_avg_diffs.append(diff_sum / 100)
        dim_diffs.append(sent_dim_diff)

    # 对每个句向量统计变化比均值大的维数
    for i in xrange(0, len(w2v_sum_vecs)):
        dim_diff_list = dim_diffs[i]  # 该句子每一维的差距
        dim_avg_diff = dim_avg_diffs[i]  # 该句子的维度平均差距
        dim_big_diff = []  # 距离每个句子中比平均维度差要大的维度数
        for j in xrange(0, len(dim_diff_list)):
            if abs(dim_diff_list[j]) > dim_avg_diff:
                dim_big_diff.append(j)
        big_diff_dims.append(dim_big_diff)

    # 对原向量在差值较大的维度上进行改变
    changed_vecs = []  # 保存改变后的句子向量
    for i in xrange(0, len(big_diff_dims)):
        sent_diff_dims = big_diff_dims[i]
        vec = w2v_sum_topic_vecs[i].split()
        if(len(sent_diff_dims)) == 0:  # 对于不需要改变的则直接加入
            changed_vecs.append(vec)
            continue
        sent_dim_diff = dim_diffs[i]
        for index in sent_diff_dims:
            vec[index] = str(float(vec[index]) + sent_dim_diff[index] * 2)
        changed_vecs.append(vec)

    return changed_vecs


# 生成增强后的数据样本
def generate_enhancing_data():
    ehc_data_path = "svm_data/svm_enhanced_topic_data.txt"
    ehc_data_file = open(ehc_data_path, 'w')
    topic_data_path = "svm_data/svm_word_sum_topic_data.txt"
    changed_vecs = compare_dim()
    i = 0
    with open(topic_data_path, 'r') as f:
        for line in f.readlines():
            if ' 1\n' in line:
                write_vec_to_file(ehc_data_file, changed_vecs[i], 1)
                i += 1
            else:
                ehc_data_file.write(line)


# 将句子特征向量写入文件
def write_vec_to_file(file, vecs, flag=None):
    if flag is None:
        for num in vecs:
            file.write(str(num)+" ")
        file.write("\n")
    else:
        for num in vecs:
            file.write(str(num)+" ")
        file.write(str(flag)+"\n")


# ====================================
# 读取训练集中的正负例句子和对应向量以及测试集中的正负例句子和对应向量
def read_sents_and_vecs(train_sent_path, train_vec_path, test_sent_path, test_vec_path):
    train_p_vecs = []  # 训练集中的正例向量
    train_n_vecs = []  # 训练集中的负例向量
    train_p_sents = []  # 训练集中的正例句子
    train_n_sents = []  # 训练集中的负例句子
    test_p_vecs = []  # 测试集中的正例向量
    test_n_vecs = []  # 测试集中的负例向量
    test_p_sents = []  # 测试集中的正例句子
    test_n_sents = []  # 测试集中的负例句子
    if train_sent_path is not None:
        train_sents_file = open(train_sent_path, 'r')
    if test_sent_path is not None:
        test_sents_file = open(test_sent_path, 'r')
    train_vecs_file = open(train_vec_path, 'r')
    test_vecs_file = open(test_vec_path, 'r')

    train_vec_lines = train_vecs_file.readlines()
    if train_sent_path is not None:
        train_sent_lines = train_sents_file.readlines()

    for i in xrange(0, len(train_vec_lines)):
        if " 5\n" in train_vec_lines[i] and train_vec_lines[i] not in train_p_vecs:
            vec = np.fromstring(train_vec_lines[i].replace(" 1\n", ""), dtype='float32', sep=' ')
            train_p_vecs.append(vec)
            if train_sent_path is not None:
                train_p_sents.append(train_sent_lines[i])
        elif " 1\n" in train_vec_lines[i] and train_vec_lines[i] not in train_n_vecs:
            vec = np.fromstring(train_vec_lines[i].replace(" 0\n", ""), dtype='float32', sep=' ')
            train_n_vecs.append(vec)
            if train_sent_path is not None:
                train_n_sents.append(train_sent_lines[i])
        print("Finish loading train sentence:"+str(i))

    test_vec_lines = test_vecs_file.readlines()
    if test_sent_path is not None:
        test_sent_lines = test_sents_file.readlines()

    for i in xrange(0, len(test_vec_lines)):
        if " 5\n" in test_vec_lines[i] and test_vec_lines[i] not in test_p_vecs:
            vec = np.fromstring(test_vec_lines[i].replace(" 1\n", ""), dtype='float32', sep=' ')
            test_p_vecs.append(vec)
            if test_sent_path is not None:
                test_p_sents.append(test_sent_lines[i])
        elif " 1\n" in test_vec_lines[i] and test_vec_lines[i] not in test_n_vecs:
            vec = np.fromstring(test_vec_lines[i].replace(" 0\n", ""), dtype='float32', sep=' ')
            test_n_vecs.append(vec)
            if test_sent_path is not None:
                test_n_sents.append(test_sent_lines[i])
        print("Finish loading test sentence:"+str(i))

    return train_p_vecs, train_p_sents, train_n_vecs, train_n_sents, test_p_vecs, test_p_sents, test_n_vecs, test_n_sents


# 计算所有句子之间的语义相似度
# 参数flag为0时表示计算与每一个正例句子最不相似的负例句子，flag为1时表示计算与每一个负例句子最不相似的正例句子
def cal_all_sentences_sim(train_sent_path, train_vec_path, test_sent_path, test_vec_path, flag, out_index_path1, out_index_path2):
    train_p_vecs, train_p_sents, train_n_vecs, train_n_sents, test_p_vecs, test_p_sents, test_n_vecs, test_n_sents = read_sents_and_vecs(
        train_sent_path, train_vec_path, test_sent_path, test_vec_path
    )

    # 将训练集和测试集的正负例句子和向量分别合并

    all_p_sents = []
    all_p_vecs = []
    all_n_sents = []
    all_n_vecs = []
    for i in xrange(0, len(train_p_vecs)):
        if " 5\n" in train_p_vecs[i] and train_p_vecs[i] not in all_p_vecs:
            all_p_sents.append(train_p_sents[i])
            all_p_vecs.append(train_p_vecs[i])
    for i in xrange(0, len(train_n_vecs)):
        if " 1\n" in train_n_vecs[i] and train_n_vecs[i] not in all_n_vecs:
            all_n_sents.append(train_n_sents[i])
            all_n_vecs.append(train_n_vecs[i])
    for i in xrange(0, len(test_p_sents)):
        if " 5\n" in test_p_vecs[i] and test_p_vecs[i] not in all_p_vecs:
            all_p_sents.append(test_p_sents[i])
            all_p_vecs.append(test_p_vecs[i])
    for i in xrange(0, len(test_n_vecs)):
        if " 1\n" in test_n_vecs[i] and test_n_vecs[i] not in all_n_vecs:
            all_n_sents.append(test_n_sents[i])
            all_n_vecs.append(test_n_vecs[i])

    if flag == 0:
        r = open(sim_sents_file_path, 'w')
        l = open(out_index_path1, 'w')
    else:
        r = open(sim_sents_file_path2, 'w')
        l = open(out_index_path2, 'w')

    if flag == 0:
        for i in xrange(0, len(train_p_vecs)):  # 比较训练集内每一个正例句子与训练集中每一个负例句子的相似度
            vec = train_p_vecs[i]
            print(i)
            most_sim_idx, least_sim_idx, most_sim, least_sim = cal_ovm_sentence_sim(vec, dissim_vecs=train_n_vecs)
            print(train_p_sents[i].replace("\n", ""))
            print(train_n_sents[least_sim_idx].replace("\n", "")+" Sim:%.4f" % least_sim)
            r.write(train_p_sents[i])
            r.write(train_n_sents[least_sim_idx])
            r.write("\n")
            l.write(str(i)+" "+str(least_sim_idx)+"\n")
            print("======================")
    elif flag == 1:
        for i in xrange(0, len(train_n_vecs)):  # 比较训练集内每一个负例句子与训练集中每一个正例句子的相似度
            vec = train_n_vecs[i]
            print(i)
            most_sim_idx, least_sim_idx, most_sim, least_sim = cal_ovm_sentence_sim(vec, dissim_vecs=train_p_vecs)
            print(train_n_sents[i].replace("\n", ""))
            print(train_p_sents[least_sim_idx].replace("\n", "")+" Sim:%.4f" % least_sim)
            r.write(train_n_sents[i])
            r.write(train_p_sents[least_sim_idx])
            r.write("\n")
            l.write(str(i)+" "+str(least_sim_idx)+"\n")
            print("======================")


# 将训练集、测试集、全集（训练集、测试集）的正负向量写入文件中
def write_vecs_to_file(train_sent_path, train_vec_path, test_sent_path, test_vec_path):
    train_p_vecs, train_p_sents, train_n_vecs, train_n_sents, test_p_vecs, test_p_sents, test_n_vecs, test_n_sents = read_sents_and_vecs(
        train_sent_path, train_vec_path, test_sent_path, test_vec_path
    )

    # 将训练集和测试集的正负例句子和向量分别合并
    all_p_sents = []
    all_p_vecs = []
    all_n_sents = []
    all_n_vecs = []
    for i in xrange(0, len(train_p_sents)):
        if " 1\n" in train_p_sents[i] and train_p_sents[i] not in all_p_sents:
            all_p_sents.append(train_p_sents[i])
            all_p_vecs.append(train_p_vecs[i])
    for i in xrange(0, len(train_n_sents)):
        if " 0\n" in train_n_sents[i] and train_n_sents[i] not in all_n_sents:
            all_n_sents.append(train_n_sents[i])
            all_n_vecs.append(train_n_vecs[i])
    for i in xrange(0, len(test_p_sents)):
        if " 1\n" in test_p_sents[i] and test_p_sents[i] not in all_p_sents:
            all_p_sents.append(test_p_sents[i])
            all_p_vecs.append(test_p_vecs[i])
    for i in xrange(0, len(test_n_sents)):
        if " 0\n" in test_n_sents[i] and test_n_sents[i] not in all_n_sents:
            all_n_sents.append(test_n_sents[i])
            all_n_vecs.append(test_n_vecs[i])

    with open("pos_neg_vecs/pos_vecs", 'w') as f:
        for vec in train_p_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/neg_vecs", 'w') as f:
        for vec in train_n_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/test_pos_vecs", 'w') as f:
        for vec in test_p_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/test_neg_vecs", 'w') as f:
        for vec in test_n_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/all_pos_vecs", 'w') as f:
        for vec in all_p_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/all_neg_vecs", 'w') as f:
        for vec in all_n_vecs:
            write_vec_to_file(f, vec)
    with open("pos_neg_vecs/pos_sents", 'w') as f:
        for p_sent in train_p_sents:
            f.write(p_sent)
    with open("pos_neg_vecs/neg_sents", 'w') as f:
        for n_sent in train_n_sents:
            f.write(n_sent)
    with open("pos_neg_vecs/test_pos_sents", 'w') as f:
        for p_sent in test_p_sents:
            f.write(p_sent)
    with open("pos_neg_vecs/test_neg_sents", 'w') as f:
        for n_sent in test_n_sents:
            f.write(n_sent)


# 对于一个正例句子，在sim_vecs中计算与其相似度最高的句子，在dissim_vecs中计算与其相似度最低的句子
def cal_ovm_sentence_sim(vec, sim_vecs=None, dissim_vecs=None):
    most_sim = 0
    most_idx = 0
    least_sim = 1
    least_idx = 0
    i = 0
    if sim_vecs is not None:
        for temp_vec in sim_vecs:
            if is_equal_vector(vec, temp_vec):  # 在同标签的样本中比较相似度最高的样本需要跳过和自身相比
                i += 1
                continue
            sim = cos_sim(vec, temp_vec)
            if sim > most_sim and sim != 1:
                most_sim = sim
                most_idx = i
            i += 1
        i = 0
    if dissim_vecs is not None:
        for temp_vec in dissim_vecs:
            sim = cos_sim(vec, temp_vec)
            if sim < least_sim:
                least_sim = sim
                least_idx = i
            i += 1
    return most_idx, least_idx, most_sim, least_sim


# 计算测试集中的句子与训练集中最相似以及最不相似的一组句子
def cal_ovm_sentence_sim_for_test(vec, sim_vecs=None, dissim_vecs=None):
    most_sim = 0
    most_idx = 0
    least_sim = 1
    least_idx = 0
    i = 0
    dic = {}  # 用来存放索引和对应的相似度
    if sim_vecs is not None:
        for temp_vec in sim_vecs:
            if is_equal_vector(vec, temp_vec):  # 在同标签的样本中比较相似度最高的样本需要跳过和自身相比
                i += 1
                continue
            sim = cos_sim(vec, temp_vec)
            dic[sim] = i
            i += 1
        sorted_key = sorted(dic.keys(), reverse=True)
        most_sim_dic = {}
        for key in sorted_key:
            if len(most_sim_dic) < 5:
                most_sim_dic[key] = dic[key]
        i = 0
    if dissim_vecs is not None:
        for temp_vec in dissim_vecs:
            sim = cos_sim(vec, temp_vec)
            if sim < least_sim:
                least_sim = sim
                least_idx = i
            i += 1
    return most_sim_dic


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


# 判断两个向量是否每一维都相同
def is_equal_vector(vec1, vec2):
    for i in xrange(0, len(vec1)):
        if vec1[i] == vec2[i]:
            continue
        else:
            return False
    return True


# 移除字典中相似度最小的元素
def remove_smallest(dic):
    smallest = 1
    smallest_idx = 0
    for idx in dic.keys():
        sim = dic[idx]
        if sim < smallest:
            smallest = sim
            smallest_idx = idx
    dic.pop(smallest_idx)
    return dic


# ==================================
# 读取数据，返回不重复的正例句子向量、负例句子向量以及与正例句子相似度最低的负例句子index对
# flag取0和1，取0时读取训练集中不重复的正例句子与其最不相似的负例句子index，为1则相反
def read_data(flag):
    pos_vecs_file_path = "pos_neg_vecs/pos_vecs"  # 用于保存不重复的正例句子对应的句向量
    neg_vecs_file_path = "pos_neg_vecs/neg_vecs"  # 用于保存不重复的负例句子对应的句向量
    n_vecs = []  # 负例句子向量
    p_vecs = []  # 正例句子向量
    dissim_sents_pair = []  # 最不相似句子对的index

    # 读取不重复的正例和负例向量
    with open(pos_vecs_file_path, 'r') as p:
        for line in p.readlines():
            p_vecs.append(line.replace("\n", ""))

    with open(neg_vecs_file_path, 'r') as n:
        for line in n.readlines():
            n_vecs.append(line.replace("\n", ""))
    # 以上为公共部分

    if flag == 0:  # 读取不重复的正例句子及与其最不相似的负例句子index
        with open(sim_sents_index_file_path, 'r') as s:
            for line in s.readlines():
                triple = line.replace("\n", "").split(" ")
                pair = (int(triple[0]), int(triple[1]))
                dissim_sents_pair.append(pair)
    elif flag == 1:  # 读取不重复的负例句子及与其最不相似的正例句子index
        with open(sim_sents_index_file_path2, 'r') as s:
            for line in s.readlines():
                triple = line.replace("\n", "").split(" ")
                pair = (int(triple[0]), int(triple[1]))
                dissim_sents_pair.append(pair)

    return p_vecs, n_vecs, dissim_sents_pair


# 根据相似句子，比较其句子向量上的维度的差异
# flag为0时比较不重复的正例句子与其最不相似的负例句子向量的差异维度
# flag为1时比较不重复的负例句子与其最不相似的正例句子向量的差异维度
def compare_sim_sent_dims(flag):
    print("Start comparing gap dimensions......")
    p_vecs, n_vecs, dissim_sents_pairs = read_data(flag)  # 读取数据

    # 将不相似句子对转化为字典，key为负例句子的index，value为正例句子index列表
    dissim_dict = get_dissim_pair_dict(dissim_sents_pairs)
    all_dict_list = []  # 保存所有结果的字典列表

    if flag == 0:
        vecs = n_vecs
        dissim_vecs = p_vecs
    elif flag == 1:
        vecs = p_vecs
        dissim_vecs = n_vecs

    for idx in dissim_dict.keys():
        vec = vecs[idx]  # 主要句子的向量
        dissim_idxs = dissim_dict[idx]  # 差异向量的索引列表
        gap_dims_list = []  # 差异的维度的列表
        for dissim_idx in dissim_idxs:
            dissim_vec = dissim_vecs[dissim_idx]  # 得到一个差异向量
            gap_dims = compare_two_vecs(vec, dissim_vec)  # 得到两个向量的差异维度
            gap_dims_list.append(gap_dims)  # 将一组差异维度加入列表中

        if len(gap_dims_list) == 1:
            dic = {}
            dic[dissim_idx] = gap_dims_list[0]  # 将差异向量的索引为key,差异维度为value添加进字典中
            all_dict_list.append(dic)
        else:
            dict_list = analyze_gap_dims(dissim_idxs, gap_dims_list)
            for dict in dict_list:
                all_dict_list.append(dict)

    print("Finished comparing dimensions.")
    return all_dict_list


# 将比较最不相似句子各维度的结果转化为一个字典返回
def get_all_dict(flag):
    if flag == 0:
        all_dict_list = compare_sim_sent_dims(0)  # 比较最不相似句子向量各维度之间的差异，返回结果
    elif flag == 1:
        all_dict_list = compare_sim_sent_dims(1)
    all_dict = {}
    for dict in all_dict_list:  # 将字典列表转化为一个字典
        for key in dict.keys():
            all_dict[key] = dict[key]
    return all_dict


# 与上述函数功能相同，但用于测试
def get_all_dict_for_test(flag):
    p_vecs, n_vecs, dissim_sents_pairs = read_data(flag)  # 读取数据
    pos_sents = []
    neg_sents = []
    with open("pos_neg_vecs/pos_sents", "r") as f:
        for line in f.readlines():
            pos_sents.append(line)
    with open("pos_neg_vecs/neg_sents", "r") as f:
        for line in f.readlines():
            neg_sents.append(line)

    if flag == 0:
        sents = neg_sents
        dissim_sents = pos_sents
    elif flag == 1:
        sents = pos_sents
        dissim_sents = neg_sents

    # flag为0时，dict的key为负例句子索引，value为正例句子索引的列表
    # flag为1时，dict的key为正例句子索引，value为负例句子索引的列表
    dissim_dict = get_dissim_pair_dict(dissim_sents_pairs)
    for key in dissim_dict.keys():
        if len(dissim_dict[key]) >= 4:
            print(sents[key].replace("\n", ""))
            for dis_idx in dissim_dict[key]:
                print(dissim_sents[dis_idx].replace("\n", ""))
            print("-----------------------------")


# 读取训练集相似度最低的句子对索引，返回两个列表
def read_dissim_indexes():
    dissim_pos_neg_pairs = []
    dissim_neg_pos_pairs = []
    # 读取不重复的正例句子及与其最不相似的负例句子index
    with open(sim_sents_index_file_path, 'r') as s:
        for line in s.readlines():
            triple = line.replace("\n", "").split(" ")
            pair = (int(triple[0]), int(triple[1]))
            dissim_pos_neg_pairs.append(pair)
    # 读取不重复的负例句子及与其最不相似的正例句子index
    with open(sim_sents_index_file_path2, 'r') as s:
        for line in s.readlines():
            triple = line.replace("\n", "").split(" ")
            pair = (int(triple[0]), int(triple[1]))
            dissim_neg_pos_pairs.append(pair)
    return dissim_pos_neg_pairs, dissim_neg_pos_pairs


# 准备好训练集进行维度调整前所需的所有数据
# 第一个参数为文本路径，第二个参数为文本中句子对应的向量路径
def prepare_data(text_file_path, vec_path):
    p_all_dict = get_all_dict(0)  # 各正例句子的差值维度，key为正例句子的index，value为该句子的差值维度
    n_all_dict = get_all_dict(1)  # 各负例句子的差值维度，key为负例句子的index，value为该句子的差值维度
    # 读取数据
    p_vecs = []
    n_vecs = []
    p_sents = []
    n_sents = []
    all_sents = []
    f = open(vec_path, 'r')
    t = open(text_file_path, 'r')
    vec_lines = f.readlines()
    text_lines = t.readlines()
    for i in xrange(0, len(vec_lines)):
        if " 1\n" in text_lines[i] and " 1\n" in vec_lines[i] and text_lines[i] not in p_sents:
            vec = np.fromstring(vec_lines[i].replace(" 1\n", ""), dtype='float32', sep=' ')
            p_vecs.append(vec)
            p_sents.append(text_lines[i])
        elif " 0\n" in text_lines[i] and " 0\n" in vec_lines[i] and text_lines[i] not in n_sents:
            vec = np.fromstring(vec_lines[i].replace(" 0\n", ""), dtype='float32', sep=' ')
            n_vecs.append(vec)
            n_sents.append(text_lines[i])
        all_sents.append(text_lines[i])

    f.close()
    t.close()

    dissim_pos_neg_pairs, dissim_neg_pos_pairs = read_dissim_indexes()

    return all_sents, p_sents, n_sents, p_vecs, n_vecs, p_all_dict, n_all_dict, dissim_pos_neg_pairs, dissim_neg_pos_pairs


# 调整训练集句子向量的维度
def adjust_sent_vec_dims(text_file_path, ori_vec_path, adjusted_vec_path):
    # 获取所有数据，包括正负句子列表，正负句子向量列表，正负句子差异维度字典以及正负最不相似句子对
    all_sents, p_sents, n_sents, p_vecs, n_vecs, p_all_dict, n_all_dict, dissim_pos_neg_pairs, dissim_neg_pos_pairs = prepare_data(
        text_file_path, ori_vec_path
    )
    #for i in xrange(0, 30, 3):
    adjust_dim_iteration(all_sents, p_sents, n_sents, p_vecs, n_vecs,
                            p_all_dict, n_all_dict, dissim_pos_neg_pairs, dissim_neg_pos_pairs, 28, adjusted_vec_path)


# 对句子向量维度调整的一次迭代，迭代后查看分类效果。参数turns为对每个句子向量进行调整的轮数
def adjust_dim_iteration(all_sents, p_sents, n_sents, p_vecs, n_vecs,
                         p_all_dict, n_all_dict, dissim_pos_neg_pairs, dissim_neg_pos_pairs, turns, adjusted_vec_path, file=None):

    sample_vecs = []  # 进行分类的特征向量
    labels = []  # 特征向量对应标签
    opt_vec_file = open(adjusted_vec_path, 'w')  # 写入调整后的向量
    print("Begin adjusting dim...")
    sum = 0
    for sent in all_sents:
        if " 0\n" in sent:  # 先得到句子的向量和在对应不重复句子列表中的索引，而后得到差异维度，从而根据差异维度对原向量进行调整
            vec, n_idx = get_sent_vec(sent, n_sents, n_vecs)  # 获得该负例句子的向量以及在不重复负例句子列表中的index
            if vec is None and n_idx is None:
                print("None")
                continue
            gap_dims = n_all_dict[n_idx]  # 根据获得的index得到对应的差异维度列表
            dissim_p_idx = dissim_neg_pos_pairs[n_idx][1]
            dissim_p_vec = p_vecs[dissim_p_idx]  # 得到与该负例句子最不相似的正例句子向量
            adjusted_vec = adjust_vec_dims(vec, gap_dims, turns, dissim_p_vec, file)
            sample_vecs.append(adjusted_vec)
            labels.append(0)
        elif " 1\n" in sent:
            vec, p_idx = get_sent_vec(sent, p_sents, p_vecs)  # 获得该正例句子的向量以及在不重复正例句子列表中的index
            if vec is None and p_idx is None:
                print("None")
                continue
            gap_dims = p_all_dict[p_idx]  # 得到差异维度的列表
            dissim_n_idx = dissim_pos_neg_pairs[p_idx][1]
            dissim_n_vec = n_vecs[dissim_n_idx]  # 得到与该正例句子最不相似的负例句子向量
            adjusted_vec = adjust_vec_dims(vec,  gap_dims, turns, dissim_n_vec, file)  # 得到调整后的句向量
            sample_vecs.append(adjusted_vec)
            labels.append(1)
        sum += 1
        print("Round-"+str(turns/3+1)+" Finish sentence:"+str(sum))

    for i in xrange(0, len(sample_vecs)):
        write_vec_to_file(opt_vec_file, sample_vecs[i], labels[i])  # 写入文件


# 对测试集向量维度的一次调整迭代
def adjust_test_dim_iteration(all_test_sents, test_p_sents, test_n_sents, test_p_vecs, test_n_vecs, p_all_dict, n_all_dict, turns):
    sample_vecs = []
    labels = []
    print("Begin adjusting dim...")
    opt_vec_file = open("train_test_data/test_adjusted_data", 'w')  # 写入调整后的向量
    sum = 0
    for sent in all_test_sents:
        if " 0\n" in sent:
            vec, n_idx = get_sent_vec(sent, test_n_sents, test_n_vecs)  # 找到该句子在
            gap_dims = n_all_dict[n_idx]
            adjusted_vec = adjust_vec_dims(vec, gap_dims=gap_dims, turns=turns)
            sample_vecs.append(adjusted_vec)
        elif " 1\n" in sent:
            vec, p_idx = get_sent_vec(sent, test_p_sents, test_p_vecs)
            gap_dims = p_all_dict[p_idx]
            adjusted_vec = adjust_vec_dims(vec, gap_dims=gap_dims, turns=turns)
            sample_vecs.append(adjusted_vec)
        sum += 1
        print("Round-" + str(turns / 3 + 1) + " Finish sentence:" + str(sum))

    with open("train_test_data/test_text", 'r') as f:
        for line in f.readlines():
            if ' 1\n' in line:
                labels.append(1)
            else:
                labels.append(0)

    for i in xrange(0, len(sample_vecs)):
        write_vec_to_file(opt_vec_file, sample_vecs[i], labels[i])  # 写入文件


# 比较两个向量的维度，返回记录相差较大的维数的列表
def compare_two_vecs(vec1, vec2):
    vec1 = np.fromstring(vec1, dtype='float32', sep=' ')
    vec2 = np.fromstring(vec2, dtype='float32', sep=' ')
    diff_sum = 0  # 两向量各维度差的和
    gap_dims = []  # 记录两个向量相差较大的维度数
    for i in xrange(0, len(vec1)):
        n1 = vec1[i]
        n2 = vec2[i]
        if n1*n2 > 0:
            diff_sum += abs(n1-n2)
    diff_avg = diff_sum/len(vec1)
    for i in xrange(0, len(vec1)):
        # 当两个向量的同一维数值异号或是同号且差值大于平均值，则认为该维存在较大差距
        if vec1[i]*vec2[i] < 0:  # or (vec1[i]*vec2[i] > 0 and abs(vec1[i]-vec2[i]) > diff_avg):
            gap_dims.append(i)
    return gap_dims


# 因为有些负例句子与多个正例句子的相似度最低，以这些负例句子index为key，与其最不相似的正例句子index构成的列表为value，返回这个字典
def get_dissim_pair_dict(dissim_sents_pairs):
    dict = {}
    for tuple in dissim_sents_pairs:
        if tuple[1] not in dict.keys():
            list = []
            list.append(tuple[0])
            dict[tuple[1]] = list
        elif tuple[1] in dict.keys():
            dict[tuple[1]].append(tuple[0])
    return dict


# 得到一个正例或负例句子的向量，sent为该句子文本，sents为不重复的正例或负例句子列表，vecs为对应句子列表的向量列表
def get_sent_vec(sent, sents, vecs):
    for i in xrange(0, len(sents)):
        if sent == sents[i]:
            return vecs[i], i
    return None, None


# 比较经过维度调整之后的正例句子向量之间的相似度
def compare_ajt_pos_sents_sim():
    p_sents = []
    p_adjusted_vecs = []
    text_file = open(text_file_path, 'r')
    adjusted_vecs_file = open(w2v_adjusted_vec_file_path, 'r')
    text_lines = text_file.readlines()
    vecs_lines = adjusted_vecs_file.readlines()
    for i in xrange(0, len(text_lines)):
        if " 1\n" in text_lines[i] and " 1\n" in vecs_lines[i] and text_lines[i] not in p_sents:
            p_sents.append(text_lines[i])
            vec = np.fromstring(vecs_lines[i].replace(" 1\n", ""), dtype='float32', sep=' ')
            p_adjusted_vecs.append(vec)

    for i in xrange(0, len(p_sents)):
        sent = p_sents[i]
        vec = p_adjusted_vecs[i]
        most_idx, least_idx, most_sim, least_sim = cal_ovm_sentence_sim(vec, sim_vecs=p_adjusted_vecs)
        print(sent.replace(" 1\n", ""))
        print(p_sents[most_idx].replace(" 1\n", ""))
        print("Sim: %.4f" % most_sim)
        print('\n')


# 将不相似的句子写入文件中，便于查看
def write_dissim_sents_to_file():
    p_sents = []
    n_sents = []
    f = open(w2v_all_features_file_path, 'r')
    t = open(text_file_path, 'r')
    vec_lines = f.readlines()
    text_lines = t.readlines()
    for i in xrange(0, len(vec_lines)):
        if " 1\n" in text_lines[i] and " 1\n" in vec_lines[i] and text_lines[i] not in p_sents:
            p_sents.append(text_lines[i])
        elif " 0\n" in text_lines[i] and " 0\n" in vec_lines[i] and text_lines[i] not in n_sents:
            n_sents.append(text_lines[i])

    dissim_sents_pairs = []  # 正例句子与其最不相似负例句子的index
    with open(sim_sents_index_file_path, 'r') as s:
        for line in s.readlines():
            triple = line.replace("\n", "").split(" ")
            pair = (int(triple[0]), int(triple[2]))
            dissim_sents_pairs.append(pair)
    dissim_dict = get_dissim_pair_dict(dissim_sents_pairs)

    w = open("pos_neg_vecs/dissim_pos_neg", 'w')
    for n_idx in dissim_dict.keys():
        n_sent = n_sents[n_idx]
        w.write(str(n_idx)+" ")
        w.write(n_sent)
        p_idxs = dissim_dict[n_idx]
        for p_idx in p_idxs:
            p_sent = p_sents[p_idx]
            w.write(str(p_idx)+" ")
            w.write(p_sent)
        w.write("\n")


# =================================================================
# 比较经过调整向量维度，减去的向量是否隐藏着一些语义在其中
# 参数为一组在结构上相似的句子
def compare_decrease(pos_sent, neg_sent):
    text_file = open(text_file_path, 'r')  # 句子文本
    ori_vec_file = open(w2v_all_features_file_path, 'r')  # 初始向量
    new_vec_file = open("svm_data/new_opt_adj_vec_data.txt", 'r')  # 维度调整后的向量
    ori_vec_lines = ori_vec_file.readlines()
    new_vec_lines = new_vec_file.readlines()
    text_lines = text_file.readlines()
    ori_pos_vecs = []
    new_pos_vecs = []
    ori_neg_vecs = []
    new_neg_vecs = []
    pos_sents = []
    neg_sents = []
    for i in xrange(0, len(ori_vec_lines)):
        if " 1\n" in text_lines[i]:
            if text_lines[i].replace(" 1\n", "") not in pos_sents:
                pos_sents.append(text_lines[i].replace(" 1\n", ""))
                ori_pos_vecs.append(np.fromstring(ori_vec_lines[i].replace(" 1\n", ""), dtype='float32', sep=' '))
                new_pos_vecs.append(np.fromstring(new_vec_lines[i].replace(" 1\n", ""), dtype='float32', sep=' '))
        elif " 0\n" in text_lines[i]:
            if text_lines[i].replace(" 0\n", "") not in neg_sents:
                neg_sents.append(text_lines[i].replace(" 0\n", ""))
                ori_neg_vecs.append(np.fromstring(ori_vec_lines[i].replace(" 0\n", ""), dtype='float32', sep=' '))
                new_neg_vecs.append(np.fromstring(new_vec_lines[i].replace(" 0\n", ""), dtype='float32', sep=' '))

    # 计算正例句子与含有正例句子产业词负例句子句向量差的相似度
    for i in xrange(0, len(pos_sents)):
        if pos_sent == pos_sents[i]:
            for j in xrange(0, len(neg_sents)):
                if neg_sent == neg_sents[j]:
                    print(pos_sent)
                    print(neg_sent)
                    pos_diff = new_pos_vecs[i] - ori_pos_vecs[i]
                    neg_diff = new_neg_vecs[j] - ori_neg_vecs[j]
                    sim = cos_sim(pos_diff, neg_diff)
                    print("Sim:"+str(sim))
                    break

    '''
    for i in xrange(0, len(pos_sents)):
        sim = 0
        sim_index = 0
        for j in xrange(0, len(pos_sents)):
            if i != j:
                diff = new_pos_vecs[i] - ori_pos_vecs[i]  # 差距向量
                diff1 = new_pos_vecs[j] - ori_pos_vecs[j]
                cur_sim = cos_sim(diff, diff1)
                if cur_sim > sim:
                    sim = cur_sim
                    sim_index = j
        print(pos_sents[i])
        print(pos_sents[sim_index])
        print(sim)
        print("=============")
    '''


train_text_path = "train_test_data/train_text"  # 训练集句子
train_vec_path = "train_test_data/train_vec"  # 训练集向量
adjusted_train_vec_path = "train_test_data/train_adjusted_data"  # 经过维度调整的训练集向量
out_path1 = "training_data/sim_sents_index"

test_text_path = "train_test_data/test_text"  # 测试集句子
test_vec_path = "train_test_data/test_vec"  # 测试集向量
adjusted_test_vec_path = "train_test_data/test_adjusted_data"  # 经过维度调整的测试集向量
out_path2 = "training_data/sim_sents_index2"

pos_vec_path = "pos_neg_vecs/pos_vecs"  # 未经维度调整的训练集不重复的正例向量
neg_vec_path = "pos_neg_vecs/neg_vecs"  # 未经维度调整的训练集不重复的负例向量

# yelp数据集
yelp_tr_path = "yelp_train_test/train_vec"
yelp_te_path = "yelp_train_test/test_vec"
out1 = "yelp_train_test/sim_sents_index1"
out2 = "yelp_train_test/sim_sents_index2"


# 当训练集和测试集变化时，需要重新运行第一、二个函数，第二个函数需要分别传入参数1和0
#write_vecs_to_file(train_text_path, train_vec_path, test_text_path, test_vec_path)
#cal_all_sentences_sim(train_text_path, train_vec_path, test_text_path, test_vec_path, 1, out_path1, out_path2)
#cal_all_sentences_sim(train_text_path, train_vec_path, test_text_path, test_vec_path, 0, out_path1, out_path2)
#adjust_sent_vec_dims(train_text_path, train_vec_path, adjusted_train_vec_path)
#get_all_dict(0)
#get_all_dict_for_test(0)

