# -*- encoding: utf-8 -*-

from sentence_classifier.word2vec_research import read_data, get_dissim_pair_dict, get_all_dict
import matplotlib.pyplot as plt
from pylab import *


def cal_word_freq(sent_path):
    pos_word_freq_dic = {}
    neg_word_freq_dic = {}
    i = 0
    with open(sent_path, 'r') as f:
        for line in f.readlines():
            words = line.replace(" 1\n", "").replace(" 0\n", "").split(" ")
            if " 1\n" in line:
                for word in words:
                    if word in pos_word_freq_dic.keys():
                        pos_word_freq_dic[unicode(word)] += 1
                    else:
                        pos_word_freq_dic[unicode(word)] = 1
            else:
                for word in words:
                    if word in neg_word_freq_dic.keys():
                        neg_word_freq_dic[unicode(word)] += 1
                    else:
                        neg_word_freq_dic[unicode(word)] = 1
            i += 1
            print("Finish sentence:"+str(i))

    pos_word_avg = sum(pos_word_freq_dic.values())/len(pos_word_freq_dic.keys())  # 训练集中正例中词语的平均出现次数
    neg_word_avg = sum(neg_word_freq_dic.values())/len(neg_word_freq_dic.keys())  # 训练集中负例中词语的平均出现次数
    return pos_word_freq_dic, neg_word_freq_dic, pos_word_avg, neg_word_avg


# 将字符串数据转化为float数组
def str_float_cast(str_list):
    float_list = []
    for str in str_list:
        float_list.append(float(str))
    return float_list


# 读取测试集句向量及其标签
def read_test_vecs(test_vec_path):
    test_vecs = []
    test_labels = []
    with open(test_vec_path, 'r') as f:
        for line in f.readlines():
            if " 1\n" in line:
                vec = line.replace(" 1\n", "").split()
                test_labels.append(1)
            else:
                vec = line.replace(" 0\n", "").split()
                test_labels.append(0)
            test_vecs.append(str_float_cast(vec))
    return test_vecs, test_labels


# 读取训练集和测试集中所有需要用到的数据
def read_all_data(test_vec_path):
    # 读取测试集所需的所有数据
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

    test_vecs, test_labels = read_test_vecs(test_vec_path)

    return p_vecs, n_vecs, neg_pos_dict, pos_neg_dict, p_all_dict, n_all_dict, p_sents, n_sents, test_vecs, test_labels


# test_id为所测句子组的下标，从0开始
# is_pos表示是否选取正例进行绘制
def draw_pic_for_vecs(test_id, is_pos=True):
    test_vecs_path = "train_test_data/test_vec"
    p_vecs, n_vecs, neg_pos_dict, pos_neg_dict, p_all_dict, n_all_dict, p_sents, n_sents, test_vecs = read_all_data(test_vecs_path)

    if is_pos:
        group_dict = neg_pos_dict
        vectors = p_vecs
        sentences = p_sents
    else:
        group_dict = pos_neg_dict
        vectors = n_vecs
        sentences = n_sents

    vec_ids_list = []
    for key in group_dict.keys():
        if len(group_dict[key]) >= 5:
            vec_ids = group_dict[key]
            vec_ids_list.append(vec_ids)

    vecs_list = []  # 向量组列表，每个元素为一组相似向量
    sents_list = []  # 句子组列表，每个元素为一组相似句子
    for vec_ids in vec_ids_list:
        vecs = []  # 保存一组向量
        sents = []  # 保存一组句子
        for id in vec_ids:
            float_vec = str_float_cast(vectors[id].split())
            vecs.append(float_vec)
            sents.append(sentences[id])
        vecs_list.append(vecs)
        sents_list.append(sents)

    for vecs in vecs_list:
        common_summits = cal_summits_for_group(vecs, 4)  # 计算一组句子向量的公共峰值维度
        print(common_summits)
    draw_pic_for_a_group_vecs(vecs_list[test_id], 4)
    #print_a_group_sents(sents_list[test_id])


# 输出一组句子
def print_a_group_sents(sents):
    for sent in sents:
        print(sent)


# 为一组相似句向量画折线图
# step为步长，表示在向量维度内每多少个维度取一次点进行绘制
def draw_pic_for_a_group_vecs(vecs, step=1):
    # 向量维度数
    x = []
    for i in range(0, 100, step):
        x.append(i)

    # 确定y轴的最高值与最低值
    max_y = -100
    min_y = 100
    for vec in vecs:
        for dim in vec:
            if dim > max_y:
                max_y = dim
        for dim in vec:
            if dim < min_y:
                min_y = dim

    plt.ylim(-30, 30)

    rc_vecs = []
    for vec in vecs:
        temp_vecs = []
        for i in range(0, 100, step):
            temp_vecs.append(vec[i])
        rc_vecs.append(temp_vecs)

    for vec in rc_vecs:
        plt.plot(x, vec,)
    plt.legend()
    plt.title("Variation of sentence vector dimensions")
    plt.xlabel("Vector dimension")
    plt.ylabel("Dimension value")
    plt.show()


############################################################
# 将一组句向量的公共极值维度和这些维度对应的最大最小值存于字典中
def lists_to_dict(sorted_summits, summit_values_dict):
    smt_max_min_value_dict = {}  # key为峰值维度,value为该维度的最大值与最小值
    for summit in sorted_summits:
        values = summit_values_dict[summit]
        sorted_values = sort(values)
        max_value = sorted_values[len(sorted_values)-1]
        min_value = sorted_values[0]
        smt_max_min_value_dict[summit] = (round(min_value, 7), round(max_value, 7))
    return smt_max_min_value_dict


# 将key-value存入字典
def add_kv_to_dict(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        list = []
        list.append(value)
        dict[key] = list
    return dict


# 为一组相似向量计算其共有的极值维度
def cal_summits_for_group(vecs, step=1):
    rc_vecs = []
    for vec in vecs:
        temp_vecs = []
        for i in range(0, 100, step):
            temp_vecs.append(vec[i])
        rc_vecs.append(temp_vecs)

    length = len(rc_vecs[0])
    summit_list = []  # 保存极值维度的列表，每一个元素为一个向量的极值维度列表
    summit_values_dict = {}  # key为极值维度，value为该组句向量中在该极值点上出现过的数值
    for vec in vecs:
        summit_sub_list = []  # 保存每一个向量的极值维度
        for i in xrange(1, length-1):
            if vec[i] > vec[i-1] and vec[i] > vec[i+1]:  # 极大值维度点
                summit_sub_list.append(i*step)  # 记录维度数，需要乘以步长
                add_kv_to_dict(summit_values_dict, i*step, vec[i])
            elif vec[i] < vec[i-1] and vec[i] < vec[i+1]:  # 极小值维度点
                summit_sub_list.append(-i*step)
                add_kv_to_dict(summit_values_dict, -i*step, vec[i])
        summit_list.append(summit_sub_list)

    # 将极值点存入字典，便于统计
    summit_dic = {}  # key为取得极值的维度，value为存在该极值的句子数
    for summits in summit_list:
        for dim in summits:
            if dim in summit_dic.keys():
                summit_dic[dim] += 1
            else:
                summit_dic[dim] = 1

    common_summits = []
    for dim in summit_dic.keys():
        if summit_dic[dim] > len(vecs) * 0.6:  # 当某一极值维度出现次数大于一定阈值便当做是公共极值
            common_summits.append(dim)

    # key为句向量组的公共极值维度，value为句向量组在该维度上出现的最小值和最大值构成的元组
    smt_extreme_val_dict = lists_to_dict(common_summits, summit_values_dict)
    return smt_extreme_val_dict


# 计算测试集中一个向量的极值维度及该维度的值，返回一个字典
def cal_summits_for_test_sent(vec, step=1):
    rc_vec = []
    for i in range(0, 100, step):  # 以步长来构造新的向量
        rc_vec.append(vec[i])

    length = len(rc_vec)
    smt_val_dict = {}
    for i in xrange(1, length-1):
        if vec[i] > vec[i-1] and vec[i] > vec[i+1]:  # 极大值维度点
            smt_val_dict[i*step] = vec[i]
        elif vec[i] < vec[i-1] and vec[i] < vec[i+1]:  # 极小值维度点
            smt_val_dict[-i*step] = vec[i]

    return smt_val_dict


# 计算训练集正例或负例中所有分组句子对应的极值维度
def cal_vec_groups_common_summits(group_dict, vectors, sentences, step):
    vec_ids_list = []
    for key in group_dict.keys():
        if len(group_dict[key]) >= 5:
            vec_ids = group_dict[key]
            vec_ids_list.append(vec_ids)

    vecs_list = []  # 向量组列表，每个元素为一组相似向量
    sents_list = []  # 句子组列表，每个元素为一组相似句子
    for vec_ids in vec_ids_list:
        vecs = []  # 保存一组向量
        sents = []  # 保存一组句子
        for id in vec_ids:
            float_vec = str_float_cast(vectors[id].split())
            vecs.append(float_vec)
            sents.append(sentences[id])
        vecs_list.append(vecs)
        sents_list.append(sents)

    # 二维数组，每一个元素为一个字典，key为一组句向量的公共极值维度，value为该组句向量在该维度上最大最小值
    smt_val_dict_list = []
    for vecs in vecs_list:
        smt_extreme_val_dict = cal_summits_for_group(vecs, step)  # 计算一组句子向量的公共峰值维度
        smt_val_dict_list.append(smt_extreme_val_dict)

    return smt_val_dict_list


# 通过维度极值点的个数，计算一个测试集句向量和训练集中各句向量组之间的相似度
def cal_sim_by_smt_num_btw_test_train(test_summits, smt_val_dict_list):
    most_common_sum = 0
    most_common_id = 0
    for i in xrange(0, len(smt_val_dict_list)):
        cur_summits = smt_val_dict_list[i].keys()
        common_summits_num = cal_common_summits_num_ovo(test_summits, cur_summits)
        if common_summits_num > most_common_sum:
            most_common_sum = common_summits_num
            most_common_id = i
    return most_common_id, most_common_sum


# 计算两个句向量共有的维度极值点
def cal_common_summits_num_ovo(summits1, summits2):
    # 比较维度极值点的数量，以数量少的为基准，判断维度数量少的列表中的维度是否出现在维度数量大的列表中
    length = min(len(summits1), len(summits2))
    if len(summits1) == length:
        shorter_summits = summits1
        longer_summits = summits2
    elif len(summits2) == length:
        shorter_summits = summits2
        longer_summits = summits1

    common_sum = 0  # 公共维度极值点的数
    for i in xrange(0, length):
        if shorter_summits[i] in longer_summits:
            common_sum += 1
    return common_sum


# 当一个测试集句向量的极值维度同时与训练集中一组正例和一组负例的公共极值维度相同时，比较其在各个维度上的值的差异
def cal_sim_by_smt_val_btw_test_train(smt_val_dict, pos_smt_val_dict, neg_smt_val_dict):
    pos_match_num = 0
    neg_match_num = 0
    for key in smt_val_dict.keys():
        if key in pos_smt_val_dict.keys():
            pos_min = pos_smt_val_dict[key][0]
            pos_max = pos_smt_val_dict[key][1]
            if pos_max > smt_val_dict[key] > pos_min:
                pos_match_num += 1
        if key in neg_smt_val_dict.keys():
            neg_min = neg_smt_val_dict[key][0]
            neg_max = neg_smt_val_dict[key][1]
            if neg_max > smt_val_dict[key] > neg_min:
                neg_match_num += 1

    if pos_match_num > neg_match_num:
        return 1
    elif pos_match_num < neg_match_num:
        return 0
    else:
        return -1


# 计算测试集中每一个句向量与训练集中各句向量组的相似度
def cal_sim_ovm(step=1):
    test_vecs_path = "train_test_data/test_vec"
    p_vecs, n_vecs, neg_pos_dict, pos_neg_dict, p_all_dict, n_all_dict, p_sents, n_sents, test_vecs, test_labels = read_all_data(
        test_vecs_path)

    test_sents = []
    with open("train_test_data/test_text", 'r') as f:
        for line in f.readlines():
            test_sents.append(line.replace("\n", ""))

    pos_smt_val_dict_list = cal_vec_groups_common_summits(neg_pos_dict, p_vecs, p_sents, step)
    neg_smt_val_dict_list = cal_vec_groups_common_summits(pos_neg_dict, n_vecs, n_sents, step)

    error_num = 0.0
    same_num = 0.0
    right_num = 0.0
    for i in xrange(0, len(test_vecs)):
        test_vec = test_vecs[i]
        smt_val_dict = cal_summits_for_test_sent(test_vec, step)
        most_pos_com_id, most_pos_com_sum = cal_sim_by_smt_num_btw_test_train(smt_val_dict.keys(), pos_smt_val_dict_list)
        most_neg_com_id, most_neg_com_sum = cal_sim_by_smt_num_btw_test_train(smt_val_dict.keys(), neg_smt_val_dict_list)
        if most_pos_com_sum > most_neg_com_sum and test_labels[i] == 1:
            print("Right")
            right_num += 1
        elif most_pos_com_sum < most_neg_com_sum and test_labels[i] == 0:
            print("Right")
            right_num += 1
        elif most_pos_com_sum == most_neg_com_sum:
            label = cal_sim_by_smt_val_btw_test_train(smt_val_dict, pos_smt_val_dict_list[most_pos_com_id], neg_smt_val_dict_list[most_neg_com_id])
            if label == test_labels[i]:
                print("Right")
                right_num += 1
            else:
                print("Unknown")
                same_num += 1
        else:
            print(test_sents[i])
            error_num += 1
            print("Error")
    print("Right ratio:"+str(right_num/len(test_labels)))
    print("Same ratio:"+str(same_num/len(test_labels)))
    print("Error ratio:"+str(error_num/len(test_labels)))


# 读取经过陈东处理后的数据
def read_cd_vec(ori_path, aft_path, step1=4, step2=10):
    ori_vecs = []
    aft_vecs = []
    if ori_path is not None:
        with open(ori_path, 'r') as f:
            for line in f:
                ori_vecs.append(np.fromstring(line.replace(' 1\n', '').replace(' 0\n', ''), dtype='float32', sep=' '))
    if aft_path is not None:
        with open(aft_path, 'r') as f:
            for line in f:
                ori_vecs.append(np.fromstring(line.replace(' \n', ''), dtype='float32', sep=' '))

    if ori_path is not None:
        draw_pic_for_a_group_vecs(ori_vecs, step1)
    if aft_path is not None:
        draw_pic_for_a_group_vecs(aft_vecs, step2)


sent_path = "training_data/seg_sent_with_label2"
train_sent_path = "train_test_data/train_text"

orinal_path = "sent2vec_cmp/r_11"
after_path = "sent2vec_cmp/war_11.txt"

#draw_pic_for_vecs(0, True)
#cal_sim_ovm(step=4)
read_cd_vec(orinal_path, after_path, step1=4, step2=4)
