# -*- encoding: utf-8 -*-
# 该类包含一些被主要程序调用的算法和功能

from math import exp
import numpy as np


# 第一个参数为一个负例句子对应的所有正例句子的index列表，在分析维度差异完成后需要以此来将每个句子与对应的差异维度联系起来
# 第二个参数为一个负例句子对应的所有正例句子差异维数的列表，列表中每个元素为一个正例句子与该负例句子存在差异维数的列表
def analyze_gap_dims(p_idxs, gap_dims_list):
    gap_dims_cats = []  # 每一个元素是一个类别
    dict_list = []  # 保存结果，每个元素为一个字典，key为一个正例句子的Index，value为对应的共同差异维度列表
    for i in xrange(0, len(gap_dims_list)):
        p_idx = p_idxs[i]  # 正例句子index
        list = gap_dims_list[i]  # 该正例句子与负例句子存在差异维度的列表
        if len(gap_dims_cats) == 0:
            gap_dims_cats.append(GapDimCategory(p_idx, list))
        else:
            is_added = False
            for cat in gap_dims_cats:
                if cat.belongs_to_cat(p_idx, list):
                    is_added = True
                    break
            if not is_added:
                gap_dims_cats.append(GapDimCategory(p_idx, list))

    for cat in gap_dims_cats:
        #print("Cat "+str(gap_dims_cats.index(cat)))
        rs = cat.get_common_gap_dims()
        dict_list.append(rs)
        #print("Common gap dims:"+str(rs))
    return dict_list


# 根据两个列表中元素的重合程度判断其是否相似
def is_similar_list(list1, list2):
    if len(list1) >= len(list2):
        short_list = list2
        long_list = list1
    else:
        short_list = list1
        long_list = list2
    threshold = 0.7 * len(short_list)  # 阈值，当较短长度列表中超过阈值个数的元素出现在较长列表中则认定两列表相似
    overlap_num = 0
    for num in short_list:
        if num in long_list:
            overlap_num += 1
    if overlap_num >= threshold:
        return True
    else:
        return False


# ====================================================
# 对句向量的差异维度进行调整，vec为需要调整的向量，dissim_vec为与vec相似度最低的向量，gap_dims为两向量存在差异的维度
# 参数turns为向量维度进行调整的轮数，file为写入权重的文件
def adjust_vec_dims(vec, gap_dims, turns, dissim_vec=None, file=None):
    cur_vec = vec.copy()
    #cur_sim = cos_sim(vec, dissim_vec)  # 初始相似度
    # 获得差异维度以外维度的初始值的符号（正负号），返回一个字典，key为维数，value为1表示正号，-1表示负号
    gap_dims_symbols = get_gap_dims_symbols(vec, gap_dims)
    learning_rate = 0.05
    # 写入初始权重
    for i in xrange(0, turns):
        #if i == 0:  # 写入初始权重
            #weights = compare_vecs_after_adjust(cur_vec,vec)
            #write_file(weights, file)

        cur_vec = adjust_vec_dims_once(cur_vec, gap_dims, learning_rate, gap_dims_symbols)
        #p_n_sim = cos_sim(cur_vec, dissim_vec)
        # 对比调整后的向量与原向量各维度的变化（即权重）
        #if i % 3 == 0 and i != 0:  # 每迭代三次查看权重变化情况并写入文件
            #weights = compare_vecs_after_adjust(vec, cur_vec)
            #write_file(weights, file)
    return cur_vec


# step为步长，对向量的不重要维度做一次调整
def adjust_vec_dims_once(vec, gap_dims, learning_rate, gap_dims_symbols):
    for i in xrange(0, len(vec)):
        if i not in gap_dims:
            if abs(vec[i]) > 1 and vec[i]*gap_dims_symbols[i]>0:  # 对于不重要的维度需要尽快减小其权重
                param = abs(vec[i])
                vec[i] *= exp(-learning_rate * param)  # 不重要维度权值不断减小
            elif abs(vec[i]) <= 1 and abs(vec[i]) != 0 and vec[i]*gap_dims_symbols[i]>0:
                param = 1/abs(vec[i])
                vec[i] *= exp(-learning_rate * param)  # 不重要维度权值不断减小
            elif vec[i] == 0:  # 若该维度被调整到0，则将其向原方向的反方向调整
                if gap_dims_symbols[i] == 1:
                    vec[i] = -exp(-5)  # 若原来为正，初始化一个负数
                else:
                    vec[i] = exp(-5)  # 若原来为负，初始化一个正数
                continue
            elif vec[i] * gap_dims_symbols[i] < 0:  # 若该维度的符号已经和原值相反，则进行调整
                if gap_dims_symbols[i] == 1:
                    vec[i] -= exp(-5)  # 若原来为正，朝着负向增加
                else:
                    vec[i] += exp(-5)  # 若原来为负，朝着正向增加
                continue
        else:
            if abs(vec[i]) > 1:  # 对于重要的维度控制其维度的增长速度
                param = 1 / abs(vec[i])
            else:
                param = abs(vec[i])

            vec[i] *= exp(learning_rate*param)  # 重要维度权值不断增大
    return vec


# 当不重要维度的值的绝对值都小于步长时说明此时向量已调整完毕
def are_dims_stable(vec, gap_dims, step):
    for i in xrange(0, len(vec)):
        if i not in gap_dims:
            if abs(vec[i]) > step:
                return False
    return True


# 获得差异维度以外维度初始值的正负，以此来对这些维度进行反向调整
def get_gap_dims_symbols(vec, gap_dims):
    symbols = {}
    for i in xrange(0, len(vec)):
        if i not in gap_dims:
            if vec[i] > 0:
                symbols[i] = 1
            else:
                symbols[i] = -1
    return symbols


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


# 对比调整后的向量与原向量各维度的差别
def compare_vecs_after_adjust(ori_vec, cur_vec):
    weights = []  # 记录每一维调整后的权重，初始权重均为1
    for i in xrange(0, len(ori_vec)):
        if cur_vec[i] == 0:
            weights.append(str(0)+'    ')
        elif cur_vec[i]*ori_vec[i] > 0:
            if cur_vec[i] == ori_vec[i]:
                weights.append('1.0  ')
            else:
                weights.append(str(cur_vec[i]/ori_vec[i])[0:5])
        elif cur_vec[i]*ori_vec[i] < 0:
            if cur_vec[i] < 0:
                weights.append(str(cur_vec[i])[0:5])
            else:
                weights.append('+'+str(cur_vec[i])[0:4])
    return weights


# 将迭代后的权重写入文件中
# is_original表示将初始权重写入，即向量各维度权重均为1
def write_file(weights, file):
    for weight in weights:
        file.write(weight+' ')
    file.write('\n')


class GapDimCategory:
    def __init__(self, p_idx, list):
        self.cat_list = []
        self.longest_list = list
        self.cat_list.append((p_idx, list))

    # 根据该列表与该类中最长列表是否相似来决定是否把该列表加入该类中
    def belongs_to_cat(self, p_idx, list):
        if is_similar_list(list, self.longest_list):
            self.cat_list.append((p_idx, list))
            if len(list) > len(self.longest_list):
                self.longest_list = list
            return True
        else:
            return False

    # 找到那些在该类多个句子向量中均有较大变化的维度
    def get_common_gap_dims(self):
        dict = {}
        common_gap_dims = []
        p_idxs = []  # 记录正例句子的index列表
        rs_dict = {}  # 记录结果的字典，key为一个正例句子的index，value为共同的差异维度列表
        if len(self.cat_list) == 1:
            p_idx = self.cat_list[0][0]
            rs_dict[p_idx] = self.cat_list[0][1]
            return rs_dict

        for list in self.cat_list:
            for num in list[1]:  # list为元组，元组第一个元素为正例句子的index，第二个元素才是差异维度的列表
                if num not in dict.keys():
                    dict[num] = 1
                else:
                    dict[num] += 1
        for dim in dict.keys():
            # 若该维度在该类60%以上的句子中变化较大，则记录
            if dict[dim] > 0.6 * len(self.cat_list):
                common_gap_dims.append(dim)
        for tup in self.cat_list:
            rs_dict[tup[0]] = common_gap_dims
        return rs_dict

    # 打印该类所存的列表
    def print_list(self):
        for list in self.cat_list:
            print(str(list))
