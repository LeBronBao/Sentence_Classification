# -*- encoding: utf-8 -*-

import numpy
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE


# 包含句子向量和其类别信息的文本路径
unique_training_data_file_path = "testing_data/unique_svm_testing_data.txt"
training_data_file_path = "testing_data/svm_testing_data.txt"
topic_based_training_data_file_path = "testing_data/topic_based_svm_data.txt"
training_weighted_data_file_path = "testing_data/svm_testing_weighted_data.txt"
topic_based_training_weighted_data_file_path = "testing_data/topic_based_weighted_svm_data.txt"
training_weighted_data_with_tfidf_vec_file_path = "testing_data/svm_testing_weighted_data_with_tfidf_vec.txt"
training_sent_vec_data = "testing_data/svm_testing_data_using_sent_vec"  # 使用n-gram直接训练得到的句向量
dimension = 100  # 维度

svm_data_folder = []
svm_data_folder.append("svm_data/svm_word_sum_data.txt")  # 0
svm_data_folder.append("svm_data/svm_word_sum_topic_data.txt")  # 1
svm_data_folder.append("svm_data/svm_word_sum_tfidf_data.txt")  # 2
svm_data_folder.append("svm_data/svm_word_sum_tfidf_topic_data.txt")  # 3
svm_data_folder.append("svm_data/svm_all_features_data.txt")  # 4
svm_data_folder.append("svm_data/svm_adjusted_vec_data.txt")  # 5
svm_data_folder.append("svm_data/svm_opt_adj_vec_data.txt")  # 6

yelp_data_list = []
yelp_data_list.append("yelp_reviews/yelp_train_test/using_sw_and_half_freq3")
yelp_data_list.append("yelp_reviews/yelp_train_test/using_sw_normal")
yelp_data_list.append("yelp_reviews/yelp_train_test/using_sw_normal_with_more_we")
yelp_data_list.append("yelp_reviews/yelp_train_test/normal_with_300d_we")
yelp_data_list.append("yelp_reviews/yelp_train_test/normal_with_100d_we")
yelp_data_list.append("yelp_reviews/yelp_train_test/using_sw_normal_with_100d_we")
yelp_data_list.append("yelp_reviews/yelp_train_test/normal_without_weights_100d_we")


# 加载数据
data = numpy.loadtxt(yelp_data_list[6], dtype=float, delimiter=" ")

# 将训练集中的向量和结果分开
training_data, labels = numpy.split(data, (dimension, ), axis=1)
train_vec, test_vec, train_label, test_label = train_test_split(training_data, labels, train_size=0.8, random_state=1, shuffle=True)

train_x1 = training_data[0:500]
train_y1 = labels[0:500]
train_x2 = training_data[500:1000]
train_y2 = labels[500:1000]
train_x3 = training_data[1000:1500]
train_y3 = labels[1000:1500]
train_x4 = training_data[1500:2000]
train_y4 = labels[1500:2000]
train_x5 = training_data[2000:]
train_y5 = labels[2000:]

train_x = numpy.vstack((train_x2, train_x3, train_x4, train_x5))
train_y = numpy.vstack((train_y2, train_y3, train_y4, train_y5))
test_x = train_x1
test_y = train_y1


def feature_selection():
    print("Start training...")

    svc = svm.LinearSVC(C=0.01, penalty='l1', dual=False)
    svc.fit(train_vec, train_label.ravel())

    tree = ExtraTreesClassifier()
    tree.fit(train_vec, train_label.ravel())

    print("Training Accuracy:%.4f" % svc.score(train_vec, train_label))
    print("Testing Accuracy:%.4f" % svc.score(test_vec, test_label))

    model = SelectFromModel(svc, prefit=True)

    rfe = RFE(svc, n_features_to_select=20, )
    rfe.fit(train_vec, train_label.ravel())

    X_train = rfe.transform(train_vec)
    X_test = rfe.transform(test_vec)

    print(X_train.shape)

    clf = svm.SVC(C=0.9, kernel='rbf', gamma=80, decision_function_shape='ovo', )
    clf.fit(X_train, train_label.ravel())
    print("After feature selection...")
    print("Training Accuracy:%.4f" % clf.score(X_train, train_label))
    print("Testing Accuracy:%.4f" % clf.score(X_test, test_label))
    return X_train, X_test


# 将使用一种方法生成的句向量进行训练
def train_svm():
    # 进行特征选择
    # X_train, X_test = feature_selection()

    # 初始化svm并进行训练、预测
    # kernel 值可以为 linear:线性核，主要用于线性可分的情况； rbf:高斯核，用于线性不可分的情况，参数多，结果依赖于参数
    #clf = svm.SVC(C=0.9, kernel='rbf', gamma=80, decision_function_shape='ovo',)
    clf = DecisionTreeClassifier()
    clf = svm.LinearSVC()
    #clf = RandomForestClassifier(n_estimators=10, verbose=2)
    #clf = ExtraTreesClassifier()
    #clf = LogisticRegression(verbose=2)
    print("Start training...")
    clf.fit(train_vec, train_label.ravel())
    accuracy = clf.score(test_vec, test_label.ravel())  # 测试集准确率

    print(accuracy)


# 供word2vec_research调用来分类
def classify(x_train, x_test, y_train, y_test):
    clf = svm.SVC(C=0.9, kernel='rbf', gamma=80, decision_function_shape='ovo')
    clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    #clf = ExtraTreesClassifier()
    #clf = LogisticRegression()
    #clf = svm.LinearSVC()
    print("Start training...")
    clf.fit(x_train, y_train.ravel())
    accuracy = clf.score(x_test, y_test.ravel())  # 测试集准确率
    return accuracy


# 分别计算正例和负例的precision和recall
def cal_precision_recall(y_true, y_predict):
    TP_num = 0  # 实际为正例，被识别为正例的样本数
    FP_num = 0  # 实际为负例，被识别为正例的样本数
    FN_num = 0  # 实际为正例，被识别为负例的样本数
    TN_num = 0  # 实际为负例，被识别为负例的样本数
    for i in range(0, len(y_true)):
        num1 = y_true[i][0]
        num2 = y_predict[i]
        if num1 == 1.0 and num2 == 1.0:
            TP_num += 1
        elif num1 == 0.0 and num2 == 1.0:
            FP_num += 1
        elif num1 == 1.0 and num2 == 0.0:
            FN_num += 1
        elif num1 == 0.0 and num2 == 0.0:
            TN_num += 1
    # 分别计算识别正例和负例的精确率和召回率
    p_precision = float(TP_num)/float(TP_num+FP_num)
    p_recall = float(TP_num)/float(TP_num+FN_num)
    n_precision = float(TN_num)/float(TN_num+FN_num)
    n_recall = float(TN_num)/float(TN_num+FP_num)
    return p_precision, p_recall, n_precision, n_recall


# 计算F1 score
def cal_f1_score(precision,  recall):
    return (2*precision*recall)/(precision+recall)


# 将以不同方法生成的句向量训练SVM进行对比
def train_all_data():
    x = PrettyTable(['Sent2vec method', 'Positive precision', 'Positive recall', 'Positive f1 score',
                     'Negative precision', 'Negative recall', 'Negative f1 score', 'Accuracy'])
    i = 0
    for file in svm_data_folder:
        print("Begin method "+str(i+1)+" training...")
        if i == 0:
            method = 'w2v sum'
        elif i == 1:
            method = 'w2v topic sum'
        elif i == 2:
            method = 'Tfidf weight w2v sum'
        elif i == 3:
            method = 'Tfidf weight w2v topic sum'
        elif i == 4:
            method = 'Tfidf weight w2v topic ed sum'
        elif i == 5:
            method = 'Adjusted-Dims sentence vec'

        data = numpy.loadtxt(file, dtype=float, delimiter=" ")
        # 将训练集中的向量和结果分开
        training_data, labels = numpy.split(data, (dimension,), axis=1)
        train_vec, test_vec, train_label, test_label = train_test_split(training_data, labels, train_size=0.8,
                                                                        random_state=1, shuffle=True)

        clf = svm.SVC(C=0.9, kernel='rbf', gamma=80, decision_function_shape='ovo', )
        clf.fit(train_vec, train_label.ravel())
        accuracy = clf.score(test_vec, test_label)
        y_score = clf.decision_function(test_vec)
        predict_y = []
        for distance in y_score:
            if distance > 0:
                predict_y.append(1.0)
            else:
                predict_y.append(0.0)
        p_precision, p_recall, n_precision, n_recall = cal_precision_recall(test_label, predict_y)
        p_f1_score = cal_f1_score(p_precision, p_recall)
        n_f1_score = cal_f1_score(n_precision, n_recall)

        x.add_row([method, "%.4f" % p_precision, "%.4f" % p_recall, "%.4f" % p_f1_score, "%.4f" % n_precision,
                   "%.4f" % n_recall, "%.4f" % n_f1_score, "%.4f" % accuracy])
        i += 1

    print(x)


# 将不同方法生成的句向量训练决策树进行对比
def train_all_data_with_tree():
    x = PrettyTable(['Sent2vec method', 'Positive precision', 'Positive recall', 'Positive f1 score',
                     'Negative precision', 'Negative recall', 'Negative f1 score', 'Accuracy'])
    i = 0
    for file in svm_data_folder:
        print("Begin method " + str(i + 1) + " training...")
        if i == 0:
            method = 'w2v sum'
        elif i == 1:
            method = 'w2v topic sum'
        elif i == 2:
            method = 'Tfidf weight w2v sum'
        elif i == 3:
            method = 'Tfidf weight w2v topic sum'
        elif i == 4:
            method = 'Tfidf weight w2v topic ed sum'
        elif i == 5:
            method = 'Adjusted-Dims sentence vec'

        data = numpy.loadtxt(file, dtype=float, delimiter=" ")
        # 将训练集中的向量和结果分开
        training_data, labels = numpy.split(data, (dimension,), axis=1)
        train_vec, test_vec, train_label, test_label = train_test_split(training_data, labels, train_size=0.8,
                                                                        random_state=1, shuffle=True)
        # 初始化分类器
        tree = DecisionTreeClassifier()
        tree.fit(train_vec, train_label.ravel())

        forest = RandomForestClassifier()
        forest.fit(train_vec, train_label.ravel())

        accuracy = forest.score(test_vec, test_label)
        predict_y = forest.predict(test_vec)
        p_precision, p_recall, n_precision, n_recall = cal_precision_recall(test_label, predict_y)
        p_f1_score = cal_f1_score(p_precision, p_recall)
        n_f1_score = cal_f1_score(n_precision, n_recall)

        x.add_row([method, "%.4f" % p_precision, "%.4f" % p_recall, "%.4f" % p_f1_score, "%.4f" % n_precision,
                   "%.4f" % n_recall, "%.4f" % n_f1_score, "%.4f" % accuracy])
        i += 1

    print(x)


def train_new_data(train_vec_path, test_vec_path):
    # 加载数据
    train_data = numpy.loadtxt(train_vec_path, dtype=float, delimiter=" ")
    test_data = numpy.loadtxt(test_vec_path, dtype=float, delimiter=" ")
    # 将训练集中的向量和结果分开
    train_x, train_y = numpy.split(train_data, (dimension,), axis=1)
    test_x, test_y = numpy.split(test_data, (dimension,), axis=1)

    clf = svm.SVC(C=0.9, kernel='rbf', gamma=80, decision_function_shape='ovo')
    print("RBF SVM:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)
    clf = DecisionTreeClassifier()
    print("Decision Tree:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)
    clf = RandomForestClassifier()
    print("Random Forest:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)
    clf = ExtraTreesClassifier()
    print("Extra Tree:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)
    clf = LogisticRegression()
    print("Logistic Regression:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)
    clf = svm.LinearSVC()
    print("Linear SVM:")
    train_test_for_one_model(clf, train_x, train_y, test_x, test_y)


def train_test_for_one_model(clf, train_x, train_y, test_x, test_y):
    print("Start training...")
    avg_sum = 0
    for i in xrange(0, 10):
        clf.fit(train_x, train_y.ravel())
        accuracy = clf.score(test_x, test_y.ravel())  # 测试集准确率
        if avg_sum == accuracy:
            return
        avg_sum += accuracy
        print("Train " + str(i) + " round:" + str(accuracy))

    print("Accuracy:" + str(avg_sum / 10.0))
    print("==============================")


train_vec_path = "using_sw_and_half_freq3/train_vec"
test_vec_path = "using_sw_and_half_freq3/test_vec"
new_train_vec_path = "using_sw_and_half_freq3/train_adjusted_data"
new_test_vec_path = "using_sw_and_half_freq3/test_adjusted_data"

#feature_selection()
train_svm()
#train_all_data()
#train_all_data_with_tree()
#train_new_data(train_vec_path, test_vec_path)
#train_new_data(new_train_vec_path, new_test_vec_path)
