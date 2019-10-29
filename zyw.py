from random import shuffle
import numpy as np
import time
import random

classify_dict = {
    '100':0,
    '101':1,
    '102':2,
    '103':3,
    '104':4,
    '106':5,
    '107':6,
    '108':7,
    '109':8,
    '110':9,
    '112':10,
    '113':11,
    '114':12,
    '115':13,
    '116':14
}
word_dict = {}
j = -1


class Node():
    def __init__(self, content, index):
        self.content = content
        self.index = index

#将句子转换成unigram字符数组
def word_unigram(str):

    global j
    for i in ',.?:-!，。？！-、 \n\t':
        str = str.replace(i, '')
    list = []
    for i in range(len(str)):
        list.append(str[i])
        if str[i] not in word_dict:
            j += 1
            word_dict[str[i]] = j
    return list
def word_bigram(str):
    global j
    for i in ',.?:-!，。？！-、 \n\t':
        str = str.replace(i, '')
    list = []
    for i in range(len(str)-1):
        list.append(str[i:i+2])
        if str[i:i+2] not in word_dict:
            j += 1
            word_dict[str[i:i+2]] = j
    return list

#计算两个向量之间的余弦相似度
def get_cos(x, y):
    dot_xy = 0
    len1 = len(x)
    len2 = len(y)
    if len1 <= len2:
        for key in x:
            if key in y:
                dot_xy += 1
    else:
        for key in y:
            if key in x:
                dot_xy += 1
    if dot_xy == 0:
        return 0
    return dot_xy * 1.0 / ((len1 * len2) ** 0.5)
#找出k个最大的余弦相似度
def sort(list, n, m, k):
    if k >= m - n:
        return list
    t = list[n]
    i = n
    j = m - 1
    while i < j:
        while i < j and list[j].content < t.content:
            j -= 1
        if i < j:
            list[i] = list[j]
            i += 1
        while i < j and list[i].content > t.content:
            i += 1
        if i < j:
            list[j] = list[i]
            j -= 1
    list[i] = t
    if i < k:
        list = sort(list, i + 1, m, k - i - 1)
    elif i > k:
        list = sort(list, n, i, k)
    return list

#给文件的每行句子创建布尔向量
def create_vector(train_neg_path, train_pos_path, test_neg_path, test_pos_path):
    sentence_train_word = []
    sentence_test_word = []
    train = []
    test = []

    train_neg = open(train_neg_path, 'r', encoding='UTF-8')
    train_pos = open(train_pos_path, 'r', encoding='UTF-8')
    test_neg = open(test_neg_path, 'r', encoding='UTF-8')
    test_pos = open(test_pos_path, 'r', encoding='UTF-8')
    contents1 = train_neg.readlines()
    contents2 = train_pos.readlines()
    contents3 = test_neg.readlines()
    contents4 = test_pos.readlines()
    train_neg.close()
    train_pos.close()
    test_neg.close()
    test_pos.close()

    train_text = contents1 + contents2
    train_labels = len(contents1) * [0] + len(contents2) * [1]
    test_text = contents3 + contents4
    test_labels = len(contents3) * [0] + len(contents4) * [1]


    for content in train_text:
        # sentence_temp = word_unigram(str(content)) + word_bigram(str(content))
        sentence_temp = word_unigram(str(content))
        # sentence_temp = word_bigram(str(content))
        sentence_train_word.append(sentence_temp)

    for content in test_text:
        # sentence_temp = word_unigram(str(content)) + word_bigram(str(content))
        sentence_temp = word_unigram(str(content))
        # sentence_temp = word_bigram(str(content))
        sentence_test_word.append(sentence_temp)

    for content in sentence_train_word:
        data = {}
        for item in content:
            data[word_dict[item]] = 1
        train.append(data)
    for content in sentence_test_word:
        data = {}
        for item in content:
            data[word_dict[item]] = 1
        test.append(data)
    return train, train_labels, test, test_labels

#计算测试集的分类以及最终准确率：穷举法求出k个最大相似度
def KNN_model(trainArray, trainLabels, testArray, testLabels, k):
    pre_labels = []
    for m in range(len(testArray)):
        test_cos = []
        k_label = np.zeros(2)
        for i in range(int(len(trainArray))):
            test_cos.append(Node(get_cos(testArray[m], trainArray[i]), trainLabels[i]))
        test_cos = sort(test_cos, 0, len(test_cos), k)
        for i in range(k):
            k_label[test_cos[i].index] += 1
        if k_label[0] > k_label[1]:
            pre_labels.append(0)
        else:
            pre_labels.append(1)
    accuracy = 0
    for i in range(len(testLabels)):
        if pre_labels[i] == testLabels[i]:
            accuracy += 1
    for i in range(len(testLabels)):
        print(pre_labels[i],' -- ',testLabels[i])
    print('accuracy: ', accuracy * 1.0 / len(testLabels))

def main():
    start = time.time()

    print('正在创建样本和测试向量...')
    trainArray, trainLabels, testArray, testLabels = create_vector('train_neg.txt', 'train_pos.txt', 'test_neg.txt', 'test_neg.txt')
    print('向量创建完毕，正在进行knn算法...')
    KNN_model(trainArray, trainLabels, testArray, testLabels, 10)
    end = time.time()
    print('总时间: ', end-start)

if __name__ == '__main__':
    main()
