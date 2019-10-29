# coding=utf-8
from __future__ import division
import re
import math
import time

class Document(object):
    def __init__(self, word, value):
        self.word = word
        self.value = value

def extract_chinese(sentence):
    """过滤掉句子中的非中文字符"""
    pattern = "[\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]"
    regex = re.compile(pattern)
    l = regex.findall(sentence)
    return "".join(l)


def read_file(filename, flag, words):
    fin =  open(filename, 'r', encoding='utf-8')
    # unigram bigram
    listUB = []
    n = len(words)
    for line in fin:
        # flag = line[0]
        # line = ''.join(re.findall(u"[\u4e00-\u9fa5]+", line))
        line = extract_chinese(line)

        dictu = dict()
        length = len(line)
        if length > 0:
            for i in range(length-1):
                # unigram dict
                if line[i] not in words:
                    words[line[i]] = n
                    dictu[n] = i
                else:
                    dictu[words[line[i]]] = i
                # bigram dict
                str = line[i]+''+line[i+1]
                if str not in words:
                    words[line[i]+''+line[i+1]] = n + 1
                    dictu[n+1] = i
                else:
                    dictu[words[str]] = i
                n += 2
            if line[length-1] not in words:
                words[line[length-1]] = n

        listUB.append(Document(dictu, flag))
    return listUB


# cosine distance
# E(Xi*Yi) / ( sqrt(E(Xi)2) * sqrt(E(Yi)2) )
def cos(inX, inY):
    numerator = 0
    for key in inX:
        if key in inY:
            numerator += 1

    # numerator = len([word for word in inX if word in inY])
    # lenX = sum([x for x in inX if x in inY])
    # lenY = sum([y for y in inY if y in inX])
    denominator = math.sqrt(len(inX) * len(inY))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


# 计算每个测试样本和所有训练样本的距离
# 返回k个距离最近的 距离和标签一一对应的矩阵
def createSimilarVector(trains,test, k):
    cosList = []*k
    temp = []*k
    for train in trains:
        cosine = cos(train.word, test.word)
        if len(cosList) < k:
            temp.append(cosine)
            cosList.append((cosine, train.value))
        elif cosine > min(temp):
            index = temp.index(min(temp))
            temp[index] = cosine
            cosList[index] = (cosine, train.value)
    return cosList


def classify(train, test, k):
    sum = 0
    for i, test_piece in enumerate(test):
        x = dict({0: 0, 1: 0})
        cosList = createSimilarVector(train, test_piece, k)
        result = dict({0: 0, 1: 0})
        for j in range(len(cosList)):
            result[cosList[j][1]] += 1
        x = 0 if result[0] > result[1] else 1
        if x == test_piece.value:
            sum += 1
        print(str(i) + ': ' + str(x) + ' ' + str(test_piece.value))
    accuracy = sum / len(test)
    print('accuracy: ' + str(accuracy))


if __name__ == '__main__':
    start = time.localtime()
    s = time.clock()
    words = dict()
    train_pos = read_file('./corpus/train_pos.txt', 1, words)
    train_neg = read_file('./corpus/train_neg.txt', 0, words)
    test_pos = read_file('./corpus/test_pos.txt', 1, words)
    test_neg = read_file('./corpus/test_neg.txt', 0, words)

    train = train_pos + train_neg
    test = test_pos + test_neg
    # k = 10
    classify(train, test, 10)
    end = time.localtime()
    e = time.clock()
    print('start time: ' + time.strftime('%H:%M:%S',start))
    print('end time: ' + time.strftime('%H:%M:%S', end))
    print('running time: ' + str(e-s) + ' seconds')
