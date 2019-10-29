#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 2:07 PM
# @Author  : Swift
# @File    : knn.py
# @Brief   : implement the chinese sentiment analysis with knn algorithm, see detail in README

import math
import time
import re


class KNN(object):

    def __init__(self):
        self._word_dict = {}

    def extract_chinese(self, sentence: str) -> str:
        """过滤掉句子中的非中文字符"""
        pattern = "[\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]"
        regex = re.compile(pattern)
        l = regex.findall(sentence)
        return "".join(l)

    def segment(self, sentence: str, mode: int) -> dict:
        """
        对中文语句进行分词
        :param sentence: 一条中文语句
        :param mode: 分词模式，有 {"unigram": 1, "bigram": 2}
        :return: 句向量
        """
        sentence_vector = {}
        for i in range(len(sentence)+1-mode):
            word = sentence[i:(i+mode)]
            if word not in self._word_dict:
                self._word_dict[word] = len(self._word_dict)
            sentence_vector["{}".format(self._word_dict[word])] = 1
        return sentence_vector

    def cosine(self, a: dict, b: dict) -> float:
        """计算两个向量的余弦距离"""
        mod_a = math.sqrt(len(a))
        mod_b = math.sqrt(len(b))

        numerator = 0
        for pos, val in a.items():
            if pos in b:
                numerator += val * b[pos]
        return numerator / (mod_a * mod_b)

    def get_sentence_vector(self, sentence: str) -> dict:
        """
        获取文本向量
        :param sentence: 文本
        :return: unigram 和 bigram 组成的 dict
        """
        unigram_d = self.segment(sentence, 1)
        bigram_d = self.segment(sentence, 2)
        return dict(unigram_d, **bigram_d)

    def similarity(self, sentence_a: str, sentence_b: str) -> float:
        """
        计算两个句子的相似度
        :param sentence_a: 句子a
        :param sentence_b: 句子b
        :return: 余弦距离
        """
        dict_a = self.get_sentence_vector(sentence_a)
        dict_b = self.get_sentence_vector(sentence_b)
        distance = self.cosine(dict_a, dict_b)
        return distance

    def read_file(self, filepath: str, polarity: str) -> list:
        """
        读取文件，返回句子和类别构成的列表
        :param filepath: 文件路径
        :return: [(sentence, category), ...]
        """
        result = []
        with open(filepath, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                line = self.extract_chinese(line)
                if len(line):
                    result.append((self.extract_chinese(line), polarity))
        return result

    def top_k(self, sorted_arr: list, K: int, element: tuple) -> None:
        """
        维护一个大小为 K 的有序数组
        :param sorted_arr: 有序数组
        :param size: 数组大小：K
        :param element: 待插入的元素
        :return: None
        """
        flag = 0
        for idx, item in enumerate(sorted_arr):
            if element[0] > item[0]:
                flag = 1
                break
        if flag == 1:
            if len(sorted_arr) == K:
                sorted_arr.pop()
            sorted_arr.insert(idx, element)
        else:
            if len(sorted_arr) != K:
                sorted_arr.append(element)

    def get_most_topK(self, sorted_arr: list) -> str:
        """获取K个元素中出现次数最多的类别"""
        d = {}
        for distance, category in sorted_arr:
            if category not in d:
                d[category] = 1
            else:
                d[category] += 1
        max_cate = None
        max_freq = 0
        for cate, frequency in d.items():
            if max_freq < frequency:
                max_cate = cate
                max_freq = frequency
        return max_cate

    def classify(self, train: list, test: list, K: int) -> float:
        """
        测试
        :param train: 训练集
        :param test: 测试集
        :param K: 邻居个数
        :return: 准确率
        """
        correct_num = 0    # 预测类别正确的个数
        wrong_num = 0    # 预测类别错误的个数
        for test_sentence, test_category in test:
            sorted_arr = []
            for train_sentence, train_category in train:
                distance = self.similarity(test_sentence, train_sentence)
                self.top_k(sorted_arr, K, (distance, train_category))
            predict_cate = self.get_most_topK(sorted_arr)

            if test_category == predict_cate:
                correct_num += 1
            else:
                wrong_num += 1
            print("{} {} {} {} 已预测正确个数为 {} 已预测错误个数为 {}".format(test_sentence, test_category, predict_cate, test_category == predict_cate, correct_num, wrong_num))
        return correct_num/len(test)
        

if __name__ == "__main__":
    start = time.time()

    solution = KNN()
    train_pos = "./corpus/train_pos.txt"
    train_neg = "./corpus/train_neg.txt"
    test_pos = "./corpus/test_pos.txt"
    test_neg = "./corpus/test_neg.txt"

    train_set = solution.read_file(train_pos, "positive") + solution.read_file(train_neg, "negative")
    test_set = solution.read_file(test_pos, "positive") + solution.read_file(test_neg, "negative")

    accuracy = solution.classify(train_set, test_set, 10)
    print("===========================================")
    print("accuracy: {}".format(accuracy))
    print("===========================================")

    end = time.time()
    print("exe_time: {}".format(end-start))


