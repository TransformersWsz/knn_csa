#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 2:07 PM
# @Author  : Swift
# @File    : knn.py
# @Brief   : implement the chinese sentiment analysis with knn algorithm, see detail in README

import math


class KNN(object):

    def __init__(self):
        # self._mode = {"unigram": 1, "bigram": 2}
        pass

    def top_k(self, sorted_arr: list, K: int, element: tuple) -> None:
        """
        维护一个大小为 K 的有序数组
        :param sorted_arr: 有序数组
        :param size: 数组大小：K
        :param element: 待插入的元素
        :return: None
        """
        flag = 0
        for index, item in enumerate(sorted_arr):
            if element[0] >= item[0]:
                flag = 1
                break
        if flag == 1:
            if len(sorted_arr) == K:
                sorted_arr.pop()
            sorted_arr.insert(index, element)
        else:
            if len(sorted_arr) != K:
                sorted_arr.append(element)

    def segment(self, sentence: str, mode: int) -> list:
        """
        对中文语句进行分词
        :param sentence: 一条中文语句
        :param mode: 分词模式，有 {"unigram": 1, "bigram": 2}
        :return: 分词后的列表
        """
        word_list = []
        for i in range(len(sentence)//mode+1):
            word = sentence[i*mode:(i+1)*mode]
            if len(word):
                word_list.append(word)
        return word_list

    def get_hybrid_segment_list(self, sentence: str) -> list:
        """
        使用 unigram 和 bigram 两种模式对中文语句进行分词，并将分词结果合并
        :param sentence: 一条中文语句
        :return: 两种分词模式的结果合并
        """
        unigram_segment = self.segment(sentence, 1)
        bigram_segment = self.segment(sentence, 2)
        return unigram_segment + bigram_segment

    def modulus(self, d: dict) -> float:
        """
        计算向量的模
        :param d: 向量
        :return: 模
        """
        sum = 0
        for val in d.values():
            sum += val * val
        return math.sqrt(sum)

    def cosine(self, a: dict, b: dict) -> float:
        mod_a = self.modulus(a)
        mod_b = self.modulus(b)

        numerator = 0
        for pos, val in a.items():
            if pos in b:
                numerator += val * b[pos]
        return numerator / (mod_a * mod_b)

    def construct_sentence_vector(self, segment_a: list, segment_b: list) -> list:
        """
        根据分好词的列表构造句向量
        :param segment_a: 分词列表a
        :param segment_b: 分词列表b
        :return: a 和 b 的句向量
        """
        

if __name__ == "__main__":
    a = {"0": 1, "2": 4}
    b = {"1:": 1, "2": 2}

    solution = KNN()
    print(solution.cosine(a, b))
    sentence = "难得你这么闲？"
    result = solution.get_hybrid_segment_list(sentence)
    print(result)
