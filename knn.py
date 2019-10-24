#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 2:07 PM
# @Author  : Swift
# @File    : knn.py
# @Brief   : implement the chinese sentiment analysis with knn algorithm, see detail in README


class KNN(object):

    def __init__(self):
        pass
        # self._mode = {"unigram": 1, "bigram": 2}

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

    def cosine(self):
        pass


if __name__ == "__main__":
    solution = KNN()
    sentence = "难得你这么闲？"
    result = solution.get_hybrid_segment_list(sentence)
    print(result)


