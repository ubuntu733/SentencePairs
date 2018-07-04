# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2018 Hisense, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import ConfigParser
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import csv
import math
import json

from scipy import spatial


import numpy as np
import pandas as pd

def generate_idf(data_file):
    idf = {}
    len = 0
    with open(data_file,'r') as fin:
        for line in fin:
            len +=1
            line = unicode(line).strip().split('\t')
            for line_list in [line[1].split(), line[2].split()]:
                for word in line_list:
                    idf[word] = idf.get(word, 0) +1
    for word in idf:
        idf[word] = math.log(len / (idf[word] + 1.)) / math.log(2.)
    return idf

def load_word_embedding(data_file):
    vector = {}
    with open(data_file, 'r') as fin:
        fin.readline()
        for line in fin:
            line = line.decode('utf-8').strip()
            parts = line.split(' ')
            word = parts[0]
            vector[word] = np.array(parts[1:], dtype='float32')
    return vector


class WordEmbeddingAveDis(object):

    def __init__(self, word_embedding_fp):
        self.we_dic = load_word_embedding(word_embedding_fp)
        self.we_len = len(self.we_dic.values()[0])

    def extract_sentence_vector(self, sentence):
        words = unicode(sentence).split(' ')
        vector = np.array(self.we_len * [0.])
        for word in words:
            if word in self.we_dic:
                vector = vector + self.we_dic[word]
        return vector

    def extract_sentences_score(self, sent1, sent2):
        sent1_vec = self.extract_sentence_vector(sent1)
        sent2_vec = self.extract_sentence_vector(sent2)
        result = {}
        result['cosine'] = spatial.distance.cosine(sent1_vec, sent2_vec)
        result['euclidean'] = spatial.distance.euclidean(sent1_vec, sent2_vec)
        result['sent1_vec'] = sent1_vec
        result['sent2_vec'] = sent2_vec
        return result


class WordEmbeddingTFIDFAveDis(object):

    def __init__(self, word_embedding_fp, qid2q_fp):
        self.idf = generate_idf(qid2q_fp)
        self.we_dic = load_word_embedding(word_embedding_fp)
        self.we_len = len(self.we_dic.values()[0])

    def extract_sentence_vector(self, sentence):
        words = unicode(sentence).split(' ')
        vec = np.array(self.we_len * [0.])
        words_cnt = {}
        for word in words:
            words_cnt[word] = words_cnt.get(word, 0.) + 1.
        for word in words_cnt:
            if word in self.we_dic:
                vec += self.idf.get(word, 0.) * words_cnt[word] * self.we_dic[word]
        return vec

    def extract_sentences_score(self, sent1, sent2):
        sent1_vec = self.extract_sentence_vector(sent1)
        sent2_vec = self.extract_sentence_vector(sent2)
        result = {}
        result['cosine'] = spatial.distance.cosine(sent1_vec, sent2_vec)
        result['euclidean'] = spatial.distance.euclidean(sent1_vec, sent2_vec)
        result['sent1_vec'] = sent1_vec
        result['sent2_vec'] = sent2_vec
        return result


if __name__ == "__main__":
    word1 = WordEmbeddingAveDis('../alibaba/fasttext.txt.vec')
    print(word1.extract_sentences_score('蚂蚁 花呗','蚂蚁 借呗'))
    word2 = WordEmbeddingTFIDFAveDis(word_embedding_fp='../alibaba/fasttext.txt.vec', qid2q_fp='../alibaba/logs8/train.csv')
    print(word2.extract_sentences_score('蚂蚁 花呗', '蚂蚁 借呗'))