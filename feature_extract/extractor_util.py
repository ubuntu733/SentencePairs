#/usr/bin/env python
#coding=utf-8

"""
定义特征抽取相关的util function
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np

from mt_metrics.bleu import compute_bleu
from mt_metrics.meteor import Meteor
from mt_metrics.nist import NISTScore
from mt_metrics.rouge import rouge
from mt_metrics.ter import ter


def extract_sent_len(sent_1, sent_2, type="word"):
    """
    sentence length and ratio feature
    :param sent:
    :return:
    """
    sent_1 = sent_1.split(" ")
    sent_2 = sent_2.split(" ")

    sent_len_dict = {}
    sent_len_dict["sent_1_len_%s" % type] = len(sent_1)
    sent_len_dict["sent_2_len_%s" % type] = len(sent_2)

    sent_diff = abs(len(sent_1) - len(sent_2))
    sent_len_dict["sent_diff_%s" % type] = sent_diff
    sent_len_dict["sent_logdiff_%s" % type] = math.log1p(sent_diff )

    sent_len_dict["sent_ratio_%s" % type] = len(sent_1) * 1.0 / (len(sent_2) + 1e-6)
    sent_len_dict["sent_logratio_%s" % type] = math.log1p(len(sent_1) * 1.0 / (len(sent_2) + 1e-6))

    return sent_len_dict


def extract_sent_interaction(sent_1, sent_2, type="word"):
    """
    sentence interaction lenght feature, with word/character n-gram
    :param sent:
    :return:
    """
    sent_1 = sent_1.split(" ")
    sent_2 = sent_2.split(" ")

    sent_len_dict = {}

    sent_len_dict["union_len_%s" % type] = len(set(sent_1) | set(sent_2))
    sent_len_dict["interaction_len_%s" % type] = len(set(sent_1) & set(sent_2))

    return sent_len_dict


def extract_n_gram(sentence, ngrams=1):
    """
    :param sentence: 分词后的string
    :param ngrams:
    :return:
    """
    return ["".join(sentence[i:i+ngrams]) for i in range(len(sentence)-ngrams+1)]


def extract_lexical_overlap_feature(sent_1, sent_2, type="word"):
    """
    9种特征，产生unigram、bigram、trigram
    lexical overlap feature with precision recal and f1
    reference:
        Paraphrase identification and semantic text similarity analysis in Arabic news tweets using lexical, syntactic, and semantic features
    :param sent_1: string
    :param sent_2:
    :param type:
    :return:
    """

    sent_1 = sent_1.split()
    sent_2 = sent_2.split()

    lexical_overlap_feature = {}
    for n_gram in range(1, 4):
        sent_1_n_grams = extract_n_gram(sent_1, n_gram)
        sent_2_n_grams = extract_n_gram(sent_2, n_gram)

        overlap_len_n_grams = len(set(sent_1_n_grams) & set(sent_2_n_grams))

        precision_n_gram = overlap_len_n_grams * 1.0 / (len(sent_1_n_grams) + 1e-6)

        recall_n_gram = overlap_len_n_grams * 1.0 / (len(sent_2_n_grams) + 1e-6)

        f1_n_gram = 2.0 * precision_n_gram * recall_n_gram / (precision_n_gram + recall_n_gram + 1e-6)

        lexical_overlap_feature["precision_%d_gram_%s" %(n_gram, type)] = precision_n_gram
        lexical_overlap_feature["recall_%d_gram_%s" % (n_gram, type)] = recall_n_gram
        lexical_overlap_feature["f1_%d_gram_%s" % (n_gram, type)] = f1_n_gram

    return lexical_overlap_feature


def extract_word_align_feature(sent_1, sent_2, type="word"):
    """
    extract two type feature based on the levenshtein distance
        ---levenshtein precision recall and f1
        ---align precision recall and f1
    reference:
        Paraphrase identification and semantic text similarity analysis in Arabic news tweets using lexical, syntactic, and semantic features
    :param sent_1: string
    :param sent_2:
    :param type:
    :return:
    """
    sent_1 = sent_1.split()
    sent_2 = sent_2.split()

    def levenshteinDistance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    distance_matrix = np.zeros((len(sent_1), len(sent_2)))

    for index_1, word_1 in enumerate(sent_1):
        for index_2, word_2 in enumerate(sent_2):
            distance_matrix[index_1, index_2] = levenshteinDistance(word_1, word_2)

    align_feature = {}

    row_min_vector = distance_matrix.min(axis=1)
    # print(row_min_vector)
    sum_v = sum(np.sort(row_min_vector)[:5])

    levenstein_pre = sum_v * 1.0 / len(sent_1)
    levenstein_rec = sum_v * 1.0 / len(sent_2)
    align_feature["levenshtein_precision_%s" % type] = levenstein_pre
    align_feature["levenshtein_recall_%s" % type] = levenstein_rec
    align_feature["levenshtein_f1_%s" % type] = 2 * levenstein_pre * levenstein_rec * 1.0 / (levenstein_pre + levenstein_rec + 1e-6)

    sum_align_vector = sum(np.argsort(distance_matrix)[:, 0] - np.array(range(0, len(sent_1))))
    align_pre = sum_align_vector * 1.0 / len(sent_1)
    align_rec = sum_align_vector * 1.0 / len(sent_2)
    align_feature["align_precision_%s" % type] = align_pre
    align_feature["align_recall_%s" % type] = align_rec
    align_feature["align_f1_%s" % type] = 2 * align_pre * align_rec * 1.0 / (
            align_pre + align_rec + 1e-6)

    return align_feature


def extract_translation_eval_metrics(sent1, sent2, type="word"):
    """
    machine translation evaluation metrics
    :param sent_1: string
    :param sent_2:
    :param n_gram:
    :param type:
    :return:
    """
    sent_1 = sent1.split()
    sent_2 = sent2.split()

    mt_eval_metrics_feature = {}

    # calculate the bleu score
    for n_gram in range(1, 5):
        bleu, precisions = compute_bleu([sent_1], [sent_2], max_order=n_gram)
        mt_eval_metrics_feature["bleu_ngram_%d_%s" % (n_gram, type)] = bleu
        mt_eval_metrics_feature["bleu_precision_ngram_%d_%s" % (n_gram, type)] = precisions[-1]

    # calculate the meteor score
    meteor_obj = Meteor()
    for n_gram in range(1, 5):
        meteor_score = meteor_obj.evaluate(sent_1, sent_2, n_gram)
        mt_eval_metrics_feature["meteor_ngram_%d_%s" %(n_gram, type)] = meteor_score

    # calculate nist score
    for n_gram in range(1, 5):
        nist_obj = NISTScore(max_ngram=n_gram)
        nist_obj.append(sent_1, sent_2)
        mt_eval_metrics_feature["nist_ngram_%d_%s" % (n_gram, type)] = nist_obj.score()

    # calculate rouge score
    mt_eval_metrics_feature.update(rouge([sent1], [sent2]))

    # calculate ter score
    mt_eval_metrics_feature["ter_%s" % type] = ter(sent_1, sent_2)

    return mt_eval_metrics_feature


if __name__ == "__main__":
    pass