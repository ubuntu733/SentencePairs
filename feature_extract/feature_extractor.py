#/usr/bin/env python
#coding="utf-8"


"""
提供特征抽取的对外接口
"""

import os
import sys
import pickle
from sklearn.feature_extraction import DictVectorizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from feature_extract.extractor_util import *


class FeatureExtractor(object):
    def __init__(self, semantic_model_path_dict={}):
        """
        :param tfkdl_path:
        :param lda_path:
        """
        self.semantic_model_path_dict = semantic_model_path_dict
        pass

    def load_semantic_model(self):
        """
        加载语义模型
        :return:
        """
        pass

    def extract_sentpair_feature(self, sent_1, sent_2):
        """
        抽取所有的特征，进行组合
        :param sent_1: sentence string splitted by space, basic unit is word
        :param sent_2:
        :return:
        """
        # get the character sentence
        sent_1_character = " ".join(list("".join(sent_1.split())))
        sent_2_character = " ".join(list("".join(sent_2.split())))

        # extract the feature set
        feature_set_dict = {}
        feature_set_dict.update(extract_sent_len(sent_1, sent_2, "word"))

        feature_set_dict.update(extract_sent_interaction(sent_1, sent_2, "word"))
        feature_set_dict.update(extract_sent_interaction(sent_1_character, sent_2_character, "character"))

        feature_set_dict.update(extract_lexical_overlap_feature(sent_1, sent_2, "word"))
        feature_set_dict.update(extract_lexical_overlap_feature(sent_1_character, sent_2_character, "character"))

        feature_set_dict.update(extract_word_align_feature(sent_1, sent_2, "word"))

        feature_set_dict.update(extract_translation_eval_metrics(sent_1, sent_2, "word"))
        feature_set_dict.update(extract_translation_eval_metrics(sent_1_character, sent_2_character, "word"))

        # TODO
        for semantic_model_name, semantic_model_path in self.semantic_model_path_dict.items():
            pass

        return feature_set_dict

    def extract_corpus_feature(self, dataset):
        """
        :param dataset:list of tuple(sent_1, sent_2--string splited by space)
        :return:list of dict, feature vector matrix(n_samples, n_features)
        """
        dataset_features = []
        for sent1, sent2 in dataset:
            dataset_features.append(self.extract_sentpair_feature(sent1, sent2))
        dict_vectorizer = DictVectorizer(sparse=False)

        return dataset_features, dict_vectorizer.fit_transform(dataset_features)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()

    dataset = []
    with open("../data/data_tfkdl.txt", "rb") as reader:
        for line in reader:
            line = line.decode("utf-8")
            line_list = line.strip().split("\t")
            dataset.append((line_list[1], line_list[2]))

    dataset_feature, dataset_vector = feature_extractor.extract_corpus_feature(dataset)

    print(dataset_feature)
    print(dataset_vector)