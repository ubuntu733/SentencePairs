#/usr/bin/env python
#coding=utf-8

"""
提供特征抽取的对外接口
"""

import os
import sys
import pickle
from multiprocessing import Pool
from sklearn.feature_extraction import DictVectorizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import copy_reg
from types import MethodType

from feature_extract.extractor_util import *
from feature_extract.tf_kdl_weight import TFKLD
from feature_extract.preprocess import my_tokenizer
from feature_extract.semantic_feature_extractor import *
import time


class FeatureExtractor(object):
    def __init__(self, model_path_dict):
        """
        :param tfkdl_path:
        :param lda_path:
        """
        self.load_semantic_model(model_path_dict)
        pass

    def load_semantic_model(self, model_path_dict):
        """
        加载语义模型
        :return:
        """
        tfkdl_params_path = model_path_dict["tfkdl_param_path"]
        tfkdl_params = pickle.load(open(tfkdl_params_path, "rb"))
        self.tfkdl_weight = tfkdl_params["weight"]
        self.tfkdl_counterizer = tfkdl_params["countvector_model"]
        self.tfkdl_object = TFKLD(None)

        self.countvectorizer_obj = pickle.load(open(model_path_dict["counter_vectorizer_path"], "rb"))
        self.tfidf_obj = pickle.load(open(model_path_dict["tfidf_model_path"], "rb"))

    def extract_sentpair_feature(self, label, sent_1, sent_2):
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
        feature_set_dict.update(extract_translation_eval_metrics(sent_1_character, sent_2_character, "character"))

        # extract the semantic similarity feature
        feature_set_dict["tfidf_sim"] = cal_tfidf_sim(self.countvectorizer_obj, self.tfidf_obj, sent_1, sent_2)
        feature_set_dict["tfkdl_sim"] = cal_tfkdl_sim(self.tfkdl_object, self.tfkdl_weight,
                                                      self.tfkdl_counterizer, sent_1, sent_2)

        feature_set_dict["label"] = label

        return feature_set_dict

    def extract_corpus_feature(self, dataset_path, sent_column=[1, 2], process="train"):
        """
        :param dataset:list of tuple(sent_1, sent_2--string splited by space)
        :return:list of dict, feature vector matrix(n_samples, n_features)
        """
        dataset_features = []
        count = 0

        with open(dataset_path, "rb") as reader:
            for line in reader:
                line = line.decode("utf-8")
                line_list = line.strip().split("\t")

                dataset_features.append(self.extract_sentpair_feature(line_list[sent_column[0]], line_list[sent_column[1]]))

                count += 1
                if count % 1000 == 0:
                    print("===there have tacked count is===", count)

        if process == "train":
            dict_vectorizer = DictVectorizer(sparse=False)
            return dict_vectorizer, dict_vectorizer.fit_transform(dataset_features)
        else:
            return dataset_features


    def process_wapper(self, args):
        return self.extract_sentpair_feature(args[0], args[1], args[2])


    def extract_corpus_feature_multiprocessing(self, dataset,  process="train", python_v="py3"):
        """
        :param dataset:list of tuple(sent_1, sent_2--string splited by space)
        :return:list of dict, feature vector matrix(n_samples, n_features)
        """
        dataset_features = []

        """
        dataset = []
        with open(dataset_path, "rb") as reader:
            for line in reader:
                line = line.decode("utf-8")
                line_list = line.strip().split("\t")
                dataset.append((int(line_list[sent_column[0]]), line_list[sent_column[1]], line_list[sent_column[2]]))
        """
        # TODO python2
        pool = Pool(4)
        if python_v == "py3":
            dataset_features = pool.starmap(self.extract_sentpair_feature, dataset)

        elif python_v == "py2":
            dataset_features = pool.map(self.process_wapper, dataset)

        pool.close()
        pool.join()

        if process == "train":
            dict_vectorizer = DictVectorizer(sparse=False)
            return dict_vectorizer, dict_vectorizer.fit_transform(dataset_features)
        else:
            return dataset_features


def run_corpus_extractor_multiprocessing(class_instance, dataset_path, sent_column=[0, 1 ,2], process="train", python_v="py3"):
    """
    """

    return class_instance.extract_corpus_feature_multiprocessing(dataset_path, sent_column, process, python_v)

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


if __name__ == "__main__":
    # TODO multiprocessing tackle

    parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"

    model_path_dict = {}
    model_path_dict["tfkdl_param_path"] = os.path.join(parent_dir, "data/m_result/tfkdl_params_train.pickle")
    model_path_dict["counter_vectorizer_path"] = os.path.join(parent_dir, "data/m_result/train_counter_vectorizer.model")
    model_path_dict["tfidf_model_path"] = os.path.join(parent_dir, "data/m_result/train_tfidf_vectorizer.model")

    """
    # for complete data
    model_path_dict["tfkdl_param_path"] = os.path.join(parent_dir, "data/m_result/tfkdl_params_complete.pickle")
    model_path_dict["counter_vectorizer_path"] = os.path.join(parent_dir, "data/m_result/complete_counter_vectorizer.model")
    model_path_dict["tfidf_model_path"] = os.path.join(parent_dir, "data/m_result/complete_tfidf_vectorizer.model")
    """

    feature_extractor = FeatureExtractor(model_path_dict)

    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)

    start_time = time.time()

    train_dataset_path = "../data/ori_data/train_process.csv"
    dict_vectorizer, dataset_vector = feature_extractor.extract_corpus_feature_multiprocessing(train_dataset_path, python_v="py2")
    #dict_vectorizer, dataset_vector = run_corpus_extractor_multiprocessing(feature_extractor, train_dataset_path, python_v="py2")
    pickle.dump(dict_vectorizer, open("../data/m_result/dict_vectorizer.model.t", "wb"), 2)
    pickle.dump(dataset_vector, open("../data/ori_data/train.featurematrix.data.t", "wb"), 2)
    print(dict_vectorizer.feature_names_)
    print(dataset_vector[:1])
    print("===========dataset shape================", dataset_vector.shape)
    print("===time using is ===", time.time() - start_time)

    """
    """

    dict_vectorizer = pickle.load(open("../data/m_result/dict_vectorizer.model.t", "rb"))

    dev_dataset_path = "../data/ori_data/dev_process.csv"
    dev_dataset_features = feature_extractor.extract_corpus_feature_multiprocessing(dev_dataset_path, process="dev", python_v="py2")
    #dev_dataset_features = run_corpus_extractor_multiprocessing(feature_extractor, train_dataset_path, python_v="py2")
    dev_dataset_vector = dict_vectorizer.transform(dev_dataset_features)

    pickle.dump(dev_dataset_vector, open("../data/ori_data/dev.featurematrix.data.t", "wb"), 2)
    print("===========dev dataset shape================", dev_dataset_vector.shape)

    """
    # ------------------------------两部分需要分开处理------------------------------------
    dataset_path = "../data/ori_data/complete_process.csv"
    dataset_features = feature_extractor.extract_corpus_feature(dataset_path, process="whole")
    dataset_vector = dict_vectorizer.transform(dataset_features)
    pickle.dump(dataset_vector, open("../data/ori_data/complete.featurematrix.data", "wb"), 2)
    print("===========dataset shape================", dataset_vector.shape)
    """
