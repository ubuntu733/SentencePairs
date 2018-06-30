#/usr/bin/env python
#coding=utf-8

import jieba
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import pickle
import numpy as np
from feature_extract.feature_extractor import FeatureExtractor
from feature_extract.preprocess import change_sentence
from feature_extract.preprocess import my_tokenizer

import copy_reg
from types import MethodType

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


def make_matrix(tfidf_vector, sent_num_list):
    """
    :param tfidf_vector
    :param sent_num_list
    """
    dataset_matrix = []
    for example_num in sent_num_list:
        sent_1_num = 2 * (example_num - 1)
        sent_2_num = sent_1_num + 1

        sent_1_vector = tfidf_vector[sent_1_num]
        sent_2_vector = tfidf_vector[sent_2_num]

        vector_sum = sent_1_vector + sent_2_vector
        vector_diff = abs(sent_1_vector - sent_2_vector)

        vector = np.concatenate((vector_sum, vector_diff))

        dataset_matrix.append(vector)

    return dataset_matrix


def process(inpath, outpath):

    model_name = "tfidf_featured.xgb.model"

    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    model = pickle.load(open(os.path.join(data_dir, model_name), "rb"))

    counter_vectorizer_name = "ngram1.complete_counter_vectorizer.model"
    tfidf_vectorizer_name = "ngram1.complete_tfidf_vectorizer.model"

    counter_vectorizer = pickle.load(open(os.path.join(data_dir, counter_vectorizer_name), "rb"))
    tfidf_vectorizer = pickle.load(open(os.path.join(data_dir, tfidf_vectorizer_name), "rb"))

    feature_dictvectorizer = pickle.load(open("./data/dict_vectorizer.model", "rb"))
    pca = pickle.load(open(os.path.join(data_dir, "complete_pca.model"), "rb"))

    model_path_dict =  {}
    model_path_dict["tfkdl_param_path"] = os.path.join(data_dir, "tfkdl_params_complete.pickle")
    model_path_dict["counter_vectorizer_path"] = os.path.join(data_dir, counter_vectorizer_name)
    model_path_dict["tfidf_model_path"] = os.path.join(data_dir, tfidf_vectorizer_name)


    feature_extractor_obj = FeatureExtractor(model_path_dict)
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)

    jieba.load_userdict("./data/dict")

    dataset_featureset = []

    dataset = []
    dataset_for_vec = []
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            lineno = int(unicode(lineno.strip()))
            sent1 = " ".join([ w for w in jieba.cut(change_sentence(sen1), HMM=False) if w.strip() ])
            sent2 = " ".join([ w for w in jieba.cut(change_sentence(sen2), HMM=False) if w.strip() ])

            dataset.append((lineno, sent1, sent2))

            dataset_for_vec.append(sent1)
            dataset_for_vec.append(sent2)

    # the feature
    dataset_featureset = feature_extractor_obj.extract_corpus_feature_multiprocessing(dataset, process="dev", python_v="py2")

    sent_number_list = [int(featureset.pop("label", None)) for featureset in dataset_featureset]

    dataset_feature_vector = feature_dictvectorizer.transform(dataset_featureset)

    # the tfidf vector
    counter_vector =  counter_vectorizer.transform(dataset_for_vec)
    tfidf_vector =  tfidf_vectorizer.transform(counter_vector)


    pca_tfidf_vector = pca.transform(tfidf_vector)
    tfidf_matrix = make_matrix(pca_tfidf_vector, sent_number_list)

    #print("=============", dataset_feature_vector.shape)
    dataset_feature_vector = np.concatenate((tfidf_matrix, dataset_feature_vector), axis=1)
    with open(outpath, "w") as fout:

        pred_scores = model.predict(dataset_feature_vector)

        lineno_pred_list = sorted(zip(sent_number_list, pred_scores), key=lambda x: x[0])

        for lineno, pred_score in lineno_pred_list:

            if pred_score >= 0.5:
                fout.write(str(lineno) + '\t1\n')
            else:
                fout.write(str(lineno)+ '\t0\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
    #process("./data/data_test.txt", "./test_pred")
