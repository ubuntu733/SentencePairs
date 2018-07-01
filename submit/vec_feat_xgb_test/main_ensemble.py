#/usr/bin/env python
#coding=utf-8

"""
writen for ensemble all the model
author liyantao<lytaaron@163.com>
"""

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


def load_model():
    """
    :return:
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    counter_vectorizer_name = "ngram1.complete_counter_vectorizer.model"
    tfidf_vectorizer_name = "ngram1.complete_tfidf_vectorizer.model"

    counter_vectorizer = pickle.load(open(os.path.join(data_dir, counter_vectorizer_name), "rb"))
    tfidf_vectorizer = pickle.load(open(os.path.join(data_dir, tfidf_vectorizer_name), "rb"))

    feature_dictvectorizer = pickle.load(open("./data/dict_vectorizer.model", "rb"))
    pca = pickle.load(open(os.path.join(data_dir, "complete_pca.model"), "rb"))

    model_path_dict = {}
    model_path_dict["tfkdl_param_path"] = os.path.join(data_dir, "tfkdl_params_complete.pickle")
    model_path_dict["counter_vectorizer_path"] = os.path.join(data_dir, counter_vectorizer_name)
    model_path_dict["tfidf_model_path"] = os.path.join(data_dir, tfidf_vectorizer_name)

    feature_extractor_obj = FeatureExtractor(model_path_dict)
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)

    return counter_vectorizer, tfidf_vectorizer, feature_dictvectorizer, pca, feature_extractor_obj


def get_dl_preds(inpath, outputname="dl_preds.txt"):
    """
    获取深度学习模型的预测结果
    :param inpath:
    :param outpath:
    :return:
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(cur_dir, outputname)

    os.system("python test.py --in_path %s \
            --word_vec_path alibaba/logs8/fasttext.txt.vec \
            --out_path %s \
            --model_prefix alibaba/logs8/SentenceMatch.alibaba" %(inpath, output_path))

    pred_result = []
    with open(output_path, "r") as reader:
        for line in reader:
            line_num, prob_1, prob_0 = line.strip().split("\t")
            pred_result.append([float(unicode(prob_1)),
                                float(unicode(prob_0))])
    return np.array(pred_result)


def get_tfidf_preds(dataset_feature_vector, dataset_for_vec, sent_number_list,
                    counter_vectorizer, tfidf_vectorizer, pca):
    """
     获取基于TFIDF向量模型的预测结果
    :param dataset_feature_vector:
    :param dataset_for_vec:
    :param sent_number_list:
    :param counter_vectorizer:
    :param tfidf_vectorizer:
    :param pca:
    :return:
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    # the tfidf vector
    counter_vector = counter_vectorizer.transform(dataset_for_vec)
    tfidf_vector = tfidf_vectorizer.transform(counter_vector)
    # dimension reduction
    pca_tfidf_vector = pca.transform(tfidf_vector)
    tfidf_matrix = make_matrix(pca_tfidf_vector, sent_number_list)

    # feature and tfidf vector concat
    dataset_feature_vector = np.concatenate((tfidf_matrix, dataset_feature_vector), axis=1)

    model_name_list = ["tfidf_featured.xgb.model"]
    model_preds_list = []
    for model_name in model_name_list:

        model = pickle.load(open(os.path.join(data_dir, model_name), "rb"))

        # 获取预测的概率, num_example * num_class
        pred_scores = model.predict_proba(dataset_feature_vector)
        order_pred_scores = np.array([val for _, val in
                                      sorted(zip(sent_number_list, pred_scores), key=lambda x: x[0])])
        model_preds_list.append(order_pred_scores)

    return model_preds_list


def get_dlembedding_preds(extracted_features, dl_embedding):
    """
    获取基于DL embedding向量模型的预测结果
    :param extracted_features:
    :param dl_embedding:
    :return:
    """
    dl_emb_preds_list = []
    return np.array(dl_emb_preds_list)


def get_mix_preds(dataset_feature_vector, dl_embedding, dataset_for_vec, sent_number_list,
                    counter_vectorizer, tfidf_vectorizer, pca):
    """
    获取基于DL embedding和TFIDF拼接向量的预测结果
    :param dataset_feature_vector:
    :param dl_embedding:
    :param dataset_for_vec:
    :param sent_number_list:
    :param counter_vectorizer:
    :param tfidf_vectorizer:
    :param pca:
    :return:
    """
    mix_preds_list = []
    return mix_preds_list


def make_ensemble_prob(tfidf_model_preds_list, dlemb_model_pred_list,
                                         mix_vec_model_preds_list, dl_model_preds):
    """
    做模型prob sum，然后比较prob_1_sum >= prob_0_sum
    :param tfidf_model_preds_list:
    :param dlemb_model_pred_list:
    :param mix_vec_model_preds_list:
    :param dl_model_preds:
    :return:
    """
    # make element-wise add
    sum_probas = dlemb_model_pred_list
    for preds in tfidf_model_preds_list:
        sum_probas = np.add(sum_probas, preds)
    for preds in dlemb_model_pred_list:
        sum_probas = np.add(sum_probas, preds)
    for preds in mix_vec_model_preds_list:
        sum_probas = np.add(sum_probas, preds)
    # return the label
    return np.argmax(sum_probas, axis=1)


def make_ensemble_count(tfidf_model_preds_list, dlemb_model_pred_list,
                                         mix_vec_model_preds_list, dl_model_preds):
    """
    做label count，预测为1的个数 >= 预测为0的个数
    :param tfidf_model_preds_list:
    :param dlemb_model_pred_list:
    :param mix_vec_model_preds_list:
    :param dl_model_preds:
    :return:
    """
    sum_count = np.argmax(dlemb_model_pred_list, axis=1)
    for preds in tfidf_model_preds_list:
        sum_count = np.add(sum_count, np.argmax(preds, axis=1))
    for preds in dlemb_model_pred_list:
        sum_count = np.add(sum_count, np.argmax(preds, axis=1))
    for preds in mix_vec_model_preds_list:
        sum_count = np.add(sum_count, np.argmax(preds, axis=1))

    total_model_count = len(tfidf_model_preds_list) + len(dlemb_model_pred_list) + \
                  len(mix_vec_model_preds_list) + 1

    return (sum_count >= (total_model_count // 2)).astype(int)


def process(inpath, outpath):
    """
    :param inpath:
    :param outpath:
    :return:
    """
    counter_vectorizer, tfidf_vectorizer, feature_dictvectorizer, \
        pca, feature_extractor_obj = load_model()

    jieba.load_userdict("./data/dict")

    dataset = []
    dataset_for_vec = []
    with open(inpath, 'r') as fin:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            lineno = int(unicode(lineno.strip()))
            sent1 = " ".join([ w for w in jieba.cut(change_sentence(sen1), HMM=False) if w.strip() ])
            sent2 = " ".join([ w for w in jieba.cut(change_sentence(sen2), HMM=False) if w.strip() ])

            dataset.append((lineno, sent1, sent2))
            dataset_for_vec.append(sent1)
            dataset_for_vec.append(sent2)

    # the extracted feature
    dataset_featureset = feature_extractor_obj.extract_corpus_feature_multiprocessing(dataset,
                                                                                      process="dev", python_v="py2")
    sent_number_list = [int(featureset.pop("label", None)) for featureset in dataset_featureset]
    dataset_feature_vector = feature_dictvectorizer.transform(dataset_featureset)

    # 获取DL embedding
    dl_embedding = []

    # 获取预测结果
    tfidf_model_preds_list = get_tfidf_preds(dataset_feature_vector, dataset_for_vec, sent_number_list,
                    counter_vectorizer, tfidf_vectorizer, pca)
    dlemb_model_pred_list = get_dlembedding_preds(dataset_feature_vector,
                                                  dlembedding=dl_embedding)
    mix_vec_model_preds_list = get_mix_preds(dataset_feature_vector, dl_embedding, dataset_for_vec, sent_number_list,
                    counter_vectorizer, tfidf_vectorizer, pca)
    dl_model_preds = get_dl_preds(inpath)
    with open(outpath, "w") as fout:
        # 做ensemble, num_example
        ensemble_results = make_ensemble_prob(tfidf_model_preds_list, dlemb_model_pred_list,
                                         mix_vec_model_preds_list, dl_model_preds)

        #ensemble_results = make_ensemble_count(tfidf_model_preds_list, dlemb_model_pred_list,
        #                                      mix_vec_model_preds_list, dl_model_preds)

        for lineno, pred_result in enumerate(ensemble_results):
            fout.write(str(lineno + 1) + '\t%s\n' %(str(pred_result)))


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
    #process("./data/data_test.txt", "./test_pred")
