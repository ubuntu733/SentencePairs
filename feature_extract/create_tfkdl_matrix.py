#coding=utf-8

import os
import sys
import pickle
import scipy.sparse as ssp
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from feature_extract.tf_kdl_weight import TFKLD


def my_tokenizer(x):
    return x.split()


def make_tfkdl_matrix(dataset_path, weight_path, save_path):
    """
    """
    tfkdl_obj = TFKLD(dataset_path)

    weight_dict = pickle.load(open(weight_path, "rb"))
    weight = weight_dict["weight"]
    counter_vectorizer = weight_dict["countvector_model"]

    text, label = tfkdl_obj.loadtext(dataset_path)

    trnM = counter_vectorizer.transform(text)
    trnM = ssp.lil_matrix(trnM).todense()

    tfkdl_trnM = tfkdl_obj.weighting_internal(trnM, weight)

    np.save(save_path, tfkdl_trnM)


if __name__ == "__main__":

    weight_path = "../data/m_result/tfkdl_params_train.pickle"

    train_dataset_path = "../data/ori_data/train_process.csv"
    train_tfkdl_matrix_path = "../data/ori_data/tfkdl.train.matrix"

    make_tfkdl_matrix(train_dataset_path, weight_path, train_tfkdl_matrix_path)

    dev_dataset_path = "../data/ori_data/dev_process.csv"
    dev_tfkdl_matrix_path = "../data/ori_data/tfkdl.dev.matrix"

    make_tfkdl_matrix(dev_dataset_path, weight_path, dev_tfkdl_matrix_path)
