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


def process(inpath, outpath):

    model_name_list = ["xgb_0.model", "xgboost_1.model", "linear_classifier.model", "xgboost_2.model", "xgboost_3.model"]
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    model_list = []
    for model_name in model_name_list:

        model = pickle.load(open(os.path.join(data_dir, model_name), "rb"))
        model_list.append(model)

    feature_dictvectorizer = pickle.load(open("./data/dict_vectorizer.model", "rb"))


    model_path_dict =  {}

    model_path_dict["tfkdl_param_path"] = os.path.join(data_dir, "tfkdl_params_complete.pickle")
    model_path_dict["counter_vectorizer_path"] = os.path.join(data_dir, "complete_counter_vectorizer.model")
    model_path_dict["tfidf_model_path"] = os.path.join(data_dir, "complete_tfidf_vectorizer.model")

    feature_extractor_obj = FeatureExtractor(model_path_dict)

    jieba.load_userdict("./data/dict")

    dataset_featureset = []
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            sent1 = " ".join([ w for w in jieba.cut(change_sentence(sen1), HMM=False) if w.strip() ])
            sent2 = " ".join([ w for w in jieba.cut(change_sentence(sen2), HMM=False) if w.strip() ])

            sent_pair_featureset = feature_extractor_obj.extract_sentpair_feature(sent1, sent2)

            dataset_featureset.append(sent_pair_featureset)

    dataset_feature_vector = feature_dictvectorizer.transform(dataset_featureset)

    #print("=============", dataset_feature_vector.shape)

    with open(outpath, "w") as fout:

        model_predlabel_list = []

        for model in model_list:

            pred_scores = model.predict(dataset_feature_vector)

            pred_label = pred_scores >= 0.5
            pred_label = pred_label.astype(int)

            model_predlabel_list.append(pred_label)

        model_predlabel_arr = np.array(model_predlabel_list)

        #print(model_predlabel_arr)

        model_pred_pos_num = np.sum(model_predlabel_arr, axis=0)

        for lineno, pred_pos_num in enumerate(model_pred_pos_num):
            if pred_pos_num >= (len(model_name_list) // 2):
                fout.write(str(lineno + 1) + '\t1\n')
            else:
                fout.write(str(lineno + 1)+ '\t0\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
    #process("./data/data_test.txt", "./test_pred")
