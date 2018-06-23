#/usr/bin/env python
#coding=utf-8

import jieba
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import pickle
from feature_extract.feature_extractor import FeatureExtractor
from feature_extract.preprocess import change_sentence
from feature_extract.preprocess import my_tokenizer


def process(inpath, outpath):

    xgb_model = pickle.load(open("./data/xgb.model", "rb"))
    feature_dictvectorizer = pickle.load(open("./data/dict_vectorizer.model", "rb"))

    feature_extractor_obj = FeatureExtractor()

    jieba.load_userdict("./data/dict")
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            sent1 = " ".join([ w for w in jieba.cut(change_sentence(sen1), HMM=False) if w.strip() ])
            sent2 = " ".join([ w for w in jieba.cut(change_sentence(sen2), HMM=False) if w.strip() ])

            sent_pair_featureset = feature_extractor_obj.extract_sentpair_feature(sent1, sent2)

            sent_pair_feature = feature_dictvectorizer.transform(sent_pair_featureset)


            pred_score = xgb_model.predict(sent_pair_feature)

            if pred_score >= 0.5:
                fout.write(lineno + '\t1\n')
            else:
                fout.write(lineno + '\t0\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
    #process("./data/data_test.txt", "./test_pred")
