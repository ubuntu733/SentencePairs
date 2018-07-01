#/usr/bin/env python
#coding=utf-8

import jieba
import sys
import pickle
from scipy.spatial.distance import cosine
import scipy.sparse as ssp
from preprocess import change_sentence
from tf_kdl_weight import TFKLD


def my_tokenizer(x):
    return x.split()


def process(inpath, outpath):

    tfkdl_params_path = "./tfkdl_params.pickle"
    tfkdl_params = pickle.load(open(tfkdl_params_path, "rb"))

    countvector_model = tfkdl_params["countvector_model"]
    tfkdl_weight = tfkdl_params["weight"]

    tfkdl_object = TFKLD(None)

    jieba.load_userdict("./dict")

    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sent1, sent2 = line.strip().split('\t')
            words1 = " ".join([ w.encode("utf-8") for w in jieba.cut(change_sentence(sent1), HMM=False) if w.strip() ])
            words2 = " ".join([ w.encode("utf-8") for w in jieba.cut(change_sentence(sent2), HMM=False) if w.strip() ])

            sent1_cv = countvector_model.transform([words1])
            sent1_cv = ssp.lil_matrix(sent1_cv).todense()
            sent1_tfkdl = tfkdl_object.weighting_internal(sent1_cv, tfkdl_weight)

            sent2_cv = countvector_model.transform([words2])
            sent2_cv = ssp.lil_matrix(sent2_cv).todense()
            sent2_tfkdl = tfkdl_object.weighting_internal(sent2_cv, tfkdl_weight)

            cosine_sim = 1 - cosine(sent1_tfkdl[0], sent2_tfkdl[0])

            if cosine_sim >= 0.5:
                fout.write(lineno + '\t1\n')
            else:
                fout.write(lineno + '\t0\n')


if __name__ == '__main__':
    #process(sys.argv[1], sys.argv[2])
    process("./data_tfkdl.txt", "./output.xt")
