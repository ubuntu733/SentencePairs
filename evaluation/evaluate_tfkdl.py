import os
import sys
import scipy.sparse as ssp
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from pickle import load, dump
from feature_extract.tf_kdl_weight import TFKLD
from feature_extract.dimention_reduction import DimReduction
from feature_extract.preprocess import my_tokenizer


if __name__ == "__main__":

    tfkdl_params_path = "../data/tfkdl_params.pickle"
    tfkdl_params = load(open(tfkdl_params_path, "rb"))

    countvector_model = tfkdl_params["countvector_model"]
    tfkdl_weight = tfkdl_params["weight"]

    tfkdl_object = TFKLD(None)

    dataset_path = "../data/data_tfkdl.txt"
    datasetT, datasetL = tfkdl_object.loadtext(dataset_path)
    datasetT = countvector_model.transform(datasetT)
    datasetT = ssp.lil_matrix(datasetT).todense()

    datasetT_weight = tfkdl_object.weighting_internal(datasetT, tfkdl_weight)

    #dr = DimReduction(datasetT_weight, 200)
    #W, H = dr.svd()
    W = datasetT_weight

    nrow, ndim = W.shape

    pred_val = []
    cosine_sim_list = []

    for index in range(0, nrow, 2):
        sent_1 = W[index, :]
        sent_2 = W[index + 1, :]

        label = datasetL[index // 2]

        cosine_val = 1 - cosine(sent_1, sent_2)

        cosine_sim_list.append(cosine_val)

        if cosine_val >= 0.5:
            pred_val.append(1)
        else:
            pred_val.append(0)

    dump(cosine_sim_list, open("../data/tfkdl_pred_withoutdr", "wb"))

    print(classification_report(datasetL, pred_val))
