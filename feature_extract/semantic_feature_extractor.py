#coding=utf-8

"""
calculate semantic feature:
    -----tf-idf similarity
    -----tfkdl similarity
    -----latent dirichlet allocation similarity
    -----neural network embedding similarity

    定义一系列函数
"""
from scipy.spatial.distance import cosine
import scipy.sparse as ssp


def cal_tfidf_sim(counterizer_obj, tfidf_obj, sent_1, sent_2):
    """
    sent_1 and sent_2 have been preprocess
    :param counterizer_obj: 
    :param tfidf_obj: 
    :param sent_1: string sentence
    :param sent_2: 
    :return: 
    """
    counter_vector1 = counterizer_obj.transform([sent_1]).toarray()
    counter_vector2 = counterizer_obj.transform([sent_2]).toarray()

    tfidf_vector1 = tfidf_obj.transform(counter_vector1).toarray()
    tfidf_vector2 = tfidf_obj.transform(counter_vector2).toarray()

    return 1 - cosine(tfidf_vector1[0], tfidf_vector2[0])


def cal_tfkdl_sim(tfkdl_object, tfkdl_weight, tfkdl_counterizer, sent_1, sent_2):
    """
    sent_1 and sent_2 have been preprocess
    :param tfkdl_object: 
    :param sent_1: 
    :param sent_2: 
    :return: 
    """
    sent1_countvector = tfkdl_counterizer.transform([sent_1])
    sent1_dense = ssp.lil_matrix(sent1_countvector).todense()
    sent1_weight = tfkdl_object.weighting_internal(sent1_dense, tfkdl_weight)

    sent2_countvector = tfkdl_counterizer.transform([sent_2])
    sent2_dense = ssp.lil_matrix(sent2_countvector).todense()
    sent2_weight = tfkdl_object.weighting_internal(sent2_dense, tfkdl_weight)

    return 1 - cosine(sent1_weight[0], sent2_weight[0])


def cal_lda_sim(lda_obj, sent_1, sent_2):
    """
    sent_1 and sent_2 have been preprocess
    :param lda_obj: 
    :param sent_1: 
    :param sent_2: 
    :return: 
    """
    pass
