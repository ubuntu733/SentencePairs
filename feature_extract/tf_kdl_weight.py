## weight.py
## Author: Yangfeng Ji
## Date: 09-06-2014
## Time-stamp: <yangfeng 09/08/2014 20:08:44>

from sklearn.feature_extraction.text import CountVectorizer
from pickle import dump
import numpy
import scipy.sparse as ssp
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

#from feature_extract.preprocess import my_tokenizer

def my_tokenizer(x):
    return x.split()


class TFKLD(object):
    def __init__(self, ftrain):
        self.ftrain = ftrain
        self.trnM, self.trnL = None, None

        self.weight = None
        self.countizer = None
        self.tfkdl_train = None

    def loadtext(self, fname):
        text, label = [], []
        with open(fname, 'rb') as fin:
            count = 0
            for line in fin:
                line = line.decode("utf-8")
                items = line.strip().split("\t")
                label.append(int(items[0]))
                text.append(items[1])
                text.append(items[2])

                count += 1
                #if count % 100 == 0:
                #    break

        return text, label


    def createdata(self):

        trnT, trnL = self.loadtext(self.ftrain)

        # Change the parameter setting in future
        # self.countizer = CountVectorizer(tokenizer=self.my_tokenizer, dtype=numpy.float,
        #                             ngram_range=(1, 2))

        self.countizer = CountVectorizer(tokenizer=my_tokenizer, dtype=numpy.float)

        trnM = self.countizer.fit_transform(trnT)
        self.trnM, self.trnL = trnM, trnL

        self.trnM = ssp.lil_matrix(self.trnM)

    def weighting(self):
        print('Create data matrix ...')
        self.createdata()
        print ('Counting features ...')
        M = self.trnM.todense()
        print ('type(M) = {}'.format(type(M)))
        L = self.trnL
        nRow, nDim = M.shape
        print('nRow, nDim = {}, {}'.format(nRow, nDim))
        # (0, F), (0, T), (1, F), (1, T)
        count = numpy.ones((4, nDim))
        for n in range(0, nRow, 2):
            if n % 1000 == 0:
                print('Process {} rows'.format(n))
            for d in range(nDim):
                label = L[n // 2]
                if ((M[n, d] > 0) and (M[n + 1, d] == 0)) or ((M[n, d] == 0) and (M[n + 1, d] > 0)):
                    # Non-shared
                    if label == 0:
                        # (0, F)
                        count[0, d] += 1.0
                    elif label == 1:
                        # (1, F)
                        count[2, d] += 1.0
                elif (M[n, d] > 0) and (M[n + 1, d] > 0):
                    # Shared
                    if label == 0:
                        # (0, T)
                        count[1, d] += 1.0
                    elif label == 1:
                        # (1, T)
                        count[3, d] += 1.0
        # Compute KLD
        print ('Compute KLD weights ...')
        self.computeKLD(count)

        # Apply weights
        print ('Weighting ...')
        self.tfkdl_train = self.weighting_internal(M, self.weight)

    def computeKLD(self, count):
        # Smoothing
        count = count + 0.05
        # Normalize
        pattern = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
        pattern = numpy.array(pattern)
        prob = count / (pattern.dot(count))
        #
        ratio = numpy.log((prob[0:2, :] / prob[2:4, :]) + 1e-7)
        self.weight = (ratio * prob[0:2, :]).sum(axis=0)
        print (self.weight.shape)

    def weighting_internal(self, datasetM, weight):
        weight = ssp.lil_matrix(weight).toarray()
        # print ('Applying weighting to training examples')
        for n in range(datasetM.shape[0]):
            # if n % 1000 == 0:
            #     print ('Process {} rows'.format(n))
            datasetM[n, :] = numpy.multiply(numpy.array(datasetM[n, :]),  numpy.array(weight))

        return datasetM

    def save(self, fname):
        D = {"weight": self.weight,
             "countvector_model": self.countizer}

        with open(fname, 'wb') as fout:
            dump(D, fout, 2)
        print ('Done')


def main(dataset_path, save_path):
    tfkld = TFKLD(dataset_path)

    #text, label = tfkld.loadtext(dataset_path)
    #print("=====", len(text))

    tfkld.weighting()

    tfkld.save(save_path)


if __name__ == "__main__":
    main("../data/ori_data/train_process.csv", "../data/m_result/tfkdl_params_train.pickle")
    #main("../data/ori_data/complete_process.csv", "../data/m_result/tfkdl_params_complete.pickle")


