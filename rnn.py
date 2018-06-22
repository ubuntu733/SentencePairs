# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2018 Hisense, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from layer import *
import tensorflow as tf


class RNN(object):

    def __init__(self, doc1, doc2, doc1_len, doc2_len, args, dropout):
        self.doc1 = doc1
        self.doc2 = doc2
        self.args = args
        self.dropout = dropout
        self.doc1_len = doc1_len
        self.doc2_len = doc2_len

    def build_graph(self):
        '''
        self.document1_encode, self.document2_encode = rnn2(
            hidden_size=self.args.hidden_size,
            document1=self.doc1,
            document2=self.doc2,
            documen1_len= self.doc1_len,
            document2_len= self.doc2_len
        )
        '''
        self.out1 = stackedRNN(self.doc1, 1-self.dropout, "side1", self.args.hidden_size)
        self.out2 = stackedRNN(self.doc2, 1-self.dropout, "side2", self.args.hidden_size)
        '''
        self.document1_encode, _ = rnn(
            rnn_type="bi-lstm",
            scope="document1_encode",
            inputs=self.doc1,
            length=None,
            hidden_size=self.args.hidden_size,
            dropout_keep_prob=1-self.dropout
        )
        self.document2_encode, _ = rnn(
            rnn_type="bi-lstm",
            inputs=self.doc2,
            scope="document2_encode",
            length=None,
            hidden_size=self.args.hidden_size,
            dropout_keep_prob=1-self.dropout
        )


        self.doc1 = tf.reduce_max(self.document1_encode,axis=1)
        self.doc2 = tf.reduce_max(self.document2_encode,axis=1)
        '''
        return self.out1, self.out2
