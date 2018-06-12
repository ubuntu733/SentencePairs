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

from .layer import rnn
import tensorflow as tf
class RNN(object):

    def __init__(self, doc1, doc2, args):
        self.doc1 = doc1
        self.doc2 = doc2
        self.args = args

    def build_graph(self):
        self.document1_encode, _ = rnn(
            rnn_type='bi-lstm',
            scope='document_encode',
            inputs=self.doc1,
            length=None,
            hidden_size=self.args.hidden_size,
            reuse=None)
        self.document2_encode, _ = rnn(
            rnn_type='bi-lstm',
            inputs=self.doc2,
            scope='document_encode',
            length=None,
            hidden_size=self.args.hidden_size,
            reuse=True)
        self.doc1 = tf.reduce_max(self.document1_encode, axis=1)
        self.doc2 = tf.reduce_max(self.document2_encode, axis=1)
        return self.doc1, self.doc2