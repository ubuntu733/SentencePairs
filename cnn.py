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

import tensorflow as tf
import tensorflow.contrib as tc
from layer import conv


class CNN(object):

    def __init__(self, doc1, doc2, args, dropout):
        self.doc1 = doc1
        self.doc2 = doc2
        self.args = args
        self.dropout = dropout

    def build_graph(self):
        with tf.variable_scope("encode_conv"):

            with tf.name_scope("conv" ):
                document1_conv = tf.layers.conv1d(
                    inputs=self.doc1,
                    filters=self.args.num_filters,
                    padding='same',
                    kernel_size=self.args.filter_size
                )

                document2_conv = tf.layers.conv1d(
                    inputs=self.doc2,
                    filters=self.args.num_filters,
                    padding='same',
                    kernel_size=self.args.filter_size
                )
                document1_pool = tf.layers.max_pooling1d(
                    inputs=document1_conv,
                    pool_size=2,
                    strides=1
                )

                document2_pool = tf.layers.max_pooling1d(
                    inputs=document2_conv,
                    pool_size=2,
                    strides=1
                )
                self.document1_represent = tf.layers.flatten(document1_pool)
                self.document2_represent = tf.layers.flatten(document2_pool)
                '''
                    document1_conv = conv(
                        inputs=self.doc1,
                        output_size=self.args.hidden_size,
                        activation=tf.nn.relu,
                        name="conv-%s" % filter_size,
                        kernel_size=filter_size,
                        reuse=None,
                    )
                    document2_conv = conv(
                        inputs=self.doc2,
                        output_size=self.args.hidden_size,
                        activation=tf.nn.relu,
                        name="conv-%s" % filter_size,
                        kernel_size=filter_size,
                        reuse=True,
                    )

                    document1_conv = tf.reshape(
                        document1_conv,
                        [
                            -1,
                            (self.args.max_document_len - filter_size + 1)
                            * self.args.hidden_size,
                        ],
                    )
                    document2_conv = tf.reshape(
                        document2_conv,
                        [
                            -1,
                            (self.args.max_document_len - filter_size + 1)
                            * self.args.hidden_size,
                        ],
                    )
                    document1_pooled_outputs.append(document1_conv)
                    document2_pooled_outputs.append(document2_conv)
            self.document1_pool = tf.concat(document1_pooled_outputs, 1)
            # self.document1_pool_flat = tf.reshape(self.document1_pool, [-1, num_filters_total])
            self.document2_pool = tf.concat(document2_pooled_outputs, 1)
            # self.document2_pool_flat = tf.reshape(self.document2_pool, [-1, num_filters_total])
            if self.args.dropout > 0:
                with tf.name_scope("encode_dropout"):
                    self.document1_drop = tf.nn.dropout(
                        self.document1_pool, 1 - self.args.dropout
                    )
                    self.document2_drop = tf.nn.dropout(
                        self.document2_pool, 1 - self.args.dropout
                    )
            else:
                self.document1_drop = self.document1_pool
                self.document2_drop = self.document2_pool
            self.document1_represent = tc.layers.fully_connected(
                inputs=self.document1_drop,
                num_outputs=self.args.hidden_size,
                activation_fn=tf.nn.tanh,
            )
            self.document2_represent = tc.layers.fully_connected(
                inputs=self.document2_drop,
                num_outputs=self.args.hidden_size,
                activation_fn=tf.nn.tanh,
            )
            '''
        return self.document1_represent, self.document2_represent
