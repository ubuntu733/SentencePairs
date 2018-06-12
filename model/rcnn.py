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
from .layer import rnn, conv
import tensorflow as tf
import tensorflow.contrib as tc

class RCNN(object):

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
        with tf.variable_scope('encode_conv'):
            document1_pooled_outputs = []
            document2_pooled_outputs = []
            '''
            self.document1_embedded_chars_expanded = tf.expand_dims(
                self.document1_encode, -1)
            self.document2_embedded_chars_expanded = tf.expand_dims(
                self.document2_encode, -1)
            '''
            for i, filter_size in enumerate(self.args.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    document1_conv = conv(
                        inputs=self.document1_encode,
                        output_size=self.args.hidden_size,
                        activation=tf.nn.relu,
                        name='conv-%s' % filter_size,
                        kernel_size=filter_size,
                        reuse=None
                    )
                    document2_conv = conv(
                        inputs=self.document2_encode,
                        output_size=self.args.hidden_size,
                        activation=tf.nn.relu,
                        name='conv-%s' % filter_size,
                        kernel_size=filter_size,
                        reuse=True
                    )

                    '''
                    filter_shape = [
                        filter_size,
                        self.args.hidden_size * 2,
                        1,
                        self.args.num_filters]
                    W1 = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b1 = tf.Variable(tf.constant(
                        0.1, shape=[self.args.num_filters]), name="b")
                    document1_conv = tf.nn.conv2d(
                        self.document1_embedded_chars_expanded,
                        W1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    document2_conv = tf.nn.conv2d(
                        self.document2_embedded_chars_expanded,
                        W1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    document1_h = tf.nn.relu(tf.nn.bias_add(document1_conv, b1), name="relu")
                    document2_h = tf.nn.relu(tf.nn.bias_add(document2_conv, b1), name="relu")

                    # Maxpooling over the outputs
                    document1_pooled = tf.nn.max_pool(
                        document1_conv,
                        ksize=[1, self.args.max_document_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    document2_pooled = tf.nn.max_pool(
                        document2_conv,
                        ksize=[1, self.args.max_document_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    '''
                    document1_conv = tf.reshape(document1_conv, [-1, (
                                self.args.max_document_len - filter_size + 1) * self.args.hidden_size])
                    document2_conv = tf.reshape(document2_conv, [-1, (
                            self.args.max_document_len - filter_size + 1) * self.args.hidden_size])
                    document1_pooled_outputs.append(document1_conv)
                    document2_pooled_outputs.append(document2_conv)
            self.document1_pool = tf.concat(document1_pooled_outputs, 1)
            # self.document1_pool_flat = tf.reshape(self.document1_pool, [-1, num_filters_total])
            self.document2_pool = tf.concat(document2_pooled_outputs, 1)
            # self.document2_pool_flat = tf.reshape(self.document2_pool, [-1, num_filters_total])
            if self.args.dropout > 0:
                with tf.name_scope("encode_dropout"):
                    self.document1_drop = tf.nn.dropout(
                        self.document1_pool, 1 - self.args.dropout)
                    self.document2_drop = tf.nn.dropout(
                        self.document2_pool, 1 - self.args.dropout)
            else:
                self.document1_drop = self.document1_pool
                self.document2_drop = self.document2_pool
            self.document1_represent = tc.layers.fully_connected(
                inputs=self.document1_drop, num_outputs=self.args.hidden_size, activation_fn=tf.nn.tanh)
            self.document2_represent = tc.layers.fully_connected(
                inputs=self.document2_drop, num_outputs=self.args.hidden_size, activation_fn=tf.nn.tanh)
        return self.document1_represent, self.document2_represent
