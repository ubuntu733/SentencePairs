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

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
def highway(x, size = None, activation = tf.nn.relu,
            num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d"%i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d"%i, reuse = reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x



def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        #outputs = layer_norm(outputs)
        if activation is not None:
            return activation(outputs)
        else:
            return tf.nn.tanh(outputs)



def rnn(rnn_type, scope,inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True, reuse=None):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
    Returns:
        RNN outputs and final state
    """
    with tf.variable_scope(scope, reuse=reuse):
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c = [state.c for state in states]
                h = [state.h for state in states]
                states = h
        else:
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
            )
            states_fw, states_bw = states
            if rnn_type.endswith('lstm'):
                c_fw = [state_fw.c for state_fw in states_fw]
                h_fw = [state_fw.h for state_fw in states_fw]
                c_bw = [state_bw.c for state_bw in states_bw]
                h_bw = [state_bw.h for state_bw in states_bw]
                states_fw, states_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                states = tf.concat([states_fw, states_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                states = states_fw + states_bw
        return outputs, states


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    cells = []
    for i in range(layer_num):
        if rnn_type.endswith('lstm'):
            cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        elif rnn_type.endswith('gru'):
            cell = tc.rnn.GRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell,
                                         input_keep_prob=dropout_keep_prob,
                                         output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias
