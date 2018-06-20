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

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=1.0, mode="FAN_AVG", uniform=True, dtype=tf.float32
)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=2.0, mode="FAN_IN", uniform=False, dtype=tf.float32
)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def highway(
    x,
    size=None,
    activation=tf.nn.relu,
    num_layers=2,
    scope="highway",
    dropout=0.0,
    reuse=None,
):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(
                x,
                size,
                bias=True,
                activation=tf.sigmoid,
                name="gate_%d" % i,
                reuse=reuse,
            )
            H = conv(
                x,
                size,
                bias=True,
                activation=activation,
                name="activation_%d" % i,
                reuse=reuse,
            )
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


def conv(
    inputs,
    output_size,
    bias=None,
    activation=None,
    kernel_size=1,
    name="conv",
    reuse=None,
):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable(
            "kernel_",
            filter_shape,
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu() if activation is not None else initializer(),
        )
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable(
                "bias_",
                bias_shape,
                regularizer=regularizer,
                initializer=tf.zeros_initializer(),
            )
        # outputs = layer_norm(outputs)
        if activation is not None:
            return activation(outputs)
        else:
            return tf.nn.tanh(outputs)


def rnn(
    rnn_type,
    scope,
    inputs,
    length,
    hidden_size,
    layer_num=1,
    dropout_keep_prob=None,
    concat=True,
    reuse=None,
):
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
        if not rnn_type.startswith("bi"):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(
                cell, inputs, sequence_length=length, dtype=tf.float32
            )
            if rnn_type.endswith("lstm"):
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
            if rnn_type.endswith("lstm"):
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
        if rnn_type.endswith("lstm"):
            cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        elif rnn_type.endswith("gru"):
            cell = tc.rnn.GRUCell(num_units=hidden_size)
        elif rnn_type.endswith("rnn"):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError("Unsuported rnn type: {}".format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(
                cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
            )
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells

def rnn2(hidden_size, document1, document2,documen1_len, document2_len,model='mean_pool', dropout=0):
    rnn_cell_fw_one = tc.rnn.LSTMCell(hidden_size, state_is_tuple=True)
    rnn_cell_bw_one = tc.rnn.LSTMCell(hidden_size, state_is_tuple=True)
    with tf.variable_scope('encode_document'):
        (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fw_one,
            cell_bw=rnn_cell_bw_one,
            dtype="float",
            sequence_length=documen1_len,
            inputs=document1,
            scope="encoded_sentence_one")
        tf.get_variable_scope().reuse_variables()
        (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fw_one,
            cell_bw=rnn_cell_bw_one,
            dtype="float",
            sequence_length=document2_len,
            inputs=document2,
            scope="encoded_sentence_one")
        if model == "mean_pool":
            # Mean pool the forward and backward RNN outputs
            pooled_fw_output_one = mean_pool(fw_output_one,
                                             documen1_len)
            pooled_bw_output_one = mean_pool(bw_output_one,
                                             documen1_len)
            pooled_fw_output_two = mean_pool(fw_output_two,
                                             document2_len)
            pooled_bw_output_two = mean_pool(bw_output_two,
                                             document2_len)
            # Shape: (batch_size, 2*rnn_hidden_size)
            encoded_sentence_one = tf.concat([pooled_fw_output_one,
                                              pooled_bw_output_one], 1)
            encoded_sentence_two = tf.concat([pooled_fw_output_two,
                                              pooled_bw_output_two], 1)
        elif model == "last":
            # Get the last unmasked output from the RNN
            last_fw_output_one = last_relevant_output(fw_output_one,
                                                      documen1_len)
            last_bw_output_one = last_relevant_output(bw_output_one,
                                                      documen1_len)
            last_fw_output_two = last_relevant_output(fw_output_two,
                                                      document2_len)
            last_bw_output_two = last_relevant_output(bw_output_two,
                                                      document2_len)
            # Shape: (batch_size, 2*rnn_hidden_size)
            encoded_sentence_one = tf.concat([last_fw_output_one,
                                              last_bw_output_one], 1)
            encoded_sentence_two = tf.concat([last_fw_output_two,
                                              last_bw_output_two], 1)
        else:
            raise ValueError("Got an unexpected value {} for "
                             "rnn_output_mode, expected one of "
                             "[mean_pool, last]")
    return encoded_sentence_one, encoded_sentence_two

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale",
            [filters],
            regularizer=regularizer,
            initializer=tf.ones_initializer(),
        )
        bias = tf.get_variable(
            "layer_norm_bias",
            [filters],
            regularizer=regularizer,
            initializer=tf.zeros_initializer(),
        )
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input


def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant