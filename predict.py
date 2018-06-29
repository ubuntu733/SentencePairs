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

from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import argparse
from vocab_utils import Vocab
import namespace_utils
from sklearn import metrics
import numpy as np
import re
import tensorflow as tf
import SentenceMatchTrainer
from SentenceMatchModelGraph import SentenceMatchModelGraph
from SentenceMatchDataStream import SentenceMatchDataStream
import jieba

def build_data(args):
    write_file = open('test.csv', 'w')
    jieba.load_userdict('alibaba/dict')
    with open(args.in_path,'r') as fin:
        for idx,line in enumerate(fin):
            line = line.decode('utf-8').strip()
            line = re.sub(
                u"[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+", "", line
            )
            line_list = line.split('\t')
            if len(line_list) !=3:
                print('第{}行数据格式错误'.format(idx+1))
                raise EOFError
            segment_document1 = [_ for _ in jieba.cut(line_list[1])]
            segment_document2 = [_ for _ in jieba.cut(line_list[2])]
            write_file.write(" ".join(segment_document1) + '\t')
            write_file.write(" ".join(segment_document2) + '\t')
            write_file.write(line_list[0] + '\n')

    write_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()

    # load the configuration file
    print('Loading configurations.')
    options = namespace_utils.load_namespace(args.model_prefix + ".config.json")

    if args.word_vec_path is None: args.word_vec_path = options.word_vec_path

    print('proprocess data...')
    build_data(args)
    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt')
    label_vocab = Vocab(args.model_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    if options.with_char:
        char_vocab = Vocab(args.model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    else:
        char_vocab = None

    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream('test.csv', word_vocab=word_vocab, char_vocab=char_vocab,
                                             label_vocab=label_vocab,
                                             isShuffle=False, isLoop=True, isSort=False, options=options,
                                             test=True)
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    best_path = args.model_prefix + ".best.model"
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=False, options=options)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        print("Restoring model from " + best_path)
        saver.restore(sess, best_path)
        print("DONE!")
        SentenceMatchTrainer.predict(sess, valid_graph, testDataStream,
                                                                outpath=args.out_path,
                                                                label_vocab=label_vocab)




