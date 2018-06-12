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

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import argparse
import logging
import os
from model import model
import dataset
import vocab
import jieba
import re


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Classification on VIDAA dataset')
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--char', action='store_true',
                        help='use char embedding')
    parser.add_argument(
        '--predict',
        action='store_true',
        help='predict the label for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--dev', type=float, default=0.2,
                                help='验证集比例')
    train_settings.add_argument('--filter_sizes', type=list, default=[5],
                                help='一维卷积核大小')
    train_settings.add_argument('--num_filters', type=int, default=32,
                                help='卷积核数量')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout', type=float, default=0,
                                help='dropout rate')
    train_settings.add_argument('--batch_norm', action='store_true',
                                help='whether use batch norm or not')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--hidden_size', type=int, default=128,
                                help='number of rnn hidden unit')
    train_settings.add_argument('--max_document_len', type=int, default=30,
                                help='max length of document')
    train_settings.add_argument('--max_word_len', type=int, default=5,
                                help='max length of word')
    model_settings = parser.add_argument_group('model settings')

    model_settings.add_argument('--embedding_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--character_embedding_size', type=int, default=100,
                                help='size of the character embeddings')
    model_settings.add_argument('--class_model', type=str, default='rnn')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--data_files',nargs='+',
                                default=['data/atec_nlp_sim_train.csv'],
                                help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--ltp_model', default='data/cws.model',
                               help='the file of ltp cws model')
    path_settings.add_argument('--preposs_file', default='data/train.data',
                               help='the file with ltp token segment')
    path_settings.add_argument('--model_dir', default='data/models/',
                               help='the dir to store vocab')
    path_settings.add_argument('--vocab_dir', default='data/vocab/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--dict_file', default='data/dict',
                               help='user dict of jieba')
    path_settings.add_argument(
        '--log_path',
        help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    Checks data, create vocab, load pretrained embedding or initialization randomly embedding
    """
    logger = logging.getLogger('alibaba')
    logger.info('Checking the data files... ')
    for data in args.data_files:
        assert os.path.exists(data), '{} file does not exist'.format(data)
    logger.info('preprocess raw data...')
    jieba.load_userdict(args.dict_file)
    logger.info('segment raw data')
    preposs_file = open(args.preposs_file, 'w')
    for data_file in args.data_files:
        with open(data_file,'r') as fin:
            for idx, line in enumerate(fin):
                line = unicode(line, encoding='utf8')
                line_re = re.sub(u'[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "",line)
                line_list = str(line_re).strip('\n').split('\t')
                if len(line_list) != 4:
                    logger.warning('{} - {} from is wrong'.format(args.data_files, idx+1))
                    continue
                document1 = line_list[1].strip().replace(' ','')
                document2 = line_list[2].strip().replace(' ','')
                segment_document1 = [ _ for _ in jieba.cut(document1)]
                segment_document2 = [ _ for _ in jieba.cut(document2)]
                preposs_file.write(" ".join(segment_document1))
                preposs_file.write("|")
                preposs_file.write(" ".join(segment_document2))
                preposs_file.write('|')
                preposs_file.write(line_list[3] + '\n')
    preposs_file.close()
    logger.info('Building vocabulary...')
    for dir_path in [
            args.vocab_dir,
            args.model_dir,
            args.result_dir,
            args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    data = dataset.Dataset(args)
    word_vocab_ = vocab.Vocab()
    for token in data.word_iter():
        word_vocab_.add(token)
    unfiltered_vocab_size = word_vocab_.size()
    word_vocab_.filter_word_by_count(min_count=2)
    filtered_num = unfiltered_vocab_size - word_vocab_.size()
    logger.info(
        'After filter {} tokens, the final word vocab size is {}'.format(
            filtered_num,
            word_vocab_.size()))

    logger.info('Assigning word embeddings...')
    word_vocab_.random_init_embeddings(args.embedding_size)


    character_vocab_ = vocab.Vocab()
    for character in data.word_iter('train', character=True):
        character_vocab_.add(character)
    unfiltered_vocab_size = character_vocab_.size()
    character_vocab_.filter_word_by_count(min_count=2)
    filtered_num = unfiltered_vocab_size - character_vocab_.size()
    logger.info('After filter {} characters, the final character vocab size is {}'.format(
                filtered_num, character_vocab_.size()))
    logger.info('Assigning character embeddings...')
    character_vocab_.random_init_embeddings(args.character_embedding_size)


    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir,'vocab.data'), 'wb') as fout:
        pickle.dump(word_vocab_, fout)

    logger.info('Saving character vocab...')
    with open(os.path.join(args.vocab_dir,'vocab_character.data'), 'wb') as fout:
            pickle.dump(character_vocab_, fout)
    logger.info('Done with preparing!')


def train(args):
    """
    Training the classification model
    """
    logger = logging.getLogger('alibaba')
    logger.info('Load data_set , vocab and label config...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        word_vocab_ = pickle.load(fin)
    with open(os.path.join(args.vocab_dir,'vocab_character.data'),'rb') as fin:
        vocab_character_ = pickle.load(fin)
    data = dataset.Dataset(args)
    logger.info('Convert word to id...')
    data.convert_to_ids(word_vocab_)
    logger.info('Convert character to id...')
    data.convert_to_ids(vocab_character_, character=True)
    logger.info('Build Model...')
    model_ = model.Model(args, word_vocab=word_vocab_, character_vocab=vocab_character_)
    logger.info('Training the model...')
    model_.train(data, args.epochs, args.batch_size, save_dir=args.model_dir,
                save_prefix='model')
    logger.info('Done with training...')


def evaluate(args):
    """
    Evaluate the classification model
    """
    logger = logging.getLogger('alibaba')
    logger.info('Load data_set , vocab and label config...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        word_vocab_ = pickle.load(fin)
    with open(os.path.join(args.vocab_dir, 'vocab_character.data'), 'rb') as fin:
        vocab_character_ = pickle.load(fin)
    data = dataset.Dataset(args)
    logger.info('Convert word to id...')
    data.convert_to_ids(word_vocab_)
    logger.info('Convert character to id...')
    data.convert_to_ids(vocab_character_, character=True)
    logger.info('Build Model...')
    model_ = model.Model(args, word_vocab=word_vocab_, character_vocab=vocab_character_)
    model_.restore(model_dir=args.model_dir, model_prefix=args.class_model)
    logger.info('Evaluating the model on dev set...')
    dev_batchs = data.get_mini_batchs(batch_size=args.batch_size, set_name='dev')
    loss_, accuracy, _ = model_.evaluate(batch_data=dev_batchs,
                                            result_dir=args.result_dir,
                                            result_prefix='dev.evaluate')
    logger.info('Loss on dev set: {}'.format(loss_))
    logger.info('Accuracy on dev set: {}'.format(accuracy))
    logger.info(
        'Predicted labels are saved to {}'.format(
            os.path.join(
                args.result_dir)))




def run():
    """
    Prepare and runs the whole system
    """
    args = parse_args()

    logger = logging.getLogger('alibaba')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path, encoding='UTF-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info('Running with args : {}'.format(args))

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)

if __name__ == "__main__":
    run()