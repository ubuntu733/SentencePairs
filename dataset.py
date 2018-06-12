# -*- coding:utf8 -*-
# ==============================================================================
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
"""
This module implements data process strategies.
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import json
import logging
import numpy as np
from collections import Counter
import jieba
import re

class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, args):
        self.logger = logging.getLogger("alibaba")
        self.args = args
        self.data_sets = self._load_dataset(args.preposs_file)
        self.train_set, self.dev_set = self._shuffle_and_split_data_set(self.data_sets)

    def _load_dataset(self, data_path):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path, 'r') as fin:
            data_set = []
            for idx, line in enumerate(fin):
                line = unicode(line, encoding='utf8')
                sample =  {}
                line_list = str(line).strip().split('|')
                if len(line_list) != 3:
                    self.logger.warning('第{}行数据格式错误'.format(idx+1))
                    continue
                else:
                    sample['document1'] = [unicode(_, 'utf8') for _ in line_list[0].split(' ')]
                    sample['document1_character'] = self._add_character(line_list[0].split(' '))
                    sample['document2'] = [unicode(_, 'utf8') for _ in line_list[1].split(' ')]
                    sample['document2_character'] = self._add_character(line_list[1].split(' '))
                    sample['label'] = self._label_2_list(int(line_list[2]))
                data_set.append(sample)
            self.logger.info('DataSet size {} sample'.format(len(data_set)))

        return data_set

    def _add_character(self, word_list):
        '''
        Add the characters
        Args:
            word_list: list of words
        Returns:
            list of characters
        '''
        character_list = []
        for word in word_list:
            character_list.append([character for character in unicode(word,'utf8')])
        return character_list

    def _shuffle_and_split_data_set(self, data_set):
        '''
        打乱并且分割数据集
        '''
        data_size = len(data_set)
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        index = int(data_size * (1 - self.args.dev))
        train_indices = indices[0:index]
        dev_indices = indices[index:-1]
        train_set = []
        dev_set = []
        for idx in train_indices:
            train_set.append(data_set[idx])
        for idx in dev_indices:
            dev_set.append(data_set[idx])
        return train_set, dev_set

    def get_mini_batchs(self, batch_size, set_name='train'):
        self.train_set, self.dev_set = self._shuffle_and_split_data_set(self.data_sets)
        if set_name == 'train':
            data_set  = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        data_size = len(data_set)
        indices = np.arange(data_size)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data_set, batch_indices)

    def _one_mini_batch(self, data, batch_indices):
        """
        Get one mini batch
        Args:
            data: all data
            batch_indices: the indices of the samples to be selected
        Returns:
            one batch of data
        """
        batch_data = {
                'raw_data': [data[i] for i in batch_indices],
                'document1_ids': [],
                'document2_ids': [],
                'document1_character_ids' : [],
                'document2_character_ids' :[],
                'label' : []
            }
        for data in batch_data['raw_data']:
            try:
                batch_data['document1_ids'].append(data['document1_ids'])
                batch_data['document2_ids'].append(data['document2_ids'])
                batch_data['document1_character_ids'].append(data['document1_character_ids'])
                batch_data['document2_character_ids'].append(data['document2_character_ids'])
                batch_data['label'].append(data['label'])
            except KeyError:
                print(' ')
        return batch_data



    def word_iter(self, set_name=None, character=False):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                if character:
                    for token in sample['document1_character']:
                        for character in token:
                            yield character
                    for token in sample['document2_character']:
                        for character in token:
                            yield character
                else:
                    for token in sample['document1']:
                        yield token
                    for token in sample['document2']:
                        yield token


    def convert_to_ids(self, vocab, character=False):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set]:
            if data_set is None:
                continue
            for sample in data_set:
                if character:
                    sample['document1_character_ids'] = vocab.convert_character_to_ids(
                        sample['document1_character'], self.args.max_document_len, self.args.max_word_len)
                    sample['document2_character_ids'] = vocab.convert_character_to_ids(
                        sample['document2_character'], self.args.max_document_len, self.args.max_word_len)
                else:
                    sample['document1_ids'] = vocab.convert_to_ids(sample['document1'], self.args.max_document_len)
                    sample['document2_ids'] = vocab.convert_to_ids(sample['document2'], self.args.max_document_len)


    def _label_2_list(self, label):
        label_list = [ 0 for _ in range(2)]
        label_list[label] = 1
        return label_list




