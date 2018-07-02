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
# WITHOUT WARRANTIES OR CPretrainedVocabONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from collections import OrderedDict
import numpy as np
import random


class PretrainedVocab(object):
    """
    加载预训练词向量
    """

    def __init__(self, args):
        self.logger = logging.getLogger("VIDAA")
        self.args = args
        self.id2token = OrderedDict()
        self.token2id = OrderedDict()
        self.embedding = None
        self.unk_token = "<unk>"
        self.pad_token = "<blank>"
        self._load_file(args.pretrained_file)

    def _load_file(self, pretrained_file):
        with open(pretrained_file, "r") as fin:
            for idx, line in enumerate(fin):
                line = unicode(line)
                if idx == 0:
                    self.size_ = int(str(line).strip("\n").split(" ")[0])
                    self.embedding_dim = int(str(line).strip("\n").split(" ")[1])
                    self.size_ += 2
                    self.embedding = np.random.rand(self.size_, self.embedding_dim)
                    assert self.embedding_dim == self.args.embedding_size
                    self._add(self.unk_token)
                    self._add(self.pad_token)
                else:
                    tmp = str(line).strip("\n").split(" ")
                    self._add(unicode(tmp[0]), values=tmp[1:-1])
        self.embedding = np.array(self.embedding)

    def _get_id(self, token):
        """
        Gets the id of token, return the id os unk token if the token is not found
        Args:
            token: a string indicating the token
        return:
            An integer
        """
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def _add(self, token, values=None):
        """
        Add the token to vocab
        Args:
            token: a string indicating a token
            values: Correspond to a embedding
        """
        size = len(self.id2token)
        self.id2token[size] = token
        self.token2id[token] = size
        if values is not None:
            self.embedding[size] = np.array(values)
        else:
            self.embedding[size] = np.zeros(self.embedding_dim)

    def convert_to_ids(self, tokens, max_length=None):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self._get_id(label) for label in tokens]
        if max_length:
            if max_length < len(tokens):
                vec = vec[0:max_length]
            else:
                for i in range(max_length - len(tokens)):
                    vec += [self.token2id[self.pad_token]]
        return vec

    def size(self):
        return self.size_
