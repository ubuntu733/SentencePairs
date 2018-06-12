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


from collections import OrderedDict
import numpy as np


class Vocab(object):
    """
    Implements a vocabulary to story the tokens in the data with their corresponding embeddings
    """

    def __init__(self, filename=None, initial_tokens=None):
        self.id2token = OrderedDict()
        self.token2id = OrderedDict()
        self.token_count = OrderedDict()

        self.embedding_dim = None
        self.embedding = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])

        for token in self.initial_tokens:
            self.add(token)

        self.fasttext_model = None
        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        Get the size of vocab
        """
        return len(self.id2token)

    def add(self, token, count=1):
        """
        Add the token to vocab
        Args:
            token: a string
            count: a num indicating the count of the token to add, default is 1
        """
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.token2id)
            self.id2token[idx] = token
            self.token2id[token] = idx

        if count > 0:
            if token in self.token_count:
                self.token_count[token] += count
            else:
                self.token_count[token] = count
        return idx

    def get_id(self, token):
        """
        Gets the id of a token, return the id of unk token if the token is not found
        Args:
            token: a string indicating the token
        return:
            An integer
        """
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, id):
        """
        Gets the token corresponding to id, return the unk token if the id is not found
        Args:
            id: an integer
        return: a string indicating the token
        """
        try:
            return self.id2token[id]
        except KeyError:
            return self.unk_token

    def load_from_file(self, file_path):
        """
        Loads the vocab
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            line = unicode(line, encoding='utf8')
            token = line.rstrip('\n')
            self.add(token)

    def load_pretrained_from_file(self, file_path, embedding_size):
        """
        Loads the pretrained embedding file
        Args:
            file_path: a pretrained file with a word with embedding each line
        """
        self.embedding_size = embedding_size
        vector = []
        words = []
        for line in open(file_path, 'r'):
            line = unicode(line, encoding='utf8')
            line_list  = line.strip('\n').split( )
            self.add(line_list[0])
            words.append(line_list[0])
            vector.append(line_list[1:-1])
        self.embedding = np.zeros(self.size(), embedding_size)
        for index,word in enumerate(words):
            self.embedding[self.get_id(word)] = np.array(vector[index])


    def filter_word_by_count(self, min_count):
        """
        Filter the tokens in vocab if there count is less than min_count
        Args:
            min_count: an integer
        """
        filter_token = [
            token for token in self.token2id if self.token_count[token] >= min_count]
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, 0)
        for token in filter_token:
            self.add(token, 0)

    def random_init_embeddings(self, embed_dim):
        """
        Randomly initializes the embedding for each token
        Args:
            embed_dim: embedding vector size
        """
        self.embedding_dim = embed_dim
        self.embedding = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embedding[self.get_id(token)] = np.zeros([embed_dim])

    def convert_to_ids(self, tokens, max_length=None):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        if max_length:
            if max_length < len(tokens):
                vec = vec[0:max_length]
            else:
                for i in range(max_length-len(tokens)):
                    vec += [self.token2id[self.pad_token]]
        return vec

    def convert_character_to_ids(self, tokens, max_length=None, max_word_length=None):
        words = []
        for token in tokens:
            word = [self.get_id(character) for character in token]
            words.append(word)
        if max_length:
            if max_length < len(words):
                words = words[0:max_length]
            else:
                for i in range(max_length-len(words)):
                    words.append([self.get_id(self.pad_token) for i in range(max_word_length)])
        if max_word_length:
            for idx,word in enumerate(words):
                if max_word_length < len(word):
                    words[idx] = word[0:max_word_length]
                else:
                    for i in range(max_word_length-len(word)):
                        word += [self.token2id[self.pad_token]]
        return words

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens

