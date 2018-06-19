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
import numpy as np
import logging
import time
from layer import *
import os
import json
from sklearn import metrics
from rcnn import RCNN
from rnn import RNN
from cnn import CNN


class Model(object):

    def __init__(self, args, word_vocab, character_vocab=None):
        self.logger = logging.getLogger("alibaba")
        self.vocab = word_vocab
        self.character_vocab = character_vocab
        self.args = args

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def _build_graph(self):
        start_time = time.time()
        self._build_setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._compute_loss()
        self._create_train_op()
        self.logger.info("Time to build graph: {} s".format(time.time() - start_time))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info("There are {} parameters in the model".format(param_num))

    def _build_setup_placeholders(self):
        self.document1 = tf.placeholder(tf.int32, [None, self.args.max_document_len])
        self.document2 = tf.placeholder(tf.int32, [None, self.args.max_document_len])
        self.document1_character = tf.placeholder(
            tf.int32, [None, self.args.max_document_len, self.args.max_word_len]
        )
        self.document2_character = tf.placeholder(
            tf.int32, [None, self.args.max_document_len, self.args.max_word_len]
        )
        self.label = tf.placeholder(tf.float32, [None])
        self.dropout = tf.placeholder(tf.float32, name='dropout')

    def _embed(self):
        with tf.variable_scope("embedding"):
            with tf.device("cpu:0"):

                self.word_embeddings = tf.Variable(
                    tf.random_uniform(
                        [self.vocab.size(), self.args.embedding_size], -1.0, 1.0
                    ),
                    name="word_embeddings",
                    trainable=True,
                )

                self.character_embeddings = tf.Variable(
                    tf.random_uniform(
                        [
                            self.character_vocab.size(),
                            self.args.character_embedding_size,
                        ]
                    ),
                    name="character_embeddings",
                )
                '''

                self.word_embeddings = tf.get_variable(
                    'word_embeddings',
                    shape=(self.vocab.size(), self.args.embedding_size),
                    initializer=tf.constant_initializer(self.vocab.embedding),
                    trainable=True
                    )
                if self.args.char:
                    self.character_embeddigns = tf.get_variable(
                        'character_embeddings',
                        shape=(self.character_vocab.size(), self.args.character_embedding_size),
                        initializer=tf.constant_initializer(self.character_vocab.embedding),
                        trainable=True
                    )
                '''
            self.document1_emb = tf.nn.embedding_lookup(
                self.word_embeddings, self.document1
            )
            self.document2_emb = tf.nn.embedding_lookup(
                self.word_embeddings, self.document2
            )
            if self.args.char:
                self.document1_character_emb = tf.nn.embedding_lookup(
                    self.character_embeddings, self.document1_character
                )
                self.document2_character_emb = tf.nn.embedding_lookup(
                    self.character_embeddings, self.document2_character
                )
                self.document1_character_emb = tf.reshape(
                    self.document1_character_emb,
                    [-1, self.args.max_word_len, self.args.character_embedding_size],
                )
                self.document2_character_emb = tf.reshape(
                    self.document2_character_emb,
                    [-1, self.args.max_word_len, self.args.character_embedding_size],
                )

                document1_character_conv = conv(
                    self.document1_character_emb,
                    self.args.hidden_size,
                    bias=True,
                    activation=tf.nn.relu,
                    kernel_size=5,
                    name="char_conv",
                    reuse=None,
                )
                document1_character_conv = tf.reduce_max(
                    document1_character_conv, axis=1
                )
                document1_character_conv = tf.reshape(
                    document1_character_conv,
                    [-1, self.args.max_document_len, self.args.hidden_size],
                )

                document2_character_conv = conv(
                    self.document2_character_emb,
                    self.args.hidden_size,
                    bias=True,
                    activation=tf.nn.relu,
                    kernel_size=5,
                    name="char_conv",
                    reuse=True,
                )
                document2_character_conv = tf.reduce_max(
                    document2_character_conv, axis=1
                )
                document2_character_conv = tf.reshape(
                    document2_character_conv,
                    [-1, self.args.max_document_len, self.args.hidden_size],
                )
                self.doc1 = highway(
                    tf.concat([self.document1_emb, document1_character_conv], axis=2),
                    size=self.args.hidden_size,
                    scope="highway",
                    dropout=self.args.dropout,
                )
                self.doc2 = highway(
                    tf.concat([self.document2_emb, document2_character_conv], axis=2),
                    size=self.args.hidden_size,
                    scope="highway",
                    dropout=self.args.dropout,
                    reuse=True,
                )
            else:
                self.doc1 = self.document1_emb
                self.doc2 = self.document2_emb

    def _encode(self):
        if self.args.class_model == "rcnn":
            self.model = RCNN(doc1=self.doc1, doc2=self.doc2, args=self.args)
        elif self.args.class_model == "rnn":
            self.model = RNN(doc1=self.doc1, doc2=self.doc2, args=self.args, dropout=self.dropout)
        elif self.args.class_model == "cnn":
            self.model = CNN(doc1=self.doc1, doc2=self.doc2, args=self.args, dropout=self.dropout)
        else:
            raise NotImplementedError(
                "Do not implement {} model".format(self.args.class_model)
            )
        self.document1_represent, self.document2_represent = self.model.build_graph()

    def _match(self):
        with tf.variable_scope("match"):
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.document1_represent, self.document2_represent)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.document1_represent), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.document2_represent), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            '''
            self.vector = tf.concat(
                [self.document1_represent, self.document2_represent], 1
            )
            self.vector = tf.nn.dropout(self.vector,1-self.dropout)
            self.score = tc.layers.fully_connected(
                self.vector, num_outputs=2, activation_fn=tf.nn.tanh
            )
            #self.score = tf.nn.softmax(self.score)
            '''
        """
            document1_len = tf.sqrt(tf.reduce_sum(tf.multiply(self.document1_represent, self.document1_represent), 1))
            document2_len = tf.sqrt(tf.reduce_sum(tf.multiply(self.document2_represent, self.document2_represent), 1))
            mul = tf.reduce_sum(tf.multiply(self.document1_represent, self.document2_represent), 1)
            tf.reduce_sum(tf.multiply(self.document1_represent, self.document2_represent), 1)
            self.score = tf.div(mul, tf.multiply(document1_len,document2_len), name="score1")
        """
        with tf.variable_scope("predict"):
            self.predict = tf.rint(self.distance)
            '''
            self.predict = tf.argmax(self.score, axis=1)
            '''
        with tf.variable_scope("accuracy"):

            '''
            correct_predictions = tf.equal(self.predict, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy"
            )
            '''
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _compute_loss(self):

        with tf.variable_scope("loss"):
            '''
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.score, labels=self.label
                )
            )
            '''
            self.loss = self.contrastive_loss(self.label, self.distance, self.args.batch_size)
            self.all_params = tf.trainable_variables()
            if self.args.weight_decay > 0:
                with tf.variable_scope("l2_loss"):
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                self.loss += self.args.weight_decay * l2_loss

    def _create_train_op(self):
        if self.args.optim == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(self.args.learning_rate)
        elif self.args.optim == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        elif self.args.optim == "rprop":
            self.optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate)
        elif self.args.optim == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.args.learning_rate)
        else:
            raise NotImplementedError(
                "Unsupported optimizer: {}".format(self.args.optim_type)
            )
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batch):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
        """
        total_num, total_loss = 0, 0.0
        index = 0
        predicts = []
        labels = []
        for idx, batch in enumerate(train_batch, 1):

            feed_dict = {
                self.document1: batch["document1_ids"],
                self.document2: batch["document2_ids"],
                self.document1_character: batch["document1_character_ids"],
                self.document2_character: batch["document2_character_ids"],
                self.label: batch["label"],
                self.dropout:self.args.dropout
            }
            _, loss, accuracy, predict, doc1, doc2= self.sess.run(
                [
                    self.train_op,
                    self.loss,
                    self.accuracy,
                    self.predict,
                    self.document1_represent,
                    self.document2_represent
                ],
                feed_dict,
            )
            doc1_list = doc1.tolist()
            doc2_list = doc2.tolist()
            predict_list = predict.tolist()
            '''
            _, loss, accuracy, score, vector, predict, doc1, doc2, a, b = self.sess.run(
                [
                    self.train_op,
                    self.loss,
                    self.accuracy,
                    self.score,
                    self.vector,
                    self.predict,
                    self.document1_represent,
                    self.document2_represent,
                    self.document1_emb,
                    self.document2_emb,
                ],
                feed_dict,
            )
            score_list = score.tolist()
            vector_list = vector.tolist()
            predict_list = predict.tolist()
            doc1_list = doc1.tolist()
            doc2_list = doc2.tolist()
            a_list = a.tolist()
            b_list = b.tolist()
            '''
            total_loss += loss
            predicts.extend(predict.tolist())
            labels.extend(batch["label"])
            # total_accuracy += accuracy * len(batch['raw_data'])
            # total_auc += auc * len(batch['raw_data'])
            index += 1
            self.logger.info(
                "batch {}, loss {}, accuracy {}".format(idx, loss, accuracy)
            )

        return 1.0 * total_loss / float(index), predicts, labels

    def train(
        self,
        data,
        epochs,
        batch_size,
        save_dir,
        save_prefix,
        evaluate=True,
        character=False,
    ):
        """
        Training the model with data.
        Args:
              data: The VIDAA Classification Data
              epochs: The number of training epochs
              batch_size:
              save_dir: The director to save model
              save_prefix: The prefix indicating the model type
              evaluate: Whether to evaluate the model on test set after each epoch
              character: Whether nor not to use character feature
        """
        max_f1score = 0.0
        for epoch in range(1, epochs + 1):
            self.logger.info("Training the model for epoch {}".format(epoch))

            train_batchs = data.get_mini_batchs(
                batch_size=batch_size, set_name="train", shuffle=True
            )
            train_loss, predicts, labels = self._train_epoch(train_batchs)
            self.logger.info(
                "Average train loss for epoch {} is {}".format(epoch, train_loss)
            )
            self.logger.info(
                "classification_report: \n {}".format(
                    metrics.classification_report(
                        labels, np.array(predicts)
                    )
                )
            )
            self.logger.info(
                "混淆矩阵为: \n {}".format(
                    metrics.confusion_matrix(
                        labels, np.array(predicts)
                    )
                )
            )
            self.logger.info(
                "completeness_score: {}".format(
                    metrics.completeness_score(
                        labels, np.array(predicts)
                    )
                )
            )
            if evaluate:
                if data.dev_set is not None:
                    dev_batches = data.get_mini_batchs(
                        batch_size=batch_size, set_name="dev"
                    )
                    loss_, accuracy_, predicts, labels = self.evaluate(dev_batches)
                    f1score = metrics.f1_score(labels, np.array(predicts))
                    self.logger.info("Dev eval loss {}".format(loss_))
                    self.logger.info("Dev eval accuracy {}".format(accuracy_))
                    self.logger.info(
                        "classification_report: \n {}".format(
                            metrics.classification_report(
                                labels, np.array(predicts)
                            )
                        )
                    )
                    self.logger.info(
                        "混淆矩阵为: \n {}".format(
                            metrics.confusion_matrix(
                                labels, np.array(predicts)
                            )
                        )
                    )
                    self.logger.info(
                        "completeness_score: {}".format(
                            metrics.completeness_score(
                                labels, np.array(predicts)
                            )
                        )
                    )
                    if f1score >= max_f1score:
                        max_f1score = f1score
                        self.save(save_dir, save_prefix)
            else:
                self.save(save_dir, save_prefix)

    def evaluate(
        self, batch_data, result_dir=None, result_prefix=None, save_predict_label=False
    ):
        """
        Evaluate the model with data
        Args:
            batch_data: iterable batch data
            result_dir: the director to save the predict answers ,
                        answers will not save if None
            result_prefix: the prefix of file for saving the predict answers,
                           answers will not save if None
            save_predict_label: if True, the pred_answers will be added to raw sample and saved
            character: use character feature
        """
        if save_predict_label:
            result = []
        total_loss, total_num, total_accuracy = 0.0, 0, 0.0
        index = 0
        labels = []
        predicts = []
        for idx, batch in enumerate(batch_data):
            feed_dict = {
                self.document1: batch["document1_ids"],
                self.document2: batch["document2_ids"],
                self.document1_character: batch["document1_character_ids"],
                self.document2_character: batch["document2_character_ids"],
                self.label: batch["label"],
                self.dropout : 0
            }

            loss, accuracy, predict = self.sess.run(
                [self.loss, self.accuracy, self.predict], feed_dict
            )
            index += 1
            total_loss += loss * len(batch["raw_data"])
            total_accuracy += accuracy
            predicts.extend(predict.tolist())
            labels.extend(batch["label"])
            """
            total_auc += auc * len(batch['raw_data'])
            total_accuracy += accuracy * len(batch['raw_data'])
            """
            total_num += len(batch["raw_data"])
            if save_predict_label:
                for idx, sample in enumerate(batch["raw_data"]):
                    result.append(
                        {
                            "id": sample["id"],
                            "document1": "".join(sample["document1"]),
                            "document2": "".join(sample["document2"]),
                            "label": sample["label"],
                            "predict": predict[idx],
                        }
                    )

        if save_predict_label:
            if result_dir is not None and result_prefix is not None:
                result_file = os.path.join(result_dir, result_prefix + ".json")
                self.logger.info("Write predict label to {}".format(result_file))
                with open(result_file, "w") as fout:
                    fout.write("id\tdoc1\tdoc2\tpredict\tlabel\n")
                    for tmp in result:
                        fout.write(self._json_2_string(tmp) + "\n")

        return (
            total_loss / float(total_num),
            total_accuracy / float(index),
            predicts,
            labels,
        )

    def predictiton(
            self, batch_data, result_file, save_predict_label=False
    ):
        """
        Evaluate the model with data
        Args:
            batch_data: iterable batch data
            result_dir: the director to save the predict answers ,
                        answers will not save if None
            result_prefix: the prefix of file for saving the predict answers,
                           answers will not save if None
            save_predict_label: if True, the pred_answers will be added to raw sample and saved
            character: use character feature
        """
        if save_predict_label:
            result = []
        index = 0
        predicts = []
        for idx, batch in enumerate(batch_data):
            feed_dict = {
                self.document1: batch["document1_ids"],
                self.document2: batch["document2_ids"],
                self.document1_character: batch["document1_character_ids"],
                self.document2_character: batch["document2_character_ids"],
                self.dropout:0
            }

            predict = self.sess.run(
                [self.predict], feed_dict
            )
            index += 1
            predict = predict[0]
            predicts.extend(predict.tolist())

            if save_predict_label:
                for idx, sample in enumerate(batch["raw_data"]):
                    result.append(
                        {
                            "id": sample["id"],
                           # "document1": "".join(sample["document1"]),
                           # "document2": "".join(sample["document2"]),
                            "predict": predict[idx],
                        }
                    )

        if save_predict_label:
            self.logger.info("Write predict label to {}".format(result_file))
            with open(result_file, "w") as fout:
                for tmp in result:
                    fout.write(self._json_2_string(tmp, predict=True) + "\n")

        return predicts,

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            "Model saved in {}, with prefix {}.".format(model_dir, model_prefix)
        )

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            "Model restored from {}, with prefix {}".format(model_dir, model_prefix)
        )

    def _json_2_string(self, json_obj, predict=False):
        if predict:
            s = json_obj['id'] + '\t' + str(json_obj['predict'])
        else:
            s = (
            json_obj["id"]
            + "\t"
            + json_obj["document1"]
            + "\t"
            + json_obj["document2"]
            + "\t"
            + str(json_obj["predict"])
            )
        return s

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2
