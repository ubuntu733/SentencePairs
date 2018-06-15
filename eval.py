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

from vocab2 import PretrainedVocab

reload(sys)
sys.setdefaultencoding("utf-8")
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
    parser = argparse.ArgumentParser("Classification on VIDAA dataset")
    parser.add_argument(
        '--predict', action='store_true', help='predict the model on test set'
    )
    parser.add_argument("--char", action="store_true", help="use char embedding")

    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")

    train_settings = parser.add_argument_group("train settings")

    train_settings.add_argument(
        "--filter_sizes", type=list, default=[5], help="一维卷积核大小"
    )
    train_settings.add_argument("--num_filters", type=int, default=32, help="卷积核数量")
    train_settings.add_argument("--optim", default="adam", help="optimizer type")
    train_settings.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    train_settings.add_argument(
        "--weight_decay", type=float, default=0, help="weight decay"
    )
    train_settings.add_argument("--dropout", type=float, default=0, help="dropout rate")
    train_settings.add_argument(
        "--batch_norm", action="store_true", help="whether use batch norm or not"
    )
    train_settings.add_argument(
        "--batch_size", type=int, default=64, help="train batch size"
    )
    train_settings.add_argument("--epochs", type=int, default=10, help="train epochs")
    train_settings.add_argument(
        "--hidden_size", type=int, default=128, help="number of rnn hidden unit"
    )
    train_settings.add_argument(
        "--max_document_len", type=int, default=10, help="max length of document"
    )
    train_settings.add_argument(
        "--max_word_len", type=int, default=5, help="max length of word"
    )
    model_settings = parser.add_argument_group("model settings")

    model_settings.add_argument(
        "--embedding_size", type=int, default=300, help="size of the embeddings"
    )
    model_settings.add_argument(
        "--character_embedding_size",
        type=int,
        default=100,
        help="size of the character embeddings",
    )
    model_settings.add_argument("--class_model", type=str, default="rnn")
    model_settings.add_argument(
        "--pretrained_embedding", action="store_true", help="use pretrained embeddings"
    )

    path_settings = parser.add_argument_group("path settings")
    path_settings.add_argument(
        "--pretrained_file",
        default="data/pretrained_embedding.utf8",
        help="the file to save pretrained word embeddings",
    )
    path_settings.add_argument(
        "--data_files",
        help="list of files that contain the preprocessed train data",
    )
    path_settings.add_argument(
        "--preposs_file",
        default="data/train.data",
        help="the file with ltp token segment",
    )
    path_settings.add_argument("--dev_fils", default="dev.data")
    path_settings.add_argument(
        "--model_dir", default="data/models/", help="the dir to store vocab"
    )
    path_settings.add_argument(
        "--vocab_dir", default="data/vocab/", help="the dir to store models"
    )
    path_settings.add_argument(
        "--result_file", help="the dir to output the results"
    )
    path_settings.add_argument(
        "--summary_dir",
        default="data/summary/",
        help="the dir to write tensorboard summary",
    )
    path_settings.add_argument(
        "--dict_file", default="data/dict", help="user dict of jieba"
    )
    path_settings.add_argument(
        "--log_path",
        help="path of the log file. If not set, logs are printed to console",
    )
    path_settings.add_argument(
        '--test_file', type=str
    )
    return parser.parse_args()


def prepare(args):
    """
    Checks data, create vocab, load pretrained embedding or initialization randomly embedding
    """
    logger = logging.getLogger("alibaba")
    logger.info("Checking the data files... ")
    logger.info("preprocess raw data...")
    jieba.load_userdict(args.dict_file)
    logger.info("segment raw data")
    preposs_file = open(args.preposs_file, "w")
    with open(args.test_file, "r") as fin:
        for idx, line in enumerate(fin):
            line = unicode(line, encoding="utf8")
            line_re = re.sub(
                u"[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+", "", line
            )
            line_list = str(line_re).strip("\n").split("\t")
            if len(line_list) != 3:
                logger.warning(
                        "{} - {} from is wrong".format(args.data_files, idx + 1)
                    )
                continue
            document1 = line_list[1].strip().replace(" ", "")
            document2 = line_list[2].strip().replace(" ", "")
            segment_document1 = [_ for _ in jieba.cut(document1)]
            segment_document2 = [_ for _ in jieba.cut(document2)]
            preposs_file.write(line_list[0])
            preposs_file.write("|")
            preposs_file.write(" ".join(segment_document1))
            preposs_file.write("|")
            preposs_file.write(" ".join(segment_document2))
            preposs_file.write("\n")
    preposs_file.close()


def evaluate(args):
    """
    Evaluate the classification model
    """
    logger = logging.getLogger("alibaba")
    logger.info("Load data_set , vocab and label config...")
    if args.pretrained_embedding:
        word_vocab_ = PretrainedVocab(args)

    else:
        with open(os.path.join(args.vocab_dir, "vocab.data"), "rb") as fin:
            word_vocab_ = pickle.load(fin)
    with open(os.path.join(args.vocab_dir, "vocab_character.data"), "rb") as fin:
        vocab_character_ = pickle.load(fin)
    data = dataset.Dataset(args)
    logger.info("Convert word to id...")
    data.convert_to_ids(word_vocab_, set_name='test')
    logger.info("Convert character to id...")
    data.convert_to_ids(vocab_character_, character=True, set_name='test')
    logger.info("Build Model...")
    model_ = model.Model(args, word_vocab=word_vocab_, character_vocab=vocab_character_)
    model_.restore(model_dir=args.model_dir, model_prefix=args.class_model)
    logger.info("Evaluating the model on dev set...")
    dev_batchs = data.get_mini_batchs(batch_size=args.batch_size, set_name="test", predict=True)
    _ = model_.predictiton(
        batch_data=dev_batchs,
        result_file=args.result_file,
        save_predict_label=True,
    )
    logger.info(
        "Predicted labels are saved to {}".format(args.result_file)
    )

if __name__=="__main__":
    args = parse_args()

    logger = logging.getLogger("alibaba")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path, encoding="UTF-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info("Running with args : {}".format(args))
    prepare(args)
    evaluate(args)