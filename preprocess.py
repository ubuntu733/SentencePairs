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
sys.setdefaultencoding("utf-8")
from langconv import *
import re
import jieba
import numpy as np

def preprocess_sentence(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    sentence = re.sub(r"蚂蚁花呗", "花呗", sentence)
    sentence = re.sub(r"蚂蚁借呗", "借呗", sentence)
    return sentence


if __name__ == "__main__":
    preposs_file = open('train.data', 'w')
    index = 1
    with open('test', "r") as fin:
        for idx, line in enumerate(fin):
            line = unicode(line, encoding="utf8")
            line_re = re.sub(
                u"[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+", "", line
            )
            line_list = str(line_re).strip("\n").split("\t")
            if len(line_list) != 4:
                print(
                    "{} - {} from is wrong".format('test', idx + 1)
                )
                continue
            document1 = line_list[1].strip().replace(" ", "")
            document2 = line_list[2].strip().replace(" ", "")
            segment_document1 = [_ for _ in jieba.cut(document1)]
            segment_document2 = [_ for _ in jieba.cut(document2)]
            preposs_file.write(str(index))
            preposs_file.write("|")
            preposs_file.write(" ".join(segment_document1))
            preposs_file.write("|")
            preposs_file.write(" ".join(segment_document2))
            preposs_file.write("|")
            preposs_file.write(line_list[3] + "\n")
            index += 1
    sample = []
    with open('train.data', 'r') as fin:
        for line in fin:
            line = line.decode('utf-8').strip()
            line_list = line.split('|')
            if len(line_list) != 4:
                print('error')
                continue
            sample.append(line_list)
    index = np.arange(len(sample))
    np.random.shuffle(index)
    start = int(len(sample) * (1 - 0.2))
    train_indices = index[0:start]
    dev_indices = index[start:-1]
    train_set = []
    dev_set = []
    for idx in train_indices:
        train_set.append(sample[idx])
    for idx in dev_indices:
        dev_set.append(sample[idx])
    train_file = open('train.csv', 'w')
    dev_file = open('dev.csv', 'w')
    for data in train_set:
        train_file.write(data[3] + '\t')
        train_file.write(data[1] + '\t')
        train_file.write(data[2] + '\t')
        train_file.write(data[0] + '\n')
    for data in dev_set:
        dev_file.write(data[3] + '\t')
        dev_file.write(data[1] + '\t')
        dev_file.write(data[2] + '\t')
        dev_file.write(data[0] + '\n')
    train_file.close()
    dev_file.close()