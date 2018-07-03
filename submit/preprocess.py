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

from langconv import Converter

reload(sys)
sys.setdefaultencoding("utf-8")
import re
def preprocess_line(line):
    return re.sub(
                u"[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+", "", line
            )


def change_sentence(sentence):
    # 去除标点符号
    sentence = Converter('zh-hans').convert(sentence)
    sentence = sentence.replace(",", "")
    sentence = sentence.replace("，", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace("。", "")
    sentence = sentence.replace("?", "")
    sentence = sentence.replace("？", "")
    sentence = sentence.replace("!", "")

    # 替换某些词语
    sentence = sentence.replace("借贝", "借呗")
    sentence = sentence.replace("花贝", "花呗")
    sentence = sentence.replace('花吧', "花呗")
    sentence = sentence.replace('借吧', '借呗')
    sentence = sentence.replace("蚂蚁借呗", "借呗")
    sentence = sentence.replace("蚂蚁花呗", "花呗")
    sentence = sentence.replace("蚂蚁花呗", "花呗")
    sentence = sentence.replace("整么", "怎么")
    sentence = sentence.replace("冻解", "冻结")
    sentence = sentence.replace("撤掉", "撤销")
    sentence = sentence.replace("提额", "提高额度")
    sentence = sentence.replace("买机票", "订机票")
    sentence = sentence.replace("什时", "什么时候")


    # 将***替换成N
    sentence = re.sub(r'[*]+', "N", sentence)

    return sentence