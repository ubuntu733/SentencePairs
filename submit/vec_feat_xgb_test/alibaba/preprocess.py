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




if __name__ == "__main__":
    files = ['alibaba/train.csv', 'alibaba/dev.csv']
    write_file = open('../fasttext.data','w')
    for file in files:
        with open(file,'r') as fin:
            for line in fin:
                line = unicode(line.strip(), encoding='utf8')
                line_list = line.split("\t")
                write_file.write(line_list[1]+'\n')
                write_file.write(line_list[2] + '\n')
    write_file.close()
