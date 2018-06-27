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
import numpy as np
import json
if __name__ == "__main__":
    with open('result.json', 'r') as fin:
        result_json = json.load(fin,encoding='utf-8')
        sample = {}
        for key in result_json:
            tmp = result_json[key]
            if tmp['prediction'] != tmp['truth']:
                sample[key] = tmp
    fp_write = open('diff.json','w')
    json.dump(sample, fp_write)
    '''
    sample = []
    with open('train.data','r') as fin:
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
    '''

