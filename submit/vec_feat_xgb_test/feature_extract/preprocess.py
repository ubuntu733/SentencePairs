#/usr/bin/env python
#coding=utf-8
import re


def change_sentence(sentence):
    # 去除标点符号
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
    sentence = sentence.replace("蚂蚁借呗", "借呗")
    sentence = sentence.replace("蚂蚁花呗", "花呗")
    sentence = sentence.replace("蚂蚁花呗", "花呗")
    sentence = sentence.replace("整么", "怎么")
    sentence = sentence.replace("冻解", "冻结")
    sentence = sentence.replace("撤掉", "撤销")
    sentence = sentence.replace("提额", "提高额度")
    sentence = sentence.replace("买机票", "订机票")

    # 将***替换成N
    sentence = re.sub(r'[*]+', "N", sentence)

    return sentence


def my_tokenizer(sentence):
    return sentence.split()
