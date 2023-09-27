#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : zlj
# @FileName: main.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd


def read_data(path):
    data, label = [], []
    with open(path) as f:
        for each in f:
            each = each.split(',')
            label.append(np.array(list(each[2:9]), dtype=int))  # Converting string labels to numeric vectors
            data.append(np.array(each[1]))
    return data, label


def read_AA_index(AA_index, embedding_length):
    AA_path = "{}/{}.txt".format("firstAAindex", "AAindex")
    data, find, not_find = [], [], []

    fr = open(AA_path, 'r')
    all_lines = fr.readlines()

    for line in all_lines[1:]:
        each_line = line.strip().split('\t')
        if each_line[0] in AA_index:
            find.append(each_line[0])
            data.append(np.array(list(each_line[1:]), dtype=float))

    for line in all_lines[1:]:
        each_line = line.strip().split('\t')
        if (len(not_find) + len(find) < embedding_length) and (each_line[0] not in AA_index):
            not_find.append(each_line[0])
            data.append(np.array(list(each_line[1:]), dtype=float))

    for i in AA_index:
        if i not in find:
            print("Sorry, can not find {}. Please check out!".format(i))
    print("find:\n", find, len(find))
    print("add:\n", not_find, len(not_find))
    return np.array(data)


def PadEncode(data, label, AA_index, max_len, embedding_length):  # 序列编码
    # encoding
    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        # length = len(data[i])
        aa = str(data[i].dtype)
        length = int(aa[2:])
        element, st = [], str(data[i]).strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            AA = AA_index[:, index]
            element.append(AA)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(element)
            seq_length.append(len(temp[b]))
            b += 1
            for k in range(max_len - length):
                element.append([0] * embedding_length)
            data_e.append(element)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e), np.array(seq_length)


def aaindex():
    train_path = "{}/{}.csv".format("firstAAindex", "train")
    test_path = "{}/{}.csv".format("firstAAindex", "test")
    AA_path = "{}/{}.txt".format("firstAAindex", "AAindex")
    train_data, train_label = read_data(train_path)
    test_data, test_label = read_data(test_path)
    aa = pd.read_csv('firstAAindex/aamr.txt_top192_mRMR_features.csv', header=None, sep=',').values[0, 1:]
    AA_indexes = list(aa)

    AA_pro = read_AA_index(AA_indexes, 192)
    print(AA_pro.shape)  # (192, 20)
    train_data, train_label, train_length = PadEncode(train_data, train_label, AA_pro, 180, 192)
    test_data, test_label, test_length = PadEncode(test_data, test_label, AA_pro, 180, 192)
    print(train_data.shape)  # (4511, 180, 192)

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':

    data_train, label_train, data_test, label_test = aaindex()

    PATH = os.getcwd()
    train_data_path = os.path.join(PATH, 'firstAAindex', 'result', 'train_data_192' + '.h5')
    test_data_path = os.path.join(PATH, 'firstAAindex', 'result', 'test_data_192' + '.h5')
    train_label_path = os.path.join(PATH, 'firstAAindex', 'result', 'train_label_192' + '.h5')
    test_label_path = os.path.join(PATH, 'firstAAindex', 'result', 'test_label_192' + '.h5')

    np.save(train_label_path, label_train)
    np.save(test_data_path, data_test)
    np.save(train_data_path, data_train)
    np.save(test_label_path, label_test)


