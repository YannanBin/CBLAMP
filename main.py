# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: main.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import KFold
from train import train_main
import tensorflow as tf
from sklearn import preprocessing

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0' 
t = time.localtime(time.time())
modelDir = 'models'
Path(modelDir).mkdir(exist_ok=True)
t = time.localtime(time.time())

filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]
seq_length = 180

def PadEncode(data, label, max_len): 
    # encoding
    amino_acids = 'JACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return data_e, label_e


def TrainAndTest(tr_data, tr_label, te_data, te_label):
    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 1
    test.append(threshold)

    for i in range(1):
        train_main(train, test, model_num, modelDir, i)

    ttt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))


def main():

    max_length = 180
    # train
    train_sequence_data = pd.read_csv('./data/train.csv', header=None).values[1:, 1]
    y_train = pd.read_csv('./data/train.csv', header=None).values[1:, 2]

    seq_train, y_train = PadEncode(train_sequence_data, y_train, max_length)

    x_train = np.array(seq_train)
    y_train = pd.DataFrame(data=y_train)
    y_train = np.array(y_train)

    # test
    test_sequence_data = pd.read_csv('./data/test.csv', header=None).values[1:, 1]
    y_test = pd.read_csv('./data/test.csv', header=None).values[1:, 2]

    seq_test, y_test = PadEncode(test_sequence_data, y_test, max_length)

    x_test = np.array(seq_test)
    y_test = pd.DataFrame(data=y_test)
    y_test = np.array(y_test)

    TrainAndTest(x_train, y_train, x_test, y_test)

    # # 五折交叉验证
    # skf = KFold(n_splits=5, shuffle=True, random_state=0)
    # for train_index, val_index in skf.split(x_train, y_train):
    #
    #     x_tra, y_tra = x_train[train_index], y_train[train_index]
    #     x_val, y_val = x_train[val_index], y_train[val_index]
    #
    #     x_tra = np.array(x_tra)
    #     x_val = np.array(x_val)
    #     y_tra = np.array(y_tra)
    #     y_val = np.array(y_val)
    #
    #     TrainAndTest(x_tra, y_tra, x_val, y_val)

if __name__ == '__main__':
    main()
