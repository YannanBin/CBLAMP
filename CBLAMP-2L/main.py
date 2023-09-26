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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
t = time.localtime(time.time())
modelDir = 'Models'
Path(modelDir).mkdir(exist_ok=True)
t = time.localtime(time.time())

filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]

def PadEncode(data, label, max_len):
    # encoding
    amino_acids = 'JACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
    for i in range(len(data)):
        length = len(data[i])
        elemt = []
        st = data[i].strip()
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

def TrainAndTest(tr_data, tr_label, te_data, te_label, data_size):
    train = [tr_data[0], tr_data[1], tr_data[2], tr_label]
    test = [te_data[0], te_data[1], te_data[2], te_label]

    threshold = 0.5
    model_num = 1 
    test.append(threshold)

    for i in range(1):
        train_main(train, test, model_num, modelDir, i, data_size)

    ttt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))

# Count the number of each category
def staticTrainandTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    print("TrainingSet:\n")
    for i in range(len(filenames)):
        print('{}:{}\n'.format(filenames[i], int(data_size_tr[i])))

    print("TestingSet:\n")
    for i in range(len(filenames)):
        print('{}:{}\n'.format(filenames[i], int(data_size_te[i])))

    return data_size_tr

# read BERT data
import json
train_json_file = "./BertData/train_-1.json"
test_json_file = "./BertData/test_-1.json"
from jsonpath import jsonpath

all_data_layer1 = []
def read_json(file):
    all_data_layer1=[]
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data_layer1 = np.zeros([180, 768], dtype=np.float32)
        content = json.loads(line)
        value = jsonpath(content, "$..values")
        for i in range(3, len(value)-1):
            data_layer1[i-3] += value[i]
        all_data_layer1.append(data_layer1)
    np_all_data_layer1 = np.array(all_data_layer1)
    return np_all_data_layer1

def main():

    max_length = 180
    # train
    train_sequence_data = pd.read_csv('./data/train.csv', header=None).values[1:, 1]
    y_train = pd.read_csv('./data/train_label.csv', header=None).values

    seq_train, y_train = PadEncode(train_sequence_data, y_train, max_length)

    seq_train = np.array(seq_train)
    y_train = np.array(y_train)

    # test
    test_sequence_data = pd.read_csv('./data/test.csv', header=None).values[1:, 1]
    y_test = pd.read_csv('./data/test_label.csv', header=None).values

    seq_test, y_test = PadEncode(test_sequence_data, y_test, max_length)

    seq_test = np.array(seq_test)
    y_test = np.array(y_test)

    data_size = staticTrainandTest(y_train, y_test)

    # Read the BERT feature files
    bert_train = read_json(train_json_file)
    bert_test = read_json(test_json_file)

    # Read the AAindex feature files
    fea_train = np.load("./AAindex/Data/train_data_192.h5.npy")
    fea_test = np.load("./AAindex/Data/test_data_192.h5.npy")

    TrainAndTest([seq_train, fea_train, bert_train], y_train, [seq_test, fea_test, bert_test], y_test, data_size)

    # #五折交叉
    # skf = KFold(n_splits=5, shuffle=True, random_state=0)
    # for train_index, val_index in skf.split(seq_train, y_train):
    #     x_tra, y_tra = seq_train[train_index], y_train[train_index]
    #     x_val, y_val = seq_train[val_index], y_train[val_index]
    #
    #     x_tra = np.array(x_tra)
    #     x_val = np.array(x_val)
    #     y_tra = np.array(y_tra)
    #     y_val = np.array(y_val)
    #
    #     TrainAndTest(x_tra, y_tra, x_val, y_val, data_size)


if __name__ == '__main__':
    main()

