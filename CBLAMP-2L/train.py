# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: train.py
# @Software: PyCharm

import math
import os
import random
import numpy as np
import tensorflow as tf
import time
from keras import Model
from test import test_my
from pathlib import Path
from model import BiGRU_base, seq_fea

filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

random.seed(101)
np.random.seed(101)


def train_my(train, para, model_num, model_path, i, data_size):
    Path(model_path).mkdir(exist_ok=True)

    # data get
    seq_train, fea_train, bert_train, y_train = train[0], train[1], train[2], train[3]
    index = np.arange(len(y_train))
    np.random.shuffle(index)

    seq_train = seq_train[index]
    fea_train = fea_train[index]
    bert_train = bert_train[index]

    y_train = y_train[index]

    # train
    length = seq_train.shape[1]
    out_length = y_train.shape[1]
    print(length)
    print(out_length)

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour, t_data.tm_min, t_data.tm_sec))

    for counter in range(1, model_num + 1):
        # get models
        if model_path == 'Models':
            model = seq_fea(length, out_length, para)
        else:
            print('no models')

        model.fit({"main_input_seq": seq_train, "main_input_fea": fea_train, "main_input_bert": bert_train}, y_train, epochs=60, batch_size=64, verbose=2)


        each_model = os.path.join(model_path, 'model' + str(i) + "_" + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter), tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))


def train_main(train, test, model_num, dir, i, data_size):
    # parameters
    ed = 192
    ps = 5
    dp = 0.6
    lr = 0.003
    fd = 64
    un = 80

    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd, 'drop_out': dp,
            'learning_rate': lr, "lstmunit": un}


    train_my(train, para, model_num, dir, i, data_size)

    test_my(test, para, model_num, dir, i)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write(
            'test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
