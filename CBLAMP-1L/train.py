# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: train.py
# @Software: PyCharm

import os
import tensorflow as tf
import time
from tensorflow.python.framework.random_seed import set_random_seed
from keras.models import Model
from test import test_my
from pathlib import Path
from model import BiGRU_base
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

set_random_seed(101)
np.random.seed(101)

filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]

def train_my(train, para, model_num, model_path, i):
    Path(model_path).mkdir(exist_ok=True)

    x_train, y_train = train[0], train[1]

    index = np.arange(len(y_train))
    np.random.shuffle(index)

    x_train = x_train[index].astype('float64')
    y_train = y_train[index].astype('float64')

    # train
    length = x_train.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour,
                                                                      t_data.tm_min, t_data.tm_sec))

    for counter in range(1, model_num + 1):
        # get my neural network models
        if model_path == 'models':
            model = BiGRU_base(length, out_length, para)
        else:
            print('no models')

        model.fit(x_train, y_train, epochs=80, batch_size=64, verbose=2)

        each_model = os.path.join(model_path, 'models' + str(i) + "_" + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter), tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min,
                                                            tt.tm_sec))

def train_main(train, test, model_num, dir, i):
    # parameters
    ed = 192
    ps = 180
    dp = 0.6
    lr = 0.003
    fd = 64
    un = 50

    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd, 'drop_out': dp,
            'learning_rate': lr, "lstmunit": un}

    train_my(train, para, model_num, dir, i)
    test_my(test, para, model_num, dir, i)
    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write(
            'test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
