# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: model.py
# @Software: PyCharm

import os
import tensorflow as tf
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, Bidirectional
from keras.layers import Flatten, Dense, Activation, CuDNNLSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def BiGRU_base(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    un = para['lstmunit']
    l2value = 0.0001

    main_input = Input(shape=(length,), dtype='float32', name='main_input')

    x = Embedding(output_dim=ed, input_dim=64, input_length=length, name='Embadding')(main_input)

    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1)(a)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1)(b)

    c = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1)(c)

    d = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=ps, strides=1)(d)

    merge = Concatenate(axis=-1)([apool, bpool, cpool, dpool])
    cf = Dropout(dp)(merge)

    #add BILSTM
    b = Bidirectional(CuDNNLSTM(units=un, return_sequences=True))(cf)
    bf = Dropout(dp)(b)

    f = Flatten()(bf)
    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(f)

    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
