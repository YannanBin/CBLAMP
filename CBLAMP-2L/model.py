# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: model.py
# @Software: PyCharm

import os
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, Bidirectional, Flatten, Dense, CuDNNLSTM
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def seq_fea(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='float32', name='main_input_seq')
    x = Embedding(output_dim=ed, input_dim=20, input_length=length, name='embedding')(main_input)
    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool1')(a)
    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool2')(b)
    c = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool3')(c)
    d = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool4')(d)
    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool, dpool])
    x1 = Dropout(dp)(merge)
    x1 = Bidirectional(CuDNNLSTM(80, return_sequences=True))(x1)
    x1 = Flatten(name='fla_seq')(x1)
    x1 = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x1)

    fea = Input(shape=(180, 192), dtype='float32', name='main_input_fea')
    d = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(fea)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(d)
    e = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(fea)
    epool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(e)
    f = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(fea)
    fpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(f)
    g = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(fea)
    gpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(g)
    mergefea = Concatenate(axis=-1)([dpool, epool, fpool, gpool])
    merge1 = Dropout(dp)(mergefea)
    x2 = Bidirectional(CuDNNLSTM(80, return_sequences=True))(merge1)
    x2 = Flatten(name='fla_fea')(x2)
    x2 = Dense(fd, activation='relu', kernel_regularizer=l2(l2value))(x2)

    bertfea = Input(shape=(180, 768), dtype='float32', name='main_input_bert')
    h = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(bertfea)
    hpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(h)
    i = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(bertfea)
    ipool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(i)
    j = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(bertfea)
    jpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(j)
    k = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(bertfea)
    kpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(k)
    mergebert = Concatenate(axis=-1)([hpool, ipool, jpool, kpool])
    merge2 = Dropout(dp)(mergebert)
    x3 = Bidirectional(CuDNNLSTM(80, return_sequences=True))(merge2)
    x3 = Flatten(name='fla_bert')(x3)
    x3 = Dense(fd, activation='relu', kernel_regularizer=l2(l2value))(x3)

    cc = Concatenate(axis=-1, name='lastLayer')([x1, x2, x3])
    output = Dense(out_length, activation='sigmoid', name='output')(cc)

    model = Model(inputs=[main_input, fea, bertfea], outputs=output, name='cnn_fea')

    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
