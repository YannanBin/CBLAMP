# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: test.py
# @Software: PyCharm

import os
import time
import pandas as pd
from keras import Model
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, roc_curve, auc, f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
from keras.optimizers import adam_v2
adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]

def scores(y_test, y_pred, th):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if int(item) < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    SN = tp * 1.0 / ((tp + fn) * 1.0)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp

def score_threshold(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]
    return threshold

def predict(x_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):
    aucv, auprv, f1v = [], [], []
    pf = []
    print("Prediction is in progress")

    for ii in range(0, len(weights)):

        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path)

        # 2.predict
        score = load_my_model.predict(x_test)

        if ii == 0:
            score_pro = score
        else:
            score_pro += score
    score_pro = np.array(score_pro)
    score_pro = score_pro / len(h5_model)

    data = []
    for i in range(len(score_pro)):
        d = []
        d.append(y_test[i][0])
        d.append(score_pro[i][0])
        data.append(np.array(d))
    np.savetxt('result/label.txt', np.asarray(data), fmt="%s\t")

    for i in range(1):
        p = []
        y_pred = score_pro[:, i]
        yy_test = np.array(y_test)
        y_pre = np.array(y_pred)
        throld = 0.5

        SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp = scores(yy_test, y_pre, throld)

        aucv.append(AUC)
        auprv.append(AUPR)
        f1v.append(F1)
        print('throld value:{}\n'.format(throld))
        print('AUC value:{}'.format(AUC))
        print('AUPR value:{}'.format(AUPR))
        print('SN:{}'.format(SN))
        print('SP:{}'.format(SP))
        p.append('throld value:{}'.format(throld))
        p.append('AUC value:{}'.format(AUC))
        p.append('AUPR value:{}'.format(AUPR))
        p.append('SN value:{}'.format(SN))
        p.append('SP value:{}'.format(SP))
        pf.append(p)
        data = []
        c = []
        data.append(
            'throld:{}, SN:{}, SP:{}, Precision:{}, F1:{}, Acc:{}, AUC:{}, AUPR:{}, tp:{}, fn:{}, tn:{}, fp:{}'.format(
                throld, SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp))
        c.append(data)
        with open("result/result.txt", 'ab') as x:
            np.savetxt(x, np.asarray(c), fmt="%s\t")

def test_my(test, para, model_num, dir, k):
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num + 1):
        weights.append('models{}.hdf5'.format(str(k) + '_' + str(i)))
        jsonFiles.append('models{}.json'.format(str(k) + '_' + str(i)))
        h5_model.append('models{}.h5'.format(str(k) + '_' + str(i)))

    # step2:predict
    print("test")
    print(test[0].shape)
    print(test[1].shape)
    print('\nSN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp\n')

    predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)

    ttt = time.localtime(time.time())
    with open("result/wqzresult.txt", 'ab') as x:
        v = []
        v.append(
            'finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))
        np.savetxt(x, np.asarray(v), fmt="%s\t")

