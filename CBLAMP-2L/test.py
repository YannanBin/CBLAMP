# -*- coding: utf-8 -*-
# @Author  : zlj
# @FileName: test.py
# @Software: PyCharm

import os
import time
from keras import Model
from evaluation import scores, evaluate
from keras.models import load_model
import numpy as np
from sklearn.metrics import roc_curve
from tfloss import focal_loss, FocalDiceLoss, AsymmetricLoss, MultiClassDiceLoss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
filenames = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite", "Antiviral"]

def score_threshold(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]
    return threshold

def predict(seq_test, fea_test, bert_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):
    aucv, auprv, f1v, throldv = [], [], [], []
    pf = []
    throldvalue = []
    print("Prediction is in progress")

    for ii in range(0, len(weights)):

        h5_model_path = os.path.join(dir, h5_model[ii])

        load_my_model = load_model(h5_model_path)

        # 2.predict
        score = load_my_model.predict({"main_input_seq": seq_test, "main_input_fea": fea_test, "main_input_bert": bert_test})

        if ii == 0:
            score_pro = score
        else:
            score_pro += score
    score_pro = np.array(score_pro)
    score_pro = score_pro / len(h5_model)
    for i in range(len(filenames)):
        p = []
        print('{}:'.format(filenames[i]))
        p.append('{}:'.format(filenames[i]))
        y_pred = score_pro[:, i]
        yy_test = np.array(y_test[:, i])
        y_pre = np.array(y_pred)
        throld = 0.5 
        throldvalue.append(throld)
        SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp = scores(yy_test, y_pre, throld)
        print(SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp)
        aucv.append(AUC)
        auprv.append(AUPR)
        f1v.append(F1)
        throldv.append(throld)
        print('AUC value:{}'.format(AUC))
        print('AUPR value:{}'.format(AUPR))
        print('F1 value:{}'.format(F1))
        print('throld value:{}\n'.format(throld))
        p.append('throld value:{}'.format(throld))
        p.append('AUC value:{}'.format(AUC))
        p.append('AUPR value:{}'.format(AUPR))
        p.append('SN value:{}'.format(SN))
        p.append('SP value:{}'.format(SP))
        pf.append(p)
        data = []
        c = []
        data.append(filenames[i])
        data.append(
            'throld:{}, SN:{}, SP:{}, Precision:{}, F1:{}, Acc:{}, AUC:{}, AUPR:{}, tp:{}, fn:{}, tn:{}, fp:{}'.format(throld, SN, SP, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp))
        c.append(data)
        with open("result/result.txt", 'ab') as x:
            np.savetxt(x, np.asarray(c), fmt="%s\t")

    score_label = score_pro
    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < throldvalue[j]:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('precision:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    data0 = []
    cc = []
    data0.append(
        'precision:{}, coverage:{}, accuracy:{}, absolute_true:{}, absolute_false:{}'.format(
            aiming, coverage, accuracy, absolute_true, absolute_false))
    cc.append(data0)
    with open("result/wqzresult.txt", 'ab') as x:
        np.savetxt(x, np.asarray(cc), fmt="%s\t")

    data1 = []
    data1.append(aucv)
    data1.append(auprv)
    data1.append(f1v)
    data1.append(throldv)
    data1.append("\n")
    with open("result/result1.txt", 'ab') as x:
        np.savetxt(x, np.asarray(data1), fmt="%s,")
        v = []
        ttt = time.localtime(time.time())
        v.append(
            'finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))
        np.savetxt(x, np.asarray(v), fmt="%s\t")

    pf.append("\n")

    with open("result/result2.txt", 'ab') as x:
        np.savetxt(x, np.asarray(pf), fmt="%s,")
        v = []
        ttt = time.localtime(time.time())
        v.append(
            'finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))
        np.savetxt(x, np.asarray(v), fmt="%s\t")


def test_my(test, para, model_num, dir, k):
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num + 1):
        weights.append('model{}.hdf5'.format(str(k) + '_' + str(i)))
        jsonFiles.append('model{}.json'.format(str(k) + '_' + str(i)))
        h5_model.append('model{}.h5'.format(str(k) + '_' + str(i)))

    # step2:predict
    print('\nRecall, SPE, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp\n')

    predict(test[0], test[1], test[2], test[3], test[4], para, weights, jsonFiles, h5_model, dir)
    # predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)

    ttt = time.localtime(time.time())
    with open("result/result.txt", 'ab') as x:
        v = []
        v.append(
            'finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))
        np.savetxt(x, np.asarray(v), fmt="%s\t")
