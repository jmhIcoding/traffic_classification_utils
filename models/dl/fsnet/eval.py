from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import classification_report
ALL_ = -1
TPR_KEY = 'TPR'
FPR_KEY = 'FPR'
FTF_KEY = 'FTF'


def _fpr_trp_app(real, pred, app_ind):
    real_app = real == app_ind
    pred_app = pred == app_ind
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for r, p in zip(real_app, pred_app):
        if r and p:
            TP += 1
        elif r and not p:
            FN += 1
        elif not r and p:
            FP += 1
        else:
            TN += 1
    return TP, TN, FP, FN


def _evaluate_fpr_and_tpr(real, pred):
    app_num = len(pred)
    real = np.concatenate(real)
    pred = np.concatenate(pred)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TPR = {}
    FPR = {}
    for app_ind in tqdm(range(app_num), ascii=True, desc='Eval'):
        TP_app, TN_app, FP_app, FN_app = _fpr_trp_app(real, pred, app_ind)
        TP += TP_app
        TN += TN_app
        FP += FP_app
        FN += FN_app
        TPR[app_ind] = TP_app / (TP_app + FN_app)
        FPR[app_ind] = FP_app / (FP_app + TN_app)
    TPR[ALL_] = TP / (TP + FN)
    FPR[ALL_] = FP / (FP + TN)
    #print("Total Accuracy:",(TP+TN)/(TP+TN+FP+FN))
    return TPR, FPR


def _evaluate_ftf(TPR, FPR, class_num):
    res = 0
    sam_num = np.array(class_num, dtype=np.float)
    sam_num /= sam_num.sum()

    for key in TPR:
        if key == ALL_:
            continue
        res += sam_num[key] * TPR[key] / (1 + FPR[key])
    return res


def save_res(res, filename):
    with open(filename, 'w') as fp:
        json.dump(res, fp, indent=1, sort_keys=True)


def evaluate(real, pred):
    print('real.shape:{0},len.shape{1}'.format(np.array(real).shape,np.array(pred).shape))
    r=0
    t=0
    y_real =[]
    y_pred =[]
    for i in range(len(real)):
       for j in range(len(real[i])):
            y_real.append(real[i][j])
            y_pred.append(pred[i][j])
            if real[i][j]==pred[i][j]:
               r+=1
            t+=1

    example_len = [len(ix) for ix in real]
    TPR, FPR = _evaluate_fpr_and_tpr(real, pred)
    FTF = _evaluate_ftf(TPR, FPR, example_len)
    res = {
        TPR_KEY: TPR,
        FPR_KEY: FPR,
        FTF_KEY: FTF
    }
    print('Accuracy:',r*1.0/t)
    print(classification_report(y_true=y_real,y_pred=y_pred, digits=5))

    return res
