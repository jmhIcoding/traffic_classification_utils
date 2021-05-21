__author__ = 'dk'
from wf_attacks.data_utils import LoadDataApp_crossversion
from wf_attacks.appscanner.feature_extractor import  feature_extract
import  lightgbm as lgb
from sklearn.metrics import  accuracy_score
import numpy as np
from wf_attacks.appscanner.min_max import  _min ,_max
from accuracy_per_class import accuracy_per_class
def main(test_set,modelpath):
    ##原始的包长序列
    global  _min,_max
    X_train_r, y_train_r, X_valid_r, y_valid_r, X_test_r, y_test_r =  LoadDataApp_crossversion(test_set)
    saved_model = modelpath #"appscanner.model"
    print('before extract feature')
    ##提取统计特征
    X_train =[]
    X_valid =[]
    X_test  =[]
    #print(X_train_r[0])
    #print(X_valid_r[1])
    #print(X_test_r[2])
    for i in range(X_train_r.shape[0]):
        X_train.append(feature_extract(X_train_r[i]))
    for i in range(X_test_r.shape[0]):
        X_test.append(feature_extract(X_test_r[i]))
    for i in range(X_valid_r.shape[0]):
        X_valid.append(feature_extract(X_valid_r[i]))
    print('feature extract well!')
    ##归一化操作


    _min = np.array(_min)
    _max =np.array(_max)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    _min = np.array(_min)
    _max =np.array(_max)
    X_train = (X_train-_min)/(_max-_min)
    X_valid = (X_valid-_min)/(_max-_min)
    X_test = (X_test-_min)/(_max-_min)
    print('normalize well!')
    print(X_train[0])
    print(X_valid[1])
    print(X_test[2])
    ##
    y_test = np.argmax(y_test_r,1)
    y_train =np.argmax(y_train_r,1)
    y_valid =np.argmax(y_valid_r,1)
    print(y_test[0:10])
    #load model
    try:
        gbm = lgb.Booster(model_file=saved_model)
    except BaseException as exp:
        raise exp
    logit = gbm.predict(data=X_test)
    label_predict = list(map(lambda x : np.argmax(x),logit))

    accuracy = accuracy_score(y_test,label_predict)
    accuracy_per_class(y_real=y_test,y_pred=label_predict)

    print("[Appscanner] Test on {0}, accuracy is {1}. ".format(test_set,accuracy))

