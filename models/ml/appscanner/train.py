__author__ = 'dk'
from wf_attacks.data_utils import LoadDataApp_crossversion
from wf_attacks.appscanner.feature_extractor import  feature_extract
import  lightgbm as lgb
from sklearn.metrics import  accuracy_score
import numpy as np
from wf_attacks.appscanner.min_max import  _max,_min
##原始的包长序列
def main(train_set,modelpath):
    global _min,_max
    X_train_r, y_train_r, X_valid_r, y_valid_r, X_test_r, y_test_r =  LoadDataApp_crossversion(train_set)
    saved_model = modelpath# "appscanner.model"
    print('before extract feature')
    ##提取统计特征
    X_train =[]
    X_valid =[]
    X_test  =[]

    for i in range(X_train_r.shape[0]):
        X_train.append(feature_extract(X_train_r[i]))
    for i in range(X_test_r.shape[0]):
        X_test.append(feature_extract(X_test_r[i]))
    for i in range(X_valid_r.shape[0]):
        X_valid.append(feature_extract(X_valid_r[i]))
    print('feature extract well!')
    ##归一化操作
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    _min = np.array(_min)
    _max =np.array(_max)
    print('_min:')
    print(_min)
    print('_max:')
    print(_max)

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
    ##开始训练

    lgb_train = lgb.Dataset(data=X_train,label=y_train)
    lgb_eval = lgb.Dataset(data=X_valid,label=y_valid)

    hyper_params = {
       'boosting_type': 'rf',
     'objective': 'multiclass',
     'num_leaves': 512,
     'learning_rate': 0.05,
     'feature_fraction': 0.9,
     'bagging_fraction': 0.8,
     'bagging_freq': 5,
     'verbose': 0,
     'num_class':55,
     'lambda_l1':0.05,
     'lambda_l2':0.15
    }

    gbm = lgb.train(params=hyper_params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=50,
                    early_stopping_rounds=5)
    #save model
    try:
        gbm.save_model(saved_model)
    except BaseException as exp:
        pass
    logit = gbm.predict(data=X_test)
    label_predict = list(map(lambda x : np.argmax(x),logit))

    accuracy = accuracy_score(y_test,label_predict)
    print('[Appscanner test on {0} acc:{1}]'.format(train_set,accuracy))
