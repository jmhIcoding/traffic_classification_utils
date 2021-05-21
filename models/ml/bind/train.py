__author__ = 'dk'
from BIND.build_vector_dataset import builder
import  lightgbm as lgb
from sklearn.metrics import  accuracy_score
import numpy as np
def main(raw_feature_dictory,modelpath,global_feature_dict_filename="./global_feature_dict.vocb"):
    ##原始的包长序列
    bd = builder(raw_feature_dictory=raw_feature_dictory,global_feature_dict_filename=global_feature_dict_filename)
    X_train,y_train,X_test,y_test,X_valid,y_valid=bd.vectorize()
    saved_model = modelpath
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
     'num_class':53,
     'lambda_l1':0.05,
     'lambda_l2':0.15
    }

    gbm = lgb.train(params=hyper_params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=100,
                    early_stopping_rounds=5)
    #save model
    try:
        gbm.save_model(saved_model)
    except BaseException as exp:
        pass
    logit = gbm.predict(data=X_test)
    label_predict = list(map(lambda x : np.argmax(x),logit))

    accuracy = accuracy_score(y_test,label_predict)
    print(accuracy)
if __name__ == '__main__':
    main("./raw_feature/",global_feature_dict_filename="./global_feature_dict.vocb")
