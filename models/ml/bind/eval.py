from BIND.build_vector_dataset import builder
import  lightgbm as lgb
from sklearn.metrics import  accuracy_score
import numpy as np
from accuracy_per_class import accuracy_per_class
def main(raw_feature_dictory,modelpath,global_feature_dict_filename="./global_feature_dict.vocb"):
    ##原始的包长序列
    bd = builder(raw_feature_dictory=raw_feature_dictory,global_feature_dict_filename=global_feature_dict_filename)
    X_train,y_train,X_test,y_test,X_valid,y_valid=bd.vectorize(test_split_ratio=0.5)
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
    gbm = lgb.Booster(model_file=saved_model)
    logit = gbm.predict(data=X_test)
    label_predict = list(map(lambda x : np.argmax(x),logit))

    accuracy = accuracy_score(y_test,label_predict)
    accuracy_per_class(y_real=y_test,y_pred=label_predict)

    print(accuracy)