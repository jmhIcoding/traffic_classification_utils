__author__ = 'jmh081701'
import sklearn
from sklearn.externals import joblib
from sklearn.metrics import  accuracy_score
import lightgbm as lgb
import  numpy as np
from src.cumul.util import CUMUL_datagenerator
dator = CUMUL_datagenerator(is_train=True)

saved_model = "saved_model/cumul.model"
train_X,train_y = dator.trainSet()
valid_X,valid_y = dator.validSet()
test_X,test_y = dator.testSet()

lgb_train = lgb.Dataset(data=train_X,label=train_y)
lgb_eval = lgb.Dataset(data=valid_X,label=valid_y)

hyper_params = {
   'boosting_type': 'rf',
 'objective': 'multiclass',
 'num_leaves': 512,
 'learning_rate': 0.05,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.8,
 'bagging_freq': 5,
 'verbose': 0,
 'num_class':100,
 'lambda_l1':0.05,
 'lambda_l2':0.15
}

gbm = lgb.train(params=hyper_params,
                train_set=lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=3000,
                early_stopping_rounds=10)

logit = gbm.predict(data=test_X)
label_predict = list(map(lambda x : np.argmax(x),logit))

accuracy = accuracy_score(test_y,label_predict)
print(accuracy)

#save model
gbm.save_model(saved_model)





