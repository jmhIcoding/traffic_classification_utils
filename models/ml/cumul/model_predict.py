__author__ = 'jmh081701'
import  lightgbm as lgb
from src.cumul.util import CUMUL_datagenerator
from sklearn.metrics import  accuracy_score
import  numpy as np
saved_model = "./saved_model/cumul.model"
model = lgb.Booster(model_file=saved_model)

dator = CUMUL_datagenerator(is_train=True)
def prediction(X):
    #X = dator.feature_extract(X)
    logit  = model.predict(data=X)
    y = list(map(lambda x : np.argmax(x),logit))
    #assert len(y.shape) == X.shape[0]

    return y

if __name__ == '__main__':
    dator = CUMUL_datagenerator(is_train=True)
    test_X,test_y = dator.testSet()
    predict_y = prediction(test_X)
    accuracy = accuracy_score(test_y,predict_y)
    print('test accuracy:{0}'.format(accuracy))

