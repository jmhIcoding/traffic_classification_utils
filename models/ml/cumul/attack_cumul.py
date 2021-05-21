__author__ = 'jmh081701'
import  lightgbm as lgb
from src.cumul.util import CUMUL_datagenerator
from sklearn.metrics import  accuracy_score
from src.df.src import utility
import  numpy as np
saved_model = "./saved_model/cumul.model"
model = lgb.Booster(model_file=saved_model)
dator = CUMUL_datagenerator(is_train=False)
def prediction(X):
    print(X.shape)
    X = dator.feature_extract(X)
    logit  = model.predict(data=X)
    y = list(map(lambda x : np.argmax(x),logit))
    #assert len(y.shape) == X.shape[0]

    return y
def flatten(X_compressed):
    X =[]
    for i in range(X_compressed.shape[0]):
        x =[]
        for j in range(X_compressed.shape[1]):
            if (X_compressed[i,j])<0:
                x += [-1] * abs(int(X_compressed[i,j]))
            elif X_compressed[i,j] >0 :
                x +=[1] * abs(int(X_compressed[i,j]))
        x+=[0] * 5000
        X.append(x[:5000])
    return np.array(X)
if __name__ == '__main__':
    preprocess = CUMUL_datagenerator(is_train=False)
    X_train, y_train, X_valid, y_valid, X_test, y_test = utility.LoadDataRetrain(is_cluster=False,dataset_dir=None)
    predict_y = prediction(flatten(X_train))
    accuracy = accuracy_score(y_train,predict_y)
    print('test accuracy:{0}'.format(accuracy))
