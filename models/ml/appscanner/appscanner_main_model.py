__author__ = 'dk'
from models.model_base import abs_model
import os
import shutil
import json
from config import raw_dataset_base, min_flow_len
from models.ml.appscanner import feature_extractor
import numpy as np
from models.ml.appscanner import min_max
import pickle
import  lightgbm as lgb
import tqdm
from sklearn.metrics import  accuracy_score,classification_report
from models.ml.appscanner.hyper_params import hyper_params

class model(abs_model):
    def __init__(self, dataset, randseed, splitrate):
        super(model,self).__init__('appscanner',randseed= randseed)
        if os.path.exists(self.database) == False:
            os.makedirs(self.database,exist_ok=True)

        self.dataset = dataset
        self.model = self.database + '/'+ self.name + '_' + dataset + '_model'
        self.data = self.database + '/'+ self.name + '_' + dataset + '/'
        self.splitrate = splitrate
        #原始数据集目录
        full_rdata = raw_dataset_base + self.dataset
        self.full_rdata = full_rdata

        if self.data_exists() == False:
            self.parser_raw_data()


    def parser_raw_data(self):
        full_rdata = self.full_rdata
        if os.path.exists(full_rdata) == False:
            raise OSError('Dataset {0} (full path: {1}) does not exist!'.format(self.dataset,full_rdata))
        #从原始数据集目录构建appscanner所需的数据集
        X = []
        y = []
        for _root, _dirs, _files in os.walk(full_rdata):
            labels = []
            for file in _files:
                labels.append(file)
            labels.sort()
            for file in tqdm.trange(len(_files)):
                file = _files[file]
                label = labels.index(file)
                file = _root + '/' + file

                with open(file) as fp:
                    rdata = json.load(fp)

                for each in rdata :
                    pkt_size= each['packet_length']
                    if len(pkt_size) < min_flow_len :
                        continue
                    x = feature_extractor.feature_extract(pkt_size)
                    X.append(x)
                    y.append(label)
        X = np.array(X)
        _max =  np.array(min_max._max)
        _min =  np.array(min_max._min)
        #归一化
        X = (X - _min)/(_max - _min)
        X = X.tolist()

        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        X_test =  []
        y_test =  []
        for i in range(len(X)):
            r = self.rand.uniform(0,1)
            if r < self.splitrate:
                X_test.append(X[i])
                y_test.append(y[i])
            elif r < self.splitrate * (2 - self.splitrate) :
                X_valid.append(X[i])
                y_valid.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])
        os.makedirs(self.data,exist_ok=True)

        with open(self.data + 'X_train.pkl','wb') as fp:
            pickle.dump(X_train, fp)

        with open(self.data + 'y_train.pkl','wb') as fp:
            pickle.dump(y_train,fp)

        with open(self.data + 'X_valid.pkl', 'wb') as fp:
            pickle.dump(X_valid,fp)

        with open(self.data + 'y_valid.pkl', 'wb') as fp:
            pickle.dump(y_valid, fp)

        with open(self.data + 'X_test.pkl', 'wb') as fp :
            pickle.dump(X_test, fp)

        with open(self.data + 'y_test.pkl' ,'wb') as fp:
            pickle.dump(y_test, fp)

    def load_data(self):
        with open(self.data + 'X_train.pkl','rb') as fp:
            X_train = pickle.load(fp)

        with open(self.data + 'y_train.pkl','rb') as fp:
            y_train = pickle.load(fp)

        with open(self.data + 'X_valid.pkl','rb') as fp:
            X_valid = pickle.load(fp)

        with open(self.data + 'y_valid.pkl','rb') as fp:
            y_valid = pickle.load(fp)

        with open(self.data + 'X_test.pkl','rb') as fp :
            X_test = pickle.load(fp)

        with open(self.data + 'y_test.pkl','rb') as fp:
            y_test = pickle.load(fp)

        return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test), np.array(y_test)

    def train(self):
        X_train, y_train, X_valid, y_valid, X_test, y_test =  self.load_data()
        lgb_train = lgb.Dataset(data=X_train,label=y_train)
        lgb_eval = lgb.Dataset(data=X_valid,label=y_valid)

        hyper_params['num_class'] = self.num_classes()
        gbm = lgb.train(params=hyper_params,
                        train_set=lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=50,
                        early_stopping_rounds=5)
        #save model
        try:
            gbm.save_model(self.model)
        except BaseException as exp:
            pass
        logit = gbm.predict(data=X_test)
        label_predict = list(map(lambda x : np.argmax(x),logit))

        accuracy = accuracy_score(y_test,label_predict)
        print('[Appscanner Test on {0} accuracy:{1}]'.format(self.dataset,accuracy))

    def test(self):
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data()
        #load model
        try:
            gbm = lgb.Booster(model_file= self.model)
        except BaseException as exp:
            raise exp
        logit = gbm.predict(data=X_test)
        label_predict = list(map(lambda x : np.argmax(x),logit))

        accuracy = accuracy_score(y_test,label_predict)
        report = classification_report(y_true=y_test,y_pred=label_predict)

        print("[Appscanner] Test on {0}, accuracy is {1}. ".format(self.dataset,accuracy))
        print(report)

if __name__ == '__main__':
    appscanner = model('website113', 128, 0.1)
    #appscanner.parser_raw_data()
    appscanner.train()
    appscanner.test()
