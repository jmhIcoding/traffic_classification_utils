__author__ = 'dk'
from models.model_base import abs_model
import os
import shutil
import json
from config import raw_dataset_base, min_flow_len, whitelist
import numpy as np
import pickle
import  lightgbm as lgb
import tqdm
import random
from sklearn.metrics import  accuracy_score,classification_report
from models.ml.rdp.rdp_config import hyper_params
from models.ml.rdp.feature_extractor import feature_extract
class model(abs_model):
    def __init__(self, dataset, randseed, splitrate):
        super(model,self).__init__('rdp',randseed= randseed)
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
        else:
            self.load_data()


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
                    if 'packet_length' not in each:
                        
                        raise  ValueError('For each flow, must contain packet length field.')
                    if 'arrive_time_delta' not in each:
                        each['arrive_time_delta'] = [0.0] * len(each['packet_length'])
			#raise ValueError('For each flow, must contain arrive_time_delta field')

                    pkt_size= each['packet_length']
                    timestamp = each['arrive_time_delta']

                    if len(pkt_size) < min_flow_len :
                        continue
                    
                    x = feature_extract(pkt_size=pkt_size, timestamps= timestamp)  ## 一条流的所有burst 特征
                    X.append(x)
                    y.append(label)
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

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def train(self, num_boost_round = 100):
        _X_train, _y_train, _X_valid, _y_valid, _X_test, _y_test =  self.load_data()
        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        X_test = []
        y_test = []
        print(len(_X_train), len(_y_train))
        print(_X_train[0])
        for i in range(len(_X_train)):
            X_train.append( _X_train[i])
            y_train += [_y_train[i]] * len(_X_train[i])
        print(len(X_train), len(X_train[0]),len(y_train))
        for i in range(len(_X_valid)):
            X_valid += _X_valid[i]
            y_valid += [_y_valid[i]] * len(_X_valid[i])

        for i in range(len(_X_test)):
            X_test += _X_test[i]
            y_test += [_y_test[i]] * len(_X_test[i])

        #打乱数据
        _indexs = [i for i in range(len(X_train))]
        random.shuffle(_indexs)
        X_train = np.array(X_train)[_indexs]
        y_train = np.array(y_train)[_indexs]

        _indexs = [i for i in range(len(X_valid))]
        random.shuffle(_indexs)
        X_valid = np.array(X_valid)[_indexs]
        y_valid = np.array(y_valid)[_indexs]

        _indexs = [i for i in range(len(X_test))]
        random.shuffle(_indexs)

        X_test = np.array(X_test)[_indexs]
        y_test = np.array(y_test)[_indexs]


        print('Train X.shape:', X_train.shape)
        print('Train y.shape:', y_train.shape)
        lgb_train = lgb.Dataset(data=X_train,label=y_train)
        lgb_eval = lgb.Dataset(data=X_valid,label=y_valid)

        hyper_params['num_class'] = self.num_classes()
        gbm = lgb.train(params=hyper_params,
                        train_set=lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=num_boost_round ,
                        early_stopping_rounds=5)
        #save model
        try:
            gbm.save_model(self.model)
        except BaseException as exp:
            pass
        logit = gbm.predict(data=X_test)
        label_predict = list(map(lambda x : np.argmax(x),logit))

        accuracy = accuracy_score(y_test,label_predict)

        print('[RDP Test on {0} accuracy:{1} (single burst)]'.format(self.dataset,accuracy))
        report = classification_report(y_true=y_test,y_pred=label_predict,digits=5)
        print(report)

    def test(self):
        X_train, y_train, X_valid, y_valid, _X_test, _y_test = self.load_data()
        #load model
        try:
            gbm = lgb.Booster(model_file= self.model)
        except BaseException as exp:
            raise exp

        X_test = []
        y_test = []
        y_test_single_burst =[]
        sample_index_map= {}
        burst_number = 0
        for i in range(len(_X_test)):
            if len(_X_test[i]) == 0:
                continue
            X_test += _X_test[i]
            y_test += [_y_test[i]]
            y_test_single_burst += [_y_test[i]] * len(_X_test[i])
            burst_number +=  len(_X_test[i])
            sample_index_map[i] = range(burst_number - len(_X_test[i]), burst_number)
        X_test = np.array(X_test)

        logit = gbm.predict(data=X_test)
        label_predict = np.array(list(map(lambda x : np.argmax(x), logit)))
        print("[RDP] Test on {0}, accuracy is {1}. (single burst) ".format(self.dataset,accuracy_score(y_test_single_burst, label_predict)))
        ##投票
        def vote_majority(votes):
            rst = {}
            for each in votes:
                if each not in rst:
                    rst[each] = 0
                rst[each] += 1
            rst = list(rst.items())
            rst.sort(key= lambda x: x[1])
            return rst[-1][0]

        _label_predict =[]
        for sample in sample_index_map:
            votes = label_predict[sample_index_map[sample]]
            final_label = vote_majority(votes)
            _label_predict.append(final_label)

        accuracy = accuracy_score(y_test,_label_predict)
        report = classification_report(y_true=y_test,y_pred=_label_predict,digits=5)

        print("[RDP] Test on {0}, accuracy is {1}. ".format(self.dataset,accuracy))
        print(report)
if __name__ == '__main__':
    rdp = model('app53', 128, 0.1)
    #rdp.parser_raw_data()
    rdp.train(num_boost_round=100)
    rdp.test()
