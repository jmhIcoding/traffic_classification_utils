__author__ = 'dk'
from models.model_base import abs_model
from config import raw_dataset_base, min_flow_len
import os
import pickle,json
import tqdm
import numpy as np
from models.ml.bind import build_vector_dataset
import lightgbm as lgb
from sklearn.metrics import  accuracy_score,classification_report
from models.ml.bind.hyper_params import hyper_params
class model(abs_model):
    def __init__(self, dataset, randseed, splitrate, topK=1000):
        super(model, self).__init__('bind', randseed= randseed)
        self.topK = topK
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
    def build_raw_trace_for_bind(self,each_trace):
        feature = {
            'Dn-Up-size':{},
            'Dn-Up-time':{},
            'Up-Dn-size':{},
            'Up-Dn-time':{},
            'Uni-size':{},
            'Uni-time':{},
            'Pkt-size':{}
        }
        burst_size =[0]
        burst_time =[0]
        direction =  0
        for i in range(len(each_trace['packet_length'])):
            #包级别的长度特征
            if abs(each_trace['packet_length'][i]) not in feature['Pkt-size']:
                feature['Pkt-size'].setdefault(abs(each_trace['packet_length'][i]),1)
            else:
                feature['Pkt-size'][abs(each_trace['packet_length'][i])] += 1

            #Uni-burst的特征
            sign = np.sign(each_trace['packet_length'][i])
            if direction == 0:
                direction = sign
            if sign == direction:
                #同向的,使用原来的burst
                pass
            else:
                #方向相反,新增一个burst
                direction =sign
                burst_size.append(0)
                burst_time.append(0)
            burst_size[-1] += each_trace['packet_length'][i]            #累积包长
            burst_time[-1] += int(each_trace['arrive_time_delta'][i] * 100)/100        #累积通信时长,保留两位小数
        #构建burst级别的特征
        for index in range(len(burst_size)):
            #uni-burst
            if abs(burst_size[index]) not in feature['Uni-size']:
                feature['Uni-size'].setdefault(abs(burst_size[index]),1)
            else:
                feature['Uni-size'][abs(burst_size[index])] += 1
            if burst_time[index] not in feature['Uni-time']:
                feature['Uni-time'].setdefault(burst_time[index] ,1)
            else:
                feature['Uni-time'][burst_time[index] ] += 1
            #bi-burst
            if index > 0:
                bi_size = (abs(burst_size[index-1]),abs(burst_size[index]))
                bi_time =(burst_time[index-1],burst_time[index])
                if np.sign(burst_size[index-1])== -1:
                    #Dn-Up Burst
                    if bi_size not in feature['Dn-Up-size']:
                        feature['Dn-Up-size'].setdefault(bi_size,1)
                    else:
                        feature['Dn-Up-size'][bi_size] += 1
                    if bi_time not in feature['Dn-Up-time']:
                        feature['Dn-Up-time'].setdefault(bi_time,1)
                    else:
                        feature['Dn-Up-time'][bi_time] += 1
                elif np.sign(burst_size[index-1]) == 1:
                    #Up-Dn Burst:
                    if bi_size not in feature['Up-Dn-size']:
                        feature['Up-Dn-size'].setdefault(bi_size,1)
                    else:
                        feature['Up-Dn-size'][bi_size] += 1
                    if bi_time not in feature['Up-Dn-time']:
                        feature['Up-Dn-time'].setdefault(bi_time,1)
                    else:
                        feature['Up-Dn-time'][bi_time] += 1
                else:
                    print('Packet size:',each_trace['packet_length'])
                    print('Uni-burst size:',burst_size)
                    print('Uni-burst time:',burst_time)
                    print('index:',index)
                    raise ValueError('Burst Direction Error!')
        return  feature

    def parser_raw_data(self):
        full_rdata = self.full_rdata
        if os.path.exists(full_rdata) == False:
            raise OSError('Dataset {0} (full path: {1}) does not exist!'.format(self.dataset,full_rdata))
        os.makedirs(self.data,exist_ok=True)

        #从原始数据集目录构建appscanner所需的数据集
        X = []
        y = []
        ##构建raw_trace目录
        raw_traces = {}
        for _root, _dirs, _files in os.walk(full_rdata):
            for file in tqdm.trange(len(_files)):
                file = _files[file]
                label = file
                file = _root + '/' + file
                if label not in raw_traces:
                    raw_traces[label] = []
                with open(file) as fp:
                    rdata = json.load(fp)
                for each in rdata :
                    pkt_size= each['packet_length']
                    if len(pkt_size) < min_flow_len :
                        continue
                    x = self.build_raw_trace_for_bind(each_trace=each)
                    raw_traces[label].append(x)

                with open(self.data + label + '.bind','wb') as fp:
                    pickle.dump(raw_traces[label],fp)
        #dator = build_vector_dataset.builder(raw_feature_dictory= self.data)
        #X_train,y_train,X_test,y_test,X_valid,y_valid = dator.vectorize(test_split_ratio= self.splitrate, K= self.topK)
        #self.save_data(X_train,y_train,X_valid,y_valid,X_test,y_test)
    def train(self):
        dator = build_vector_dataset.builder(raw_feature_dictory= self.data)
        X_train,y_train ,X_valid,y_valid, X_test,y_test= dator.vectorize(test_split_ratio= self.splitrate, K= self.topK)

        hyper_params['num_class']= self.num_classes()
        ##开始训练
        lgb_train = lgb.Dataset(data=X_train,label=y_train)
        lgb_eval = lgb.Dataset(data=X_valid,label=y_valid)
        gbm = lgb.train(params=hyper_params,
                        train_set=lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=100,
                        early_stopping_rounds=5)
        #save model
        try:
            gbm.save_model(self.model)
        except BaseException as exp:
            pass
        logit = gbm.predict(data=X_test)
        label_predict = list(map(lambda x : np.argmax(x),logit))
        accuracy = accuracy_score(y_test,label_predict)
        print('[Bind Test on {0}, accuracy: {1}]'.format(self.dataset,accuracy))

    def test(self):
        dator = build_vector_dataset.builder(raw_feature_dictory= self.data)
        X_train,y_train ,X_valid,y_valid, X_test,y_test= dator.vectorize(test_split_ratio= self.splitrate, K= self.topK)
        gbm = lgb.Booster(model_file= self.model)
        logit = gbm.predict(data=X_test)
        label_predict = list(map(lambda x : np.argmax(x),logit))

        accuracy = accuracy_score(y_test,label_predict)
        print('[Bind Test on {0}, accuracy: {1}]'.format(self.dataset,accuracy))
        report = classification_report(y_true=y_test,y_pred=label_predict)
        print(report)

if __name__ == '__main__':
    bind = model('website113', randseed= 128, splitrate= 0.1,topK=500)
    bind.train()
    bind.test()