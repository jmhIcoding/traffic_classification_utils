__author__ = 'dk'
from models.model_base import abs_model
from config import raw_dataset_base, min_flow_len
import os, pickle, tqdm, json
import torch as th
from torch import optim
from models.dl.mimetic.build_model import MIMETICModel
from  torch.nn import functional as F
from models.dl.mimetic import model_seriealization
import  random, numpy as np
from sklearn.metrics import classification_report

def payload_hex2float(payload, payload_sz):
    rst = []
    for i in range(0, len(payload),2):
        if len(rst) > payload_sz:
            break

        _hex = payload[i:i+2]
        rst.append(int(_hex, base=16)/ 256.0)
    rst = rst + [0.0] * (payload_sz - len(rst))
    return rst[:payload_sz]

class Dataloder:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.watch = 0
        self.epoch = 0
        assert len(self.X_data) == len(self.y_data)
        self.instance_nb = len(y_data)
    def next_batch(self, batch_size):
        fields = []
        payload = []
        y = []
        while len(y) < batch_size:
            field = np.array(self.X_data[self.watch][0]).T
            #print(field.shape)
            #raise BaseException('xxxx')
            fields.append(field)
            payload.append(self.X_data[self.watch][1])
            y.append(self.y_data[self.watch])
            self.watch = (self.watch + 1) % self.instance_nb
            if self.watch == 0 :
                self.epoch += 1
        return th.tensor(fields).float(), th.tensor(payload).float(), th.tensor(y)
class MIMETIC(abs_model):
    def __init__(self,
                 dataset,               ##数据集名称
                 payload_sz=600,        ##使用载荷的前几个字节
                 packet_nb=64,         ##包长序列的最大长度
                 splitrate=0.1          ##测试集的大小
        ):
        super(MIMETIC, self).__init__(name='MIMETIC', randseed=128)
        self.payload_size = payload_sz
        self.packet_max_number = packet_nb
        self.dataset = dataset
        self.splitrate=splitrate
        self.model = self.database + '/'+ self.name + '_' + dataset + '_model'
        self.data = self.database + '/'+ self.name + '_' + dataset + '/'

        #原始数据集目录
        full_rdata = raw_dataset_base + self.dataset
        self.full_rdata = full_rdata
        if self.data_exists() == False:
            self.parser_raw_data()

        self.mimetic_model = MIMETICModel(payload_sz= payload_sz,
                                        packet_nb=packet_nb,
                                        class_nb=self.num_classes()
            )
    def parser_raw_data(self):
        full_rdata = self.full_rdata
        if os.path.exists(full_rdata) == False:
            raise OSError('Dataset {0} (full path: {1}) does not exist!'.format(self.dataset,full_rdata))
        #从原始数据集目录构建app-net所需的数据形式
        #1. payload; 2. 包长
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
                    if 'payload' not in each:
                        raise ValueError('For each flow, must contain payload field')
                    if 'window_size' not in each:
                        raise ValueError('For each flow, must contain window_size field')
                    if 'arrive_time_delta' not in each:
                        raise ValueError('For each flow, must contain arrive_time_delta field')

                    pkt_size =  each['packet_length']
                    arrival_time_interval = each['arrive_time_delta']
                    win_size = each ['window_size'][: len(pkt_size)]

                    payload = "".join(each['payload']) ##TLS的载荷信息
                    if len(pkt_size) < min_flow_len :
                        continue

                    payload = payload_hex2float(payload, payload_sz= self.payload_size)

                    pkt_size = pkt_size + [0] * (self.packet_max_number - len(pkt_size))
                    pkt_size = pkt_size[: self.packet_max_number]
                    pkt_sign = np.sign(pkt_size).tolist()

                    arrival_time_interval = arrival_time_interval + [0] * (self.packet_max_number - len(arrival_time_interval))
                    arrival_time_interval = arrival_time_interval[: self.packet_max_number]

                    win_size = win_size + [0] * (self.packet_max_number - len(win_size))
                    win_size = win_size[: self.packet_max_number]

                    X.append([(pkt_size, pkt_sign, arrival_time_interval ), payload])
                    y.append(label)
        #打乱数据

        _indexs = [i for i in range(len(X))]
        random.shuffle(_indexs)
        X = np.array(X)[_indexs]
        y = np.array(y)[_indexs]

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

    def train(self, max_epochs=60, batch_size=256, lr = 4e-5, device = 'cuda:0'):
        _X_train, _y_train, _X_valid, _y_valid, _X_test, _y_test =  self.load_data()
        train_data = Dataloder(_X_train, _y_train)
        valid_data = Dataloder(_X_valid, _y_valid)

        loss = th.nn.CrossEntropyLoss().cuda(device=device)
        optimizer = optim.Adam(self.mimetic_model.parameters(), lr=1e-4)
        self.mimetic_model.cuda(device=device)
        lastacc = 0
        for epoch in tqdm.trange(max_epochs):
            while epoch== train_data.epoch:
                fields, payload, y = train_data.next_batch(batch_size=batch_size)
                fields = fields.to(th.device(device))
                payload = payload.to(th.device(device))
                y = y.to(th.device(device))

                logit = self.mimetic_model(fields, payload)
                err = loss(logit, y)
                optimizer.zero_grad()
                err.backward()
                optimizer.step()

            fields, payload, labels = valid_data.next_batch(batch_size=batch_size)
            fields = fields.to(th.device(device))
            payload = payload.to(th.device(device))
            labels = labels.to(th.device(device))
            logit = self.mimetic_model(fields, payload)
            argmax_labels = th.argmax( F.softmax(logit,1),1 )

            acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
            print('Epoch {0}, valid acc: {1}'.format(epoch, acc))
            if acc > lastacc:
                lastacc = acc
                model_seriealization.save(self.mimetic_model, model_path= self.model)

    def test(self, modelpath=None, batch_size=8, device='cuda:0'):
        if modelpath is None:
            modelpath = self.model

        self.mimetic_model = model_seriealization.load(self.mimetic_model, model_path= modelpath, use_gpu=True, device=device)
        self.mimetic_model = self.mimetic_model.cuda(device)
        _X_train, _y_train, _X_valid, _y_valid, _X_test, _y_test =  self.load_data()
        test_data = Dataloder(_X_test, _y_test)
        acc_list =[]
        y_true = []
        y_pred =[]
        while test_data.epoch == 0:
            fields, payload, labels = test_data.next_batch(batch_size=batch_size)
            fields = fields.to(th.device(device))
            payload = payload.to(th.device(device))
            labels = labels.to(th.device(device))
            logit = self.mimetic_model(fields, payload)
            argmax_labels = th.argmax( F.softmax(logit,1),1 )

            acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
            acc_list.append(acc)
            y_true += labels.tolist()
            y_pred += argmax_labels.tolist()

        import numpy as np
        print('Average Acc: {0}'.format(np.average(acc_list)))
        print(classification_report(y_true, y_pred, digits=5))
        
if __name__ == '__main__':
    model = MIMETIC(dataset='D5_payload')
    #model.train()
    #model.test()
    model.test('/home3/jmh/traffic_classification_utils/models/dl/mimetic/data/MIMETIC_D2_payload_model')
