__author__ = 'dk'
from models.model_base import abs_model
from config import raw_dataset_base, min_flow_len
import os, pickle, tqdm, json
import torch as th
from torch import optim
from models.dl.appnet.build_model import AppNetModel
from  torch.nn import functional as F
from models.dl.appnet import model_seriealization
from sklearn.metrics import classification_report

def payload_hex2int(payload, payload_sz):
    rst = []
    for i in range(0, len(payload),2):
        if len(rst) > payload_sz:
            break

        _hex = payload[i:i+2]
        rst.append(int(_hex, base=16))
    rst = rst + [0] * (payload_sz - len(rst))
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
        packet_size = []
        payload = []
        y = []
        while len(y) < batch_size:
            packet_size.append(self.X_data[self.watch][0])
            payload.append(self.X_data[self.watch][1])
            y.append(self.y_data[self.watch])
            self.watch = (self.watch + 1) % self.instance_nb
            if self.watch == 0 :
                self.epoch += 1
        return th.tensor(packet_size).long(), th.tensor(payload).long(), th.tensor(y)
class AppNet(abs_model):
    def __init__(self,
                 dataset,               ##数据集名称
                 payload_sz=1014,      ##使用client hello的前几个字节
                 payload_embed_sz=256, ##字节embedding后向量的大小
                 packet_nb=20,         ##包长序列的最大长度
                 packet_embed_sz=128,  ##包长的embedding后的向量的大小
                 splitrate=0.1          ##测试集的大小
        ):
        super(AppNet, self).__init__(name='AppNet', randseed=128)
        self.payload_size = payload_sz
        self.payload_embedded_size = payload_embed_sz

        self.packet_max_number = packet_nb
        self.packet_embedded_size= packet_embed_sz

        self.dataset = dataset
        self.splitrate=splitrate
        self.model = self.database + '/'+ self.name + '_' + dataset + '_model'
        self.data = self.database + '/'+ self.name + '_' + dataset + '/'

        #原始数据集目录
        full_rdata = raw_dataset_base + self.dataset
        self.full_rdata = full_rdata
        if self.data_exists() == False:
            self.parser_raw_data()

        self.appnet_model = AppNetModel(payload_sz= payload_sz,
                                        payload_embed_sz = payload_embed_sz,
                                        packet_nb=packet_nb,
                                        packet_embed_sz=packet_embed_sz,
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

                    pkt_size= [abs(x+1500) % 3000 for x in each['packet_length']]
                    payload = each['payload'][0]  ##TLS的client-hello的载荷信息
                    if len(pkt_size) < min_flow_len :
                        continue

                    payload = payload_hex2int(payload, payload_sz= self.payload_size)

                    pkt_size = pkt_size + [0] * (self.packet_max_number - len(pkt_size))
                    pkt_size = pkt_size[: self.packet_max_number]
                    X.append([pkt_size, payload])
                    y.append(label)
        #打乱数据
        import  random, numpy as np
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
        test_data = Dataloder(_X_test, _y_test)

        loss = th.nn.CrossEntropyLoss().cuda(device=device)
        optimizer = optim.Adam(self.appnet_model.parameters(), lr=1e-4)
        self.appnet_model.cuda(device=device)
        lastacc = 0
        for epoch in tqdm.trange(max_epochs):
            while epoch== train_data.epoch:
                pkt_size, payload, y = train_data.next_batch(batch_size=batch_size)
                pkt_size = pkt_size.to(th.device(device))
                payload = payload.to(th.device(device))
                y = y.to(th.device(device))

                logit = self.appnet_model(pkt_size, payload)
                err = loss(logit, y)
                optimizer.zero_grad()
                err.backward()
                optimizer.step()

            pkt_size, payload, labels = valid_data.next_batch(batch_size=batch_size)
            pkt_size = pkt_size.to(th.device(device))
            payload = payload.to(th.device(device))
            labels = labels.to(th.device(device))
            logit = self.appnet_model(pkt_size, payload)
            argmax_labels = th.argmax( F.softmax(logit,1),1 )

            acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
            print('Epoch {0}, valid acc: {1}'.format(epoch, acc))
            if acc > lastacc:
                lastacc = acc
                model_seriealization.save(self.appnet_model, model_path= self.model)

    def test(self, modelpath=None, batch_size=8, device='cuda:0'):
        if modelpath is None:
            modelpath = self.model

        self.appnet_model = model_seriealization.load(self.appnet_model, model_path= modelpath, use_gpu=True, device=device)
        self.appnet_model = self.appnet_model.cuda(device)
        _X_train, _y_train, _X_valid, _y_valid, _X_test, _y_test =  self.load_data()
        test_data = Dataloder(_X_test, _y_test)
        acc_list =[]
        y_true =[]
        y_pred = []
        while test_data.epoch == 0:
            pkt_size, payload, labels = test_data.next_batch(batch_size=batch_size)
            pkt_size = pkt_size.to(th.device(device))
            payload = payload.to(th.device(device))
            labels = labels.to(th.device(device))
            logit = self.appnet_model(pkt_size, payload)
            argmax_labels = th.argmax( F.softmax(logit,1),1 )

            acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
            y_true+= labels.tolist()
            y_pred+= argmax_labels.tolist()
            acc_list.append(acc)
        import numpy as np
        print('Average Acc: {0}'.format(np.average(acc_list)))
        print(classification_report(y_true=y_true, y_pred = y_pred, digits=5))


if __name__ == '__main__':
    appnet = AppNet(dataset='D4_payload')
    #appnet.train()
    #appnet.test()
    appnet.test(r'/home3/jmh/traffic_classification_utils/models/dl/appnet/data/AppNet_D5_payload_model')
