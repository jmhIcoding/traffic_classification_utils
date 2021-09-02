__author__ = 'dk'
import random
import os
import pickle
import numpy as np
import gzip
class abs_model:
    def __init__(self, name, randseed):
        self.database = './data/'
        self.name = name
        self.rand = random.Random(x = randseed)
        self.data = None
        self.model = None
        self.full_rdata = []

    def data_exists(self):
        return  os.path.exists(self.data)
    def model_exist(self):
        return  os.path.exists(self.model)

    def train(self):
        pass

    def test(self):
        pass

    def parser_raw_data(self):
        ##从原始通用数据集获取自己所需格式数据集能力
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
    def save_data(self,X_train, y_train, X_valid, y_valid, X_test, y_test):
        fp = gzip.GzipFile(self.data + 'data.gzip','wb')
        pickle.dump({
            'X_train':X_train,
            'y_train':y_train,
            'X_valid':X_valid,
            'y_valid':y_valid,
            'X_test':X_test,
            'y_test':y_test
        },file=fp)
        fp.close()
    def load_data(self):
        fp = gzip.GzipFile(self.data + 'data.gzip','rb')
        data = pickle.load(fp)
        fp.close()
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_test = data['X_test']
        y_test = data['y_test']
        return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test), np.array(y_test)
    def num_classes(self):
        for _root, _dir, _files in os.walk(self.full_rdata):
            classes = _files
        return len(classes)

