__author__ = 'dk'

from models.model_base import abs_model
from config import raw_dataset_base
from models.dl.graphDapp.data_builder import Dataset
from models.dl.graphDapp.train import main as train_main
from models.dl.graphDapp.test import main as test_main

import os
class model(abs_model):
    def __init__(self, dataset, randseed, splitrate ,max_len=200):
        super(model,self).__init__('graphDapp',randseed= randseed)
        if os.path.exists(self.database) == False:
            os.makedirs(self.database,exist_ok=True)

        self.dataset = dataset
        self.model = self.database + '/'+ self.name + '_' + dataset + '_model'
        self.data = self.database + '/'+ self.name + '_' + dataset + '/'
        self.splitrate = splitrate
        #原始数据集目录
        full_rdata = raw_dataset_base + self.dataset
        self.full_rdata = full_rdata
        self.max_len = max_len
        if self.data_exists() == False:
            self.parser_raw_data()
    def parser_raw_data(self):
        def pad_sequence(x, max_len= self.max_len, pad_value=0):
            r =  x + [pad_value] * (max_len - len(x))
            return r[:max_len]
        full_rdata = self.full_rdata
        if os.path.exists(full_rdata) == False:
            raise OSError('Dataset {0} (full path: {1}) does not exist!'.format(self.dataset,full_rdata))
        os.makedirs(self.data, exist_ok=True)
        ##从原始数据集构建graphDApp所需数据
        dator = Dataset(raw_dir=full_rdata,
                        dumpfile=self.data + '{0}.gzip'.format(self.dataset),
                        split_rate=self.splitrate,
                        renew= True)
        dator.save_dumpfile()

    def train(self):
        train_main(dataset_name=self.data + '{0}.gzip'.format(self.dataset), modelpath= self.model)

    def test(self):
        test_main(dataset_name=self.data + '{0}.gzip'.format(self.dataset), modelpath= self.model)

if __name__ == '__main__':
    graphdapp_model = model('awf200_burst', randseed= 128, splitrate=0.1)
    graphdapp_model.parser_raw_data()
    graphdapp_model.train()
    graphdapp_model.test()
