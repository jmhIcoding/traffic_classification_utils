__author__ = 'dk'
from models.dl.attacks import DF_model, parser_raw_data
from models.dl.df import df_model_config
from models.model_base import abs_model
import os
from config import raw_dataset_base
from keras.utils import np_utils
import numpy as np
class model(abs_model):
    def __init__(self, dataset, randseed, splitrate):
        super(model,self).__init__('df',randseed= randseed)
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
        os.makedirs(self.data, exist_ok=True)
        ##从原始数据集构建DF所需的数据集
        X_train,y_train, X_valid, y_valid, X_test, y_test = parser_raw_data(self, self.full_rdata, max_len = df_model_config.learning_params_template['in_dim'])

        self.save_data(X_train,y_train, X_valid, y_valid, X_test, y_test)


    def train(self):
        X_train,y_train, X_valid, y_valid, X_test, y_test = self.load_data()
        num_class = self.num_classes()
        df_model_config.nb_classes_template = num_class

        y_train = np_utils.to_categorical(y_train, num_classes=num_class)
        y_valid = np_utils.to_categorical(y_valid, num_classes=num_class)
        y_test = np_utils.to_categorical(y_test, num_classes= self.num_classes())

        X_train = X_train[:, :,np.newaxis]
        X_valid = X_valid[:, :,np.newaxis]
        X_test  = X_test[:, :,np.newaxis]

        df_model = DF_model(num_class = num_class)
        df_model.build_model()

        df_model.fit(X_train=X_train,y_train=y_train,
                     X_valid= X_valid, y_valid = y_valid,
                     batch_size= df_model_config.learning_params_template['batch_size'],
                     epochs=df_model_config.learning_params_template['epoch'])

        df_model.save_model(path=self.model)
        score = df_model.evaluate(X_test=X_test, y_test = y_test)
        print('[Deep Fingerprinting Test on {0} accuracy {1}'.format(self.dataset, score))
    def test(self):
        X_train,y_train, X_valid, y_valid, X_test, y_test = self.load_data()
        y_test = np_utils.to_categorical(y_test, num_classes= self.num_classes())
        X_test  = X_test[:, :,np.newaxis]

        df_model = DF_model(num_class= self.num_classes())
        df_model.load_model(self.model)
        score = df_model.evaluate(X_test=X_test,y_test=y_test)
        print('Deep Fingerprinting Test on {0} accuracy :{1}'.format(self.dataset,score))

if __name__ == '__main__':
    df_model = model('cloud', randseed= 128, splitrate=0.1)

    df_model.train()
    df_model.test()
