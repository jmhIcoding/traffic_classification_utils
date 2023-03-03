__author__ = 'dk'
from models.dl.attacks import DF_model, parser_raw_data
from models.dl.df import df_model_config
from models.model_base import abs_model
import os
from config import raw_dataset_base
from keras.utils import np_utils
import numpy as np
os.environ['CUDA_VISBALE_DEIVCES'] ='cuda:2'
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

        self.df_model = None
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

    def predict(self,pkt_size):
        def pad_sequence(x, max_len, pad_value=0):
            r =  x + [pad_value] * (max_len - len(x))
            return r[:max_len]

        if self.df_model == None:
            self.df_model = DF_model(num_class= self.num_classes())
            self.df_model.load_model(self.model)

        x = [pad_sequence(_pkt_size, max_len= df_model_config.learning_params_template['in_dim']) for _pkt_size in pkt_size]
        x = np.array(x)[:, :,np.newaxis]
        y_logit = self.df_model.predict(x, actual_lable=True)
        return y_logit.tolist()
    def get_feature(self):
        X_train,y_train, X_valid, y_valid, X_test, y_test = self.load_data()
        #y_test = np_utils.to_categorical(y_test, num_classes= self.num_classes())
        X_test  = X_test[:5000]
        X_test  = X_test[:, :,np.newaxis]

        df_model = DF_model(num_class= self.num_classes())
        df_model.load_model(self.model)
        logit, feature = df_model.predict(X_test=X_test,actual_lable=False, return_feature=True)
        print(feature.shape, logit.shape)
        logit = logit.tolist()
        feature = feature.tolist()
        #feature = logit
        y_true = y_test[:5000].tolist()
        feature_set = {}
        feature_vector = []
        for i in range(len(y_true)):
           if y_true[i] not in feature_set:
              feature_set[y_true[i]] = []
           feature_set[y_true[i]].append([feature[i]])
        import pickle
        with open('feature_set_D1_53_DF.pkl','wb') as fp:
            pickle.dump(feature_set, fp)
        print(y_true[-1],logit[-1])
        print(feature[-1])		
if __name__ == '__main__':
  for test_rate in [0.1]:
    print(test_rate)
    dataset='app150'
    df_model = model(dataset, randseed= 128, splitrate=test_rate)
    #df_model.parser_raw_data()
    df_model.train()
    df_model.test()
    print(dataset)
    print(test_rate)
    #import os
    #os.remove(df_model.model)
    #df_model.get_feature()
    break
