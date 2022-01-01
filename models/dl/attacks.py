__author__ = 'dk'
import  os
#设置Tensorflow的日志等级
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

from abc import abstractmethod
import keras
try:
    from  tensorflow import Graph,Session
except:
    pass
import  numpy as np
from keras import backend as K
from models.dl.select_gpu import set_visible_gpu
from models.dl.cnn import cnn_model
from models.dl.df import df_model
from models.dl.lstm import lstm_model,lstm_model_config
from models.dl.sdae import  sdae_model
from models.dl.accuracy_per_class import accuracy_per_class
from models.dl.beauty import cnn_model as beauty_model
import tqdm,json
from config import min_flow_len
#自动选择空闲内存最大的显卡
set_visible_gpu()

class attack_base:
    def __init__(self):
        self.path_prefix = os.path.realpath(os.path.curdir)
        self.model_path = None
        self.model = None

        #define tensorflow Graph and session objector for this model. 支持同一进程空间下多个graph同时运行
        self.graph = Graph()
        self.session = Session(graph=self.graph)
        keras.backend.set_session(self.session)
        #keras.backend.set_learning_phase(0)
    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def load_model(self,path=None):
        if path == None:
            path = self.model_path
        if os.path.exists(path):
            print('Load {0} model from pre-trained model file {1}.'.format(self.model_name(),path))
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = keras.models.load_model(path)
        else:
            print('{0} model cant load historical model file from {0}'.format(self.model_name(),path))

    def save_model(self,path=None):
        if path == None:
            path = self.model_path
        if os.path.exists(path):
            print('Overwrite {0} model on {1} now.'.format(self.model_name(),path))
        else:
            os.makedirs(os.path.dirname(path),exist_ok=True)
            print('save {0} model on {1} now.'.format(self.model_name(),path))
        with self.session.as_default():
            with self.graph.as_default():
                keras.models.save_model(self.model,path,overwrite=True)

    def fit(self,X_train,y_train,X_valid,y_valid,batch_size,epochs,verbose=2):
        ''' 训练模型
            :param X_train:         X_train 训练数据的向量部分 shape: [样本个数,特征维度1,特征维度2...]
            :param y_train:         y_train 训练数据的标签部分,注意它得是one-hot的向量   shape: [样本个数,标签个数]
            :param X_valid:         X_valid 验证集数据的向量部分 shape: [样本个数,特征维度1,特征维度2...]
            :param y_valid:         y_valid 验证集数据的标签部分,注意它得是one-hot的向量   shape: [样本个数,标签个数]
            :param batch_size:      训练的batch_size
            :param epochs:          训练的迭代次数
            :param verbose:         训练输出的结果
            :return:                无
        '''
        if self.model_name() == "sdae_model" :
            if len(X_train.shape) > 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
            if len(X_valid.shape) > 2:
                X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1])

        with self.session.as_default():
            with self.graph.as_default():
                self.model.fit(X_train,y_train,
                               validation_data=(X_valid,y_valid),
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose)

    def fit_generator(self,train_gen,batch_size,epochs,val_gen,verbose=2):
        pass

    def evaluate(self,X_test,y_test,verbose=2):
        '''测试一波数据,并计算准确率
            :param X_test : 待测试数据 shape: [样本个数,特征维度1,特征维度2...]
            :param y_test : 待测试数据的真实标签,one-hot数据 shape: [样本个数,标签个数]
            :param verbose: 模型输出的提示信息
            :return  直接返回准确率,是个float数值
        '''
        if self.model_name() == "sdae_model" :
            if len(X_test.shape) > 2:
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        with self.session.as_default():
            with self.graph.as_default():
                score_test = self.model.evaluate(X_test,y_test,verbose=verbose)
                y_pred=self.predict(X_test,actual_lable=True)
                accuracy_per_class(y_real=np.argmax(y_test,1),y_pred=y_pred)
                return score_test[1]
    def get_feature_map(self,X,layer_name='block1_conv1'):
        ''' 获取特定中间层的特征图

        :param X:           输入数据
        :param layer_name:  层的名字,str
        :return:
        '''
        with self.session.as_default():
            with self.graph.as_default():
                layer =   self.model.get_layer(layer_name)
                if layer != None:
                    value = layer.output
                    get_value= K.function(inputs=self.model.inputs,outputs=[value])
                    output = get_value([X])[0]
                    return  output
                else:
                    raise ValueError('Model{0} could not find layer named {1}.'.format(self.model_name,layer_name))
    def predict(self,X_test,actual_lable=False, return_feature=False, verbose=2):
        ''' 预测标签
        :param X_test:  待测数据 ,shape: [样本个数,特征维度1,特征维度2...]
        :param verbose:
        :return:  各个样本的概率分布向量 shape:[batch_size,label_dimension]
        '''
        if self.model_name() == "sdae_model" :
            if len(X_test.shape) > 2:
                X_test = X_train.reshape(X_test.shape[0], X_test.shape[1])
        with self.session.as_default():
            with self.graph.as_default():
                  prob = self.model.predict(X_test,verbose=2)
        if return_feature == True:
           features = self.get_feature_map(X_test, 'fc1')
        if actual_lable== False:
        	#返回概率分布
            if return_feature == False:
               return prob
            else:
               return prob, features
        else:
        	#返回真实的标签,每个标签是个一维的数字
            labels =  np.argmax(prob,axis=1)
            if return_feature == False:
               return  labels
            else:
               return  labels, features

class CNN_model(attack_base):
    def __init__(self, num_class):
        super(CNN_model,self).__init__()
        self.model_path = self.path_prefix + "/cnn/saved_model/cnn_model.h5"
        self.num_class = num_class
    def model_name(self):
        return "cnn_model"
    def build_model(self):
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = cnn_model.build_model(nb_classes= self.num_class)
class Beauty_model(attack_base):
    def __init__(self, num_class):
        super(Beauty_model,self).__init__()
        self.model_path = self.path_prefix + "/cnn/saved_model/beauty_model.h5"
        self.num_class = num_class
    def model_name(self):
        return "beauty_model"
    def build_model(self):
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = beauty_model.build_model(nb_classes= self.num_class)


class DF_model(attack_base):
    def __init__(self, num_class):
        super(DF_model,self).__init__()
        self.model_path = self.path_prefix + "/df/saved_model/df_model.h5"
        self.num_class = num_class
    def model_name(self):
        return "df_model"
    def build_model(self):
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = df_model.build_model(classes= self.num_class)
class LSTM_model(attack_base):
    def __init__(self):
        super(LSTM_model,self).__init__()
        self.model_path = self.path_prefix + "/lstm/saved_model/lstm_model.h5"

    def model_name(self):
        return "lstm_model"
    def build_model(self):
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = lstm_model.build_model()

class SDAE_model(attack_base):
    def __init__(self):
        super(SDAE_model,self).__init__()
        self.model_path = self.path_prefix + "/sdae/saved_model/sdae_model.h5"
        self.pretrained = False
    def model_name(self):
        return "sdae_model"
    def build_model(self):
        if os.path.exists(self.model_path):
            self.load_model()
            self.pretrained = True
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = sdae_model.build_model()

    def pre_train(self,x_train,x_test):
            if self.pretrained == True:
                print("{0} pretrained well!".format(self.model_name()))
                return
            if len(x_train.shape) > 2:
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
            if len(x_test.shape) > 2:
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
            with self.session.as_default():
                with self.graph.as_default():
                    self.model = sdae_model.pre_train(self.model,x_train=x_train,x_test=x_test)


def burstification_direction_operator(x):
    rst = [x[0]]
    for i in range(1, len(x)):
        if int(x[i]) == 0:
            break
        if np.sign(rst[-1]) == np.sign(x[i]):
            rst[-1]+= x[i]
        else:
            rst.append(x[i])
    return  rst
def burstification_time_operator(x, pkt_time, threshold=1e-5):
    rst = [x[0]]
    for i in range(1,len(x)):
        if int(x[i])== 0 :
            break
        if abs(pkt_time[i]) < threshold and np.sign(rst[-1]) == np.sign(x[i]):
            rst[-1] += x[i]
        else:
            rst.append(x[i])
    return rst
def parser_raw_data(self,path, max_len, burstification = False, burst_direction= True):
    def pad_sequence(x, max_len=max_len, pad_value=0):
        r =  x + [pad_value] * (max_len - len(x))
        return r[:max_len]
    X = []
    y = []
    for _root, _dirs, _files in os.walk(path):
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

                if burstification == True:
                    pkt_time= each['arrive_time_delta']
                    if burst_direction == True:
                        pkt_size = burstification_direction_operator(pkt_size)
                    else:
                        pkt_size = burstification_time_operator(pkt_size, pkt_time=pkt_time)

                x = pad_sequence(pkt_size)
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
            X_valid.append(X[i])
            y_valid.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return X_train,y_train, X_valid, y_valid, X_test, y_test

if __name__ == '__main__':
    x= [-199,2000,1,1,1,1,-1,-1,-2]
    print(burstification_operator(x))
