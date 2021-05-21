__author__ = 'jmh081701'
#特征提取
############
import numpy as np
import  os
import  sys
import copy
from src.df.src.utility import LoadDataWakieTalkie_Single_DataSet
class CUMUL_datagenerator:

    def __init__(self,
                 feature_length=100,min=-2305,max=2305,
                 equidistance=None,cell_size=512,is_train=False):
        self.feature_length = feature_length        #cumul模型的输入向量的长度,默认是100
        self.cell_size = cell_size                  #Tor的cell的大小
        self.equidistance = None                    #采样的间距

        #标准化的参数
        self.min = min
        self.max = max
        ##############训练模型使用的数据

        self.is_train = is_train

        self.train_X = None
        self.train_y = None

        self.valid_X = None
        self.valid_y = None

        self.test_X = None
        self.test_y = None

        if is_train:
            self.load_tor_cell_sequence()

    def feature_extract(self,trace_sequence,cell_size=None):
        """feature_extract() : 从[-1,1,1...]的cell的方向序列中，生成CUMUL模型所需的特征向量

        :param trace_sequence: `numpy.narray` ,形状：batch_size * trace_length
                            输入的[-1,1,1...]向量,-1表示ingoing的cell,+1表示outgoing的流
        :param cell_size:   每个cell的大小,默认None,因为最后还得归一化
        :return:
        """
        if cell_size == None:
            cell_size = 1
        if not isinstance(type(trace_sequence),np.ndarray):
            trace_sequence = np.array(trace_sequence)
        shape = trace_sequence.shape
        culmulative_sum_a = np.zeros(shape=shape,dtype = np.float)
        culmulative_sum_c = np.zeros(shape=shape,dtype = np.float)
        xp = np.linspace(0,shape[1]-1,shape[1])
        features  = np.zeros(shape=(shape[0],2*self.feature_length),dtype = np.float)
        #计算累计和
        for i in range(0,shape[0]):
            for j in range(1,shape[1]):
                culmulative_sum_a[i,j] += culmulative_sum_a[i,j-1] + abs(trace_sequence[i,j])
                culmulative_sum_c[i,j] += culmulative_sum_c[i,j-1] + trace_sequence[i,j]
        #加上cell_size
        culmulative_sum_a = cell_size * culmulative_sum_a
        culmulative_sum_c = cell_size * culmulative_sum_c

        #线性采样n个特征
        if self.equidistance != None:
            equidistance = self.equidistance
        else:
            equidistance = (shape[1]-1)/self.feature_length
        xval = np.arange(0,equidistance * self.feature_length,equidistance)
        for i in range(shape[0]):
            #print(i,culmulative_sum_a[i])
            #print(i,culmulative_sum_c[i])
            a_interp = (np.interp(xval,xp,culmulative_sum_a[i])-self.min)/(self.max-self.min)
            c_interp = (np.interp(xval,xp,culmulative_sum_c[i])-self.min)/(self.max-self.min)

            features[i,0:2*self.feature_length:2]=copy.deepcopy(a_interp)
            features[i,1:2*self.feature_length:2]=copy.deepcopy(c_interp)
            #print(i,features[i])
            #print('#'*30)
        return  features

    def load_tor_cell_sequence(self):
        if not self.is_train :
            return

        _,__,self.train_X,self.train_y = LoadDataWakieTalkie_Single_DataSet('train',is_cluster=False,normalized=False)
        _,__,self.valid_X,self.valid_y = LoadDataWakieTalkie_Single_DataSet('valid',is_cluster=False,normalized=False)
        _,__,self.test_X,self.test_y = LoadDataWakieTalkie_Single_DataSet('test',is_cluster=False,normalized=False)

        self.train_X = self.feature_extract(self.train_X)
        print('feature extract....')
        self.valid_X =self.feature_extract(self.valid_X)
        self.test_X = self.feature_extract(self.test_X)

        print('Load tor cell sequence dataset well.')
        print('X train shape:',self.train_X.shape)
        print('y train shape:',self.train_y.shape)
        print('X valid shape:',self.valid_X.shape)
        print('y valid shape:',self.valid_y.shape)
        print('X test shape:',self.test_X.shape)
        print('y test shape:',self.test_y.shape)

    def trainSet(self):
        return self.train_X,self.train_y
    def validSet(self):
        return self.valid_X,self.valid_y
    def testSet(self):
        return self.test_X,self.test_y


if __name__ == '__main__':
    dator = CUMUL_datagenerator(is_train=True)
    print(dator.test_X[1],dator.test_y[1])
