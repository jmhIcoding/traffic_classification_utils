__author__ = 'dk'
import numpy as np
import  os
import  sys
import copy
class cumul_feature_extractor:
    def __init__(self,
                 feature_length = 100,
                 min = 0, max = 1
                 ):
        self.feature_length = feature_length        #cumul模型的输入向量的长度,默认是100
        self.equidistance = None                    #采样的间距
        #标准化的参数
        self.min = min
        self.max = max
        ##############训练模型使用的数据

    def feature_extract(self,trace_sequence, cell_size=None):
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

            features[i,0:2*self.feature_length:2]=copy.deepcopy(a_interp)[:self.feature_length]
            features[i,1:2*self.feature_length:2]=copy.deepcopy(c_interp)[:self.feature_length]
            #print(i,features[i])
            #print('#'*30)
        return  features
