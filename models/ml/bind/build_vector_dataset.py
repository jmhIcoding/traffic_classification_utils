__author__ = 'dk'
#从raw_dataset,构建向量化后的数据集
import pickle
import os
import numpy as np
import random

class builder:
    def __init__(self,raw_feature_dictory,global_feature_dict_filename=None):
        self.rand = random.Random(x=128)

        self.raw_feature_dictory= raw_feature_dictory
        if global_feature_dict_filename == None:
            global_feature_dict_filename = raw_feature_dictory + '/global_feature_dict.vocb'
        if os.path.exists(global_feature_dict_filename)==False:
            #构建一个字典
            self.global_feature_dict ={
                        'Dn-Up-size':{},
                        'Dn-Up-time':{},
                        'Up-Dn-size':{},
                        'Up-Dn-time':{},
                        'Uni-size':{},
                        'Uni-time':{},
                        'Pkt-size':{},
                        'Label':{}
                    }
            for _root,_dirs,_files in os.walk(raw_feature_dictory):
                for file in _files:
                    print(file)
                    if not file.split('.')[-1]=='bind':
                        continue
                    packageName = file.split('.bind')[0]
                    file = (_root + file).replace("\\","/")
                    if packageName not in self.global_feature_dict['Label']:
                        self.global_feature_dict['Label'].setdefault(packageName,len(self.global_feature_dict['Label']))

                    with open(file,'rb') as fp:
                        raw_traces = pickle.load(fp)
                    for trace in raw_traces:
                        for key in trace['Dn-Up-size']:
                            if key not in self.global_feature_dict['Dn-Up-size']:
                                self.global_feature_dict['Dn-Up-size'].setdefault(key,trace['Dn-Up-size'][key])
                            else:
                                self.global_feature_dict['Dn-Up-size'][key] += trace['Dn-Up-size'][key]

                        for key in trace['Dn-Up-time']:
                            if key not in self.global_feature_dict['Dn-Up-time']:
                                self.global_feature_dict['Dn-Up-time'].setdefault(key,trace['Dn-Up-time'][key])
                            else:
                                self.global_feature_dict['Dn-Up-time'][key] += trace['Dn-Up-time'][key]

                        for key in trace['Up-Dn-size']:
                            if key not in self.global_feature_dict['Up-Dn-size']:
                                self.global_feature_dict['Up-Dn-size'].setdefault(key,trace['Up-Dn-size'][key])
                            else:
                                self.global_feature_dict['Up-Dn-size'][key] += trace['Up-Dn-size'][key]

                        for key in trace['Up-Dn-time']:
                            if key not in self.global_feature_dict['Up-Dn-time']:
                                self.global_feature_dict['Up-Dn-time'].setdefault(key,trace['Up-Dn-time'][key])
                            else:
                                self.global_feature_dict['Up-Dn-time'][key] += trace['Up-Dn-time'][key]

                        for key in trace['Uni-size']:
                            if key not in self.global_feature_dict['Uni-size']:
                                self.global_feature_dict['Uni-size'].setdefault(key,trace['Uni-size'][key])
                            else:
                                self.global_feature_dict['Uni-size'][key] += trace['Uni-size'][key]

                        for key in trace['Uni-time']:
                            if key not in self.global_feature_dict['Uni-time']:
                                self.global_feature_dict['Uni-time'].setdefault(key,trace['Uni-time'][key])
                            else:
                                self.global_feature_dict['Uni-time'][key] += trace['Uni-time'][key]

                        for key in trace['Pkt-size']:
                            if key not in self.global_feature_dict['Pkt-size']:
                                self.global_feature_dict['Pkt-size'].setdefault(key,trace['Pkt-size'][key])
                            else:
                                self.global_feature_dict['Pkt-size'][key] += trace['Pkt-size'][key]
            with open(raw_feature_dictory+"/" + 'global_feature_dict.vocb','wb') as fp:
                pickle.dump(self.global_feature_dict,fp)
        else:
            with open(global_feature_dict_filename,'rb') as fp:
                self.global_feature_dict = pickle.load(fp)

    def topK_key(self,dict,K):
        rst_list= [item[0] for item in sorted(dict.items(),key = lambda  item :item[1])[-K:]]
        rst ={}
        for each in rst_list:
            rst.setdefault(each,len(rst))
        return  rst
    def filter(self,K=1000):
        #只保留字典里面TopK的数据
        self.global_feature_dict['Dn-Up-size'] =self.topK_key(self.global_feature_dict['Dn-Up-size'],K)
        self.global_feature_dict['Dn-Up-time'] =self.topK_key(self.global_feature_dict['Dn-Up-time'],K)
        self.global_feature_dict['Up-Dn-size'] =self.topK_key(self.global_feature_dict['Up-Dn-size'],K)
        self.global_feature_dict['Up-Dn-time'] =self.topK_key(self.global_feature_dict['Up-Dn-time'],K)

        self.global_feature_dict['Uni-size'] =self.topK_key(self.global_feature_dict['Uni-size'],K)
        self.global_feature_dict['Uni-time'] =self.topK_key(self.global_feature_dict['Uni-time'],K)
        self.global_feature_dict['Pkt-size'] =self.topK_key(self.global_feature_dict['Pkt-size'],K)



    def summary(self):
        print('Unique Dn-Up-size:',len(self.global_feature_dict['Dn-Up-size']))
        print('Unique Dn-Up-time:',len(self.global_feature_dict['Dn-Up-time']))
        print('Unique Up-Dn-size:',len(self.global_feature_dict['Up-Dn-size']))
        print('Unique Up-Dn-size:',len(self.global_feature_dict['Up-Dn-size']))
        print('Unique Uni-size:',len(self.global_feature_dict['Uni-size']))
        print('Unique Uni-time:',len(self.global_feature_dict['Uni-time']))
        print('Unique Pkt-size:',len(self.global_feature_dict['Pkt-size']))
    def vectorize(self,K=1000,raw_feature_dictory=None,test_split_ratio=0.1):
        self.filter(K)
        self.X =[]
        self.y=[]

        if raw_feature_dictory == None:
            raw_feature_dictory = self.raw_feature_dictory
        for _root,_dirs,_files in os.walk(raw_feature_dictory):
                for file in _files:
                    if not file.split('.')[-1]=='bind':
                        continue
                    packageName = file.split(".bind")[0]
                    file = (_root + file).replace("\\","/")

                    with open(file,'rb') as fp:
                        raw_traces = pickle.load(fp)


                    for trace in raw_traces:
                        Dn_Up_size = np.zeros(K,dtype = np.int8)
                        Dn_Up_time = np.zeros(K,dtype = np.int8)
                        Up_Dn_size = np.zeros(K,dtype = np.int8)
                        Up_Dn_time = np.zeros(K,dtype = np.int8)
                        Uni_size =np.zeros(K,dtype = np.int8)
                        Uni_time =np.zeros(K,dtype = np.int8)
                        Pkt_size =np.zeros(K,dtype = np.int8)
                        for key in trace['Dn-Up-size']:
                            if key in self.global_feature_dict['Dn-Up-size']:
                                Dn_Up_size[self.global_feature_dict['Dn-Up-size'][key]] += trace['Dn-Up-size'][key]

                        for key in trace['Dn-Up-time']:
                            if key in self.global_feature_dict['Dn-Up-time']:
                                Dn_Up_time[self.global_feature_dict['Dn-Up-time'][key]] += trace['Dn-Up-time'][key]

                        for key in trace['Up-Dn-size']:
                            if key in self.global_feature_dict['Up-Dn-size']:
                                Up_Dn_size[self.global_feature_dict['Up-Dn-size'][key]] += trace['Up-Dn-size'][key]

                        for key in trace['Up-Dn-time']:
                            if key in self.global_feature_dict['Up-Dn-time']:
                                Up_Dn_time[self.global_feature_dict['Up-Dn-time'][key]] += trace['Up-Dn-time'][key]

                        for key in trace['Uni-size']:
                            if key in self.global_feature_dict['Uni-size']:
                                Uni_size[self.global_feature_dict['Uni-size'][key]] += trace['Uni-size'][key]

                        for key in trace['Uni-time']:
                            if key in self.global_feature_dict['Uni-time']:
                                Uni_time[self.global_feature_dict['Uni-time'][key]] += trace['Uni-time'][key]

                        for key in trace['Pkt-size']:
                            if key in self.global_feature_dict['Pkt-size']:
                                Pkt_size[self.global_feature_dict['Pkt-size'][key]] += trace['Pkt-size'][key]
                        #合并
                        X=np.concatenate([Dn_Up_size,Dn_Up_time,Up_Dn_size,Up_Dn_time,Uni_size,Uni_time,Pkt_size])
                        #print(self.global_feature_dict['Label'])
                        y=self.global_feature_dict['Label'][packageName]
                        self.X.append(X)
                        self.y.append(y)
        assert  len(self.X) == len(self.y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        print('X shape', self.X.shape)
        print('y shape', self.y.shape)
        p =[x for x in range(self.y.shape[0])]
        self.rand.shuffle(p)

        self.X = self.X[p]
        self.y =self.y[p]

        test_split_index= int(self.y.shape[0] * test_split_ratio)
        X_test = self.X[:test_split_index]
        y_test = self.y[:test_split_index]
        X_train = self.X[test_split_index:]
        y_train=self.y[test_split_index:]
        X_valid=self.X[:test_split_index]
        y_valid=self.y[:test_split_index]
        return X_train,y_train,X_test,y_test,X_valid,y_valid
if __name__ == '__main__':
    bd = builder(raw_feature_dictory='./raw_feature/',global_feature_dict_filename="./raw_feature/global_feature_dict.vocb")
    bd.vectorize()
