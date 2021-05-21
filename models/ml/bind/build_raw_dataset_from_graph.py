__author__ = 'dk'
#从Traffic_graph_generator里面的graph产生原始数据集
import json
import copy
import numpy as np
import os
import pickle
class raw_dataset_builder:
    def __init__(self,graph_json_directory,mode='clear',dst_directory='./raw_feature/'):
        if os.path.isdir(graph_json_directory)== False:
            info = '{0} is not a directory'.format(graph_json_directory)
            logger_wrappers.error(info)
            raise BaseException(info)
        if os.path.exists(dst_directory)== False:
            os.makedirs(dst_directory)
        assert mode in ['clear','noise','all']
        self.labelName = []
        self.labelNameSet={}
        self.labelId  =[]
        self.raw_feature ={}

        for _root,_dirs,_files in os.walk(graph_json_directory):
            if _root == graph_json_directory or len(_files)==0:
                continue
            _root =_root.replace("\\","/")
            packageName = _root.split("/")[-2]
            labelName = packageName
            self.labelNameSet.setdefault(labelName,len(self.labelNameSet))
            print(labelName)
            self.raw_feature.setdefault(labelName,[])
            for file in _files:
                json_fname = (_root +"\\" + file).replace("\\","/")

                if mode != 'all' and mode not in file:
                    continue
                with open(json_fname) as fp:
                    g = json.load(fp)

                if g == None or 'nodes' not in g:
                    continue

                for each_trace in g['nodes']:
                    feature = {
                        'Dn-Up-size':{},
                        'Dn-Up-time':{},
                        'Up-Dn-size':{},
                        'Up-Dn-time':{},
                        'Uni-size':{},
                        'Uni-time':{},
                        'Pkt-size':{}
                    }
                    burst_size =[0]
                    burst_time =[0]
                    direction =  0
                    for i in range(len(each_trace['packet_length'])):
                        #包级别的长度特征
                        if abs(each_trace['packet_length'][i]) not in feature['Pkt-size']:
                            feature['Pkt-size'].setdefault(abs(each_trace['packet_length'][i]),1)
                        else:
                            feature['Pkt-size'][abs(each_trace['packet_length'][i])] += 1

                        #Uni-burst的特征
                        sign = np.sign(each_trace['packet_length'][i])
                        if direction == 0:
                            direction = sign
                        if sign == direction:
                            #同向的,使用原来的burst
                            pass
                        else:
                            #方向相反,新增一个burst
                            direction =sign
                            burst_size.append(0)
                            burst_time.append(0)
                        burst_size[-1] += each_trace['packet_length'][i]            #累积包长
                        burst_time[-1] += int(each_trace['arrive_time_delta'][i] * 100)/100        #累积通信时长,保留两位小数
                    #构建burst级别的特征
                    for index in range(len(burst_size)):
                        #uni-burst
                        if abs(burst_size[index]) not in feature['Uni-size']:
                            feature['Uni-size'].setdefault(abs(burst_size[index]),1)
                        else:
                            feature['Uni-size'][abs(burst_size[index])] += 1
                        if burst_time[index] not in feature['Uni-time']:
                            feature['Uni-time'].setdefault(burst_time[index] ,1)
                        else:
                            feature['Uni-time'][burst_time[index] ] += 1
                        #bi-burst
                        if index > 0:
                            bi_size = (abs(burst_size[index-1]),abs(burst_size[index]))
                            bi_time =(burst_time[index-1],burst_time[index])
                            if np.sign(burst_size[index-1])== -1:
                                #Dn-Up Burst
                                if bi_size not in feature['Dn-Up-size']:
                                    feature['Dn-Up-size'].setdefault(bi_size,1)
                                else:
                                    feature['Dn-Up-size'][bi_size] += 1
                                if bi_time not in feature['Dn-Up-time']:
                                    feature['Dn-Up-time'].setdefault(bi_time,1)
                                else:
                                    feature['Dn-Up-time'][bi_time] += 1
                            elif np.sign(burst_size[index-1]) == 1:
                                #Up-Dn Burst:
                                if bi_size not in feature['Up-Dn-size']:
                                    feature['Up-Dn-size'].setdefault(bi_size,1)
                                else:
                                    feature['Up-Dn-size'][bi_size] += 1
                                if bi_time not in feature['Up-Dn-time']:
                                    feature['Up-Dn-time'].setdefault(bi_time,1)
                                else:
                                    feature['Up-Dn-time'][bi_time] += 1
                            else:
                                print('Packet size:',each_trace['packet_length'])
                                print('Uni-burst size:',burst_size)
                                print('Uni-burst time:',burst_time)
                                print('index:',index)
                                raise ValueError('Burst Direction Error!')
                    #print(feature)
                    #
                    self.raw_feature[packageName].append(copy.deepcopy(feature))
            with open(dst_directory+'/'+packageName+'.bind','wb') as fp:
                pickle.dump(self.raw_feature[packageName],fp)
if __name__ == '__main__':
    raw_dataset_builder("../Traffic_graph_generator/graph/",dst_directory="./D1_53/")
    raw_dataset_builder("../Traffic_graph_generator/graph/",mode='noise',dst_directory="./D1_53.noise/")
    raw_dataset_builder("../Traffic_graph_generator/graph0705/",dst_directory="./D2_53/")
    raw_dataset_builder("../Traffic_graph_generator/graph_meizunote2_0713/",dst_directory="./D3_53/")
    raw_dataset_builder("../Traffic_graph_generator/graph_oneplus6t/",dst_directory="./D4_53/")
    raw_dataset_builder("../Traffic_graph_generator/graph_xiaomi5plus_0720/",dst_directory="./D5_53/")

