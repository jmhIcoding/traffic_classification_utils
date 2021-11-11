__author__ = 'dk'
import os
import json
import dgl
import numpy as np
import random as Random
import gzip
import pickle
import torch as th
import tqdm

from scipy import sparse

class Dataset_fgnet:
    '''
    FGNet 数据集的访问类
    '''
    MTU = 1500
    def __init__(self, raw_dir, dumpfile, renew=False, batch_size=128, split_rate=0.1, randseed=128):
        '''

        :param raw_dir: FG-Net 数据集的原始目录，这些数据应该是在data/fgnet_dataset目录
        :param dumpfile:  数据集序列化文件,
        :param renew:  是否重新构建数据集
        :param split_rate: 数据集划分比例; 测试集和验证集的占比
        :param batch_size: 批大小
        :return:
        '''

        self.split_rate = split_rate
        self.randseed = 128
        self.my_rand = Random.Random(self.randseed)

        self.batch_size = batch_size
        self.raw_directory = raw_dir
        self.dumpfile = dumpfile

        self.flow_length = []
        self.flow_graph  = []
        self.flow_label  = []
        self.labelname = {} #标签对应的真实app名称

        self.train_set = []
        self.valid_set = []
        self.test_set = []

        self.train_watch = 0
        self.test_watch = 0
        self.valid_watch = 0
        self.epoch_num = 0

        if renew == False:
            if os.path.exists(dumpfile):
                self.load_dumpfile()
            else:
                self.load_raw_dateset()
        else:
            self.load_raw_dateset()

        self.my_rand.shuffle(self.train_set)
        self.my_rand.shuffle(self.test_set)
        self.my_rand.shuffle(self.valid_set)
    def generate_traffic_interact_graph(self, packet_length, cmpnet = False):
        graph = dgl.DGLGraph()
        packet_length = packet_length[:200]     #最多使用200个包长
        if cmpnet == False:
            graph.add_nodes(len(packet_length))

        pkt_length_matrix = np.zeros(shape=(len(packet_length),1),dtype=np.float32)
        for i in range(len(packet_length)):
            pkt_length_matrix[i] = packet_length[i]
        if cmpnet == False:
            ##按照沈蒙的图结构
            #inter-burst
            class _burst_:
                def __init__(self,s=0):
                    self.start = s
                    self.end= s
            burst = [_burst_(s=0) ]
            for i in range(1,len(packet_length)):
                if np.sign(packet_length[i]) == np.sign(packet_length[burst[-1].end]):
                    graph.add_edge(i-1,i)
                    burst[-1].end= i
                else:
                    burst.append(_burst_(s=i))
            #intra-burst
            for i in range(len(burst)-1):
                graph.add_edge(burst[i].start,burst[i+1].start)
                graph.add_edge(burst[i].end,burst[i+1].end)
        else:
            adj_matrix = np.ones(shape=(len(packet_length),len(packet_length)))
            graph.from_scipy_sparse_matrix(sparse.csr_matrix(adj_matrix))
        graph.ndata['pkt_length'] = pkt_length_matrix
        return graph
    def load_raw_dateset(self, raw_dir= None):
        if raw_dir == None:
            raw_dir = self.raw_directory
        self.flow_length = []
        self.flow_graph  = []
        self.flow_label  = []
        self.labelname = {} #标签对应的真实app名称

        flow_length = []
        flow_label = []
        for _root, _dirs, _files in os.walk(raw_dir):
            for file in _files:
                if '.json'  in file:
                    package = ".".join(file.split(".")[:-1])
                    if package not in self.labelname :
                        self.labelname.setdefault(package,len(self.labelname))

                    file  = _root + "/" + file
                    with open(file) as fp:
                        flows = json.load(fp)
                    #if len(flow_label) > 50 :
                    #    break
                    for flow in flows:
                        packet_length = flow['packet_length']
                        packet_length = self.refine_packet_length(packet_length)
                        flow_length.append(packet_length)
                        flow_label.append(self.labelname[package])

        ##依次生成TIG
        for i in tqdm.trange(len(flow_length)):
            each = flow_length[i]
            graph = self.generate_traffic_interact_graph(each)
            if len(graph.nodes) == 0 or len(graph.edges) == 0:
                continue
            self.flow_graph.append(graph)
            self.flow_length.append(flow_length[i])
            self.flow_label.append(flow_label[i])
        ##划分数据集
        for i in range(len(self.flow_label)):
            r = self.my_rand.uniform(0, 1)
            if r < self.split_rate:
                self.test_set.append(i)
            elif r < (self.split_rate + self.split_rate * (1-self.split_rate)):
                self.valid_set.append(i)
            else:
                self.train_set.append(i)

        self.save_dumpfile()
    def refine_packet_length(self,packet_length):
        rst =[]
        for each in packet_length:
            if abs(each) < 1500 :
                rst.append(each)
            else:
                sign = np.sign(each)
                each = abs(each)
                while each > 1500:
                    rst.append(sign * 1500)
                    each = each -1500
        return  rst
    def load_dumpfile(self, dumpfile = None):
        print('Load data dumpfile for history.')
        if dumpfile == None :
            dumpfile = self.dumpfile
        fp = gzip.GzipFile(dumpfile,'rb')
        data = pickle.load(fp)
        fp.close()
        self.flow_label = data['flow_label']
        self.flow_length = data['flow_length']
        self.flow_graph = data['flow_graph']
        self.train_set = data['train_set']
        self.test_set = data['test_set']
        self.valid_set = data['valid_set']
        self.labelname = data['labelname']

    def save_dumpfile(self,dumpfile=None ):
        if dumpfile== None:
            dumpfile = self.dumpfile
        fp = gzip.GzipFile(dumpfile,'wb')
        pickle.dump({
            'flow_graph': self.flow_graph,
            'flow_length': self.flow_length,
            'flow_label': self.flow_label,
            'train_set': self.train_set,
            'test_set': self.test_set,
            'valid_set': self.valid_set,
            'labelname': self.labelname
        },file = fp,protocol=2)
        fp.close()

    def __next_batch(self,name,batch_size):
        graphs =[]
        labels =[]

        for i in range(batch_size):
            if name == 'train':
                #print(self.train_watch,len(self.train_set),len(self.flow_graph),self.train_set[self.train_watch])
                graphs.append(self.flow_graph[self.train_set[self.train_watch]])
                labels.append(self.flow_label[self.train_set[self.train_watch]])

                if (self.train_watch + 1) == len(self.train_set):
                    self.epoch_num += 1

                self.train_watch = (self.train_watch + 1) % len(self.train_set)
            elif name =='valid':
                graphs.append(self.flow_graph[self.valid_set[self.valid_watch]])
                labels.append(self.flow_label[self.valid_set[self.valid_watch]])
                self.valid_watch = (self.valid_watch + 1) % len(self.valid_set)
            else:
                graphs.append(self.flow_graph[self.test_set[self.test_watch]])
                labels.append(self.flow_label[self.test_set[self.test_watch]])
                self.test_watch = (self.test_watch + 1) % len(self.test_set)
        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return self.__next_batch('train', batch_size)
    def next_valid_batch(self,batch_size= None):
        if batch_size == None:
            batch_size = self.batch_size
        return self.__next_batch('valid', batch_size)

    def next_test_batch(self, batch_size= None):
        if batch_size == None:
            batch_size = self.batch_size
        return self.__next_batch('test', batch_size)

    def __str__(self):
        return  'Total Sample:{0}, train sample: {1}, test sample: {2}, valid sample: {3}, class num: {4}'.format(len(self.flow_label),len(self.train_set),len(self.test_set),len(self.valid_set),len(self.labelname))

Dataset = Dataset_fgnet

if __name__ == '__main__':
    #packet_length =[-571,1514,1142,-118,-140,-330,618,85,-85,-361,279,93,-93,55]
    #graph = Dataset_fgnet.generate_traffic_interact_graph(packet_length)
    #print(packet_length)

    #nxg = graph.to_networkx()
    #pos = nx.spring_layout(nxg)

    #labels ={i: str(packet_length[i]) for i in range(len(packet_length))}
    #nx.draw(nxg,with_labels=True,labels=labels)

    #plt.show()
    #exit()
    dator = Dataset_fgnet(raw_dir=r'../data/datacon',dumpfile='datacon.gzip',renew= True)
    print(dator.labelname)
    print(dator)
    print(dator.next_train_batch(batch_size=1))
    dator.next_test_batch(batch_size=2)
    print(dator.next_valid_batch(batch_size=2))
