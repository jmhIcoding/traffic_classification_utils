__author__ = 'dk'
import numpy as np
import  torch as th
import dgl
import torch.nn as nn
from dgl.nn.pytorch import GINConv
from  models.dl.graphDapp.data_builder import Dataset_fgnet
class DApp_MLP(nn.Module):
    def __init__(self,in_feats,out_feats=64, layer_nums = 3):
        super(DApp_MLP,self).__init__()
        self.linear_layers =nn.ModuleList()
        for each in range(layer_nums):
            if each == 0 :
                in_features= in_feats
            else:
                in_features = out_feats
            self.linear_layers.append(nn.Linear(in_features= in_features,out_features=out_feats))
        self.activate = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x1 = x
        for mod in self.linear_layers :
            x1 = mod(x1)
            x1 = self.activate(x1)

        x2 = self.batchnorm(x1)
        x3 = self.dropout(x2)
        return x3

class DApp_classifier(nn.Module):
    def __init__(self, nb_classes=53, gin_layer_num=3, gin_hidden_units=64, iteration_nums = 3, graph_pooling_type='sum',
                 neighbor_pooling_type='sum',use_gpu=False, device='cpu', iteration_first=True, embedding= True):
        #DApp: 3个GIN,顺序级联在一起
        super(DApp_classifier,self).__init__()

        self.nb_classes = nb_classes
        self.gin_layer_num = gin_layer_num
        self.gin_hidden_uints = gin_hidden_units
        self.iteration_nums = iteration_nums

        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type= neighbor_pooling_type

        self.use_gpu = use_gpu
        self.device = device

        self.gin_layers = []
        self.interation_first = iteration_first
        self.embedding = embedding
        self.embedding_dim = gin_hidden_units      #embedding的设置为gin的隐藏神经元个数

        if embedding :
            self.embedding_layer = th.nn.Embedding(num_embeddings= 3100, embedding_dim= self.embedding_dim)
        #添加GIN层
        if iteration_first == False:
            for each in range(gin_layer_num):
                if each == 0:
                    in_feats = self.embedding_dim if self.embedding == True else 1
                else:
                    in_feats = gin_hidden_units
                mlp = DApp_MLP(in_feats, out_feats= gin_hidden_units, layer_nums= self.gin_layer_num)
                print(mlp)
                if use_gpu :
                    mlp = mlp.to(th.device(device))
                gin_layer =GINConv(
                    apply_func= mlp,
                    aggregator_type= self.neighbor_pooling_type,
                    learn_eps=True
                )
                if use_gpu:
                    gin_layer = gin_layer.to(th.device(device))
                self.gin_layers.append(gin_layer)
        else:
            if embedding == False:
                mlp = DApp_MLP(1,out_feats=gin_hidden_units,layer_nums= self.gin_layer_num)
            else:
                mlp = DApp_MLP(self.embedding_dim, gin_hidden_units, layer_nums= self.gin_layer_num)
            if use_gpu:
                mlp = mlp.to(th.device(device))
            print(mlp)
            gin_layer = GINConv(
                apply_func=mlp,
                aggregator_type= self.neighbor_pooling_type,
                learn_eps=True
            )
            if use_gpu:
                gin_layer = gin_layer.to(th.device(device))
            self.gin_layers.append(gin_layer)
        #最后的全连接分类层
        self.linear = nn.Linear(in_features=iteration_nums * gin_hidden_units,out_features=nb_classes)


    def forward(self, g):

        node_feature = g.ndata['pkt_length']

        if self.embedding == True:
            node_feature = self.embedding_layer(th.reshape(node_feature.long(),(-1,)) + Dataset_fgnet.MTU)

        graph_feature_history = []
        ##gin
        if self.interation_first == False:
            for layer in self.gin_layers:
                node_feature = layer(g, node_feature.to(th.device(self.device)))
                g.ndata['iterated_feature'] = node_feature
                if self.graph_pooling_type == 'sum':
                    graph_feature = dgl.sum_nodes(g,'iterated_feature')
                elif self.graph_pooling_type == 'mean':
                    graph_feature = dgl.mean_nodes(g,'iterated_feature')

                graph_feature_history.append(graph_feature)
        else:
            layer = self.gin_layers[-1]
            # 只有一个MLP
            for i in range(self.iteration_nums):
                node_feature = layer(g, node_feature.to(th.device(self.device)))
                g.ndata['iterated_feature'] = node_feature
                if self.graph_pooling_type == 'sum':
                    graph_feature = dgl.sum_nodes(g,'iterated_feature')
                elif self.graph_pooling_type == 'mean':
                    graph_feature = dgl.mean_nodes(g,'iterated_feature')

                graph_feature_history.append(graph_feature)

        ##把所有的历史concate起来,

        graph_features = th.cat(graph_feature_history,-1)

        #全连接分类
        power = self.linear(graph_features)
        return  power







