__author__ = 'dk'
###这个模型来自于：Automated website fingerprinting through deep learning (Vera Rimmer et.al )
#学习的参数
nb_classes_template  = 100       #分类的目标类别数目,指网站的数目【此处需要修改】
learning_params_template={
    "nb_epochs" : 30,
    "maxlen" : 40,             #原始向量的长度【此处需要修改】
    "features" : 2,
    "batch_size" : 32,
    "val_split" : 0.05,
    "test_split" : 0.05,
    "optimizer" : "sgd",
    "nb_layers" : 3,
    "lr" : 0.001,
    "momentum" : 0.9,
    "decay" : 0.0,
    "nesterov" : True,
    "layers":#各个自编码器层的参数设置
    [
        {   #第一层的超参数
            "in_dim" : 40, #encoder输入向量长度【此处需要修改】
            "out_dim" : 700,#decoder输出向量长度
            "epochs": 20,
            "batch_size": 128,
            "dropout" :0.2, #dropout的概率
            "optimizer" : "sgd",    #本层的优化器,可选性:sgd(随机梯度下降),adam,rmsprop
            "enc_activation" : "tanh",#编码器的激活函数
            "dec_activation" : "linear",#解码器的激活函数
            "lr":0.001,     #sgd的优化器参数
            "momentum" : 0.9,
            "decay" : 0.0
        },
        {   #第二层超参数
            "in_dim": 700,
            "out_dim": 500,
            "epochs": 10,
            "batch_size": 128,
            "dropout":0.2,
            "optimizer":"sgd",
            "enc_activation": "tanh",
            "dec_activation":"linear",
            "lr": 0.001,
            "momentum":0.9,
            "decay": 0.0
        },
        {   #第三层超参数
            "in_dim" : 500,
            "out_dim":  300,
            "epochs":  10,
            "batch_size": 128,
            "dropout": 0.2,
            "optimizer": "sgd",
            "enc_activation": "tanh",
            "dec_activation": "linear",
            "lr" : 0.001,
            "momentum":  0.9,
            "decay" : 0.0
        }
    ]
}
