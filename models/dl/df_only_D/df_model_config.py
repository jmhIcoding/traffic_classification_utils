__author__ = 'dk'
nb_classes_template = 55  #标签的个数 number of classes【此处需要修改】
learning_params_template ={
    "epoch":200,
    "batch_size":128,
    "in_dim":200,              #输入向量的长度【此处需要修改】
    "input_length":200,        #输入向量的长度【此处需要修改】
    "lr":0.002,                  #学习速率
    "beta_1":0.9,
    "beta_2":0.999,
    "epsilon":1e-08,
    "decay":0.0
}

assert  learning_params_template['in_dim']==learning_params_template['input_length']
