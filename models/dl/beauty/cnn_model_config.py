__author__ = 'dk'
'''
###这个模型来自于：@inproceedings{schuster2017beauty,
title={Beauty and the burst: Remote identification of encrypted video streams},
author={Schuster, Roei and Shmatikov, Vitaly and Tromer, Eran},
booktitle={26th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 17)},
pages={1357--1374},
year={2017}
}
'''
learning_params_template={
  "epoch": 200,
  "input_length":200,
  "maxlen": 200,                #向量长度【此处需要修改】
  "nb_features": 1,
  "batch_size": 256,
  "val_split": 0.05,
  "test_split": 0.05,
  "optimizer": "adam",
  "lr": 0.0008,
  "decay": 0,
  'momentum':0.9,
  "nb_layers": 7,
  "layers": [
    {
      "name": "conv",
      "filters": 32,
      "kernel_size": 5,
      "activation": "relu",
      "stride": 1
    },
    {
      "name": "conv",
      "filters": 32,
      "kernel_size": 5,
      "activation": "relu",
      "stride": 1
    },
    {
      "name": "conv",
      "filters": 32,
      "kernel_size": 5,
      "activation": "relu",
      "stride": 1,
    },
    {
        'name':'dropout',
        'rate': 0.5
    },
    {
      "name": "maxpooling",
      "pool_size": 4
    },
        {
        'name':'dropout',
        'rate': 0.3
    },
    {
      "units": 64,                     #这个就是最后一层的输出神经元个数,必须等于nb_classes
      "name": "dense",
      "activation": "relu",
      "regularization": 0.0
    },
    {
        'name':'dropout',
        'rate': 0.5
    },
    {
          'name':'flatten'
    },
    {
      "last": True,
      "units": None,                     #这个就是最后一层的输出神经元个数,必须等于nb_classes
      "name": "dense",
      "activation": "softmax",
      "regularization": 0.0
    }
  ]
}
try:
    assert  learning_params_template['maxlen']==learning_params_template['input_length']
except AssertionError as exp:
    print("cnn model: The max_len should be equal to input_length, because they are alias name for each other.")
    print("{0}:{1}".format(__file__,str(exp)))
    raise AssertionError(exp)
