__author__ = 'dk'
###这个模型来自于：Automated website fingerprinting through deep learning (Vera Rimmer et.al )
nb_classes_template  = 100       #分类的目标类别数目,指网站的数目【此处需要修改】
learning_params_template={
  "nb_epochs": 15,
  "input_length":40,
  "maxlen": 40,                #向量长度【此处需要修改】
  "nb_features": 1,
  "batch_size": 256,
  "val_split": 0.05,
  "test_split": 0.05,
  "optimizer": "rmsprop",
  "lr": 0.0008,
  "decay": 0,
  'momentum':0.9,
  "nb_layers": 7,
  "layers": [
    {
      "name": "conv",
      "rate": 0.25,
      "filters": 32,
      "kernel_size": 5,
      "activation": "relu",
      "stride": 1
    },
    {
      "name": "conv",
      "pool_size": 4,
      "filters": 32,
      "kernel_size": 5,
      "activation": "relu",
      "stride": 1
    },
    {
      "name": "maxpooling",
      "pool_size": 4
    },
    {
      "name": "lstm",
      "units": 128
    },
    {
      "last": True,
      "units": nb_classes_template,                     #这个就是最后一层的输出神经元个数,必须等于nb_classes
      "name": "dense",
      "activation": "softmax",
      "regularization": 0
    }
  ]
}
try:
    assert nb_classes_template == learning_params_template['layers'][-1]['units']
except  AssertionError as exp:
    print("cnn model: The last layer units should be equals to the number of classes.")
    print("{0}:{1}".format(__file__,str(exp)))
    raise AssertionError(exp)
try:
    assert  learning_params_template['maxlen']==learning_params_template['input_length']
except AssertionError as exp:
    print("cnn model: The max_len should be equal to input_length, because they are alias name for each other.")
    print("{0}:{1}".format(__file__,str(exp)))
    raise AssertionError(exp)
