
__author__ = 'dk'
###这个模型来自于：Automated website fingerprinting through deep learning (Vera Rimmer et.al )
#学习的参数
nb_classes_template  = 100       #分类的目标类别数目,指网站的数目【此处需要修改】
learn_params_template={
  "nb_epochs": 50,
  "maxlen": 40,                    #向量最大长度,最大包长序列长度【此处需要修改】
  "nb_features": 1,                #这个是每个向量的每个分量的维度,类似于embed后的长度,默认就是+1,-1的序列，所以长度为1。
  "batch_size": 256,
  "val_split": 0.15,
  "test_split": 0.15,
  "optimizer": "rmsprop",
  "nb_layers": 2,
  "layers": [
    {
      "units": 128,
      "dropout": 0.22244615886559121,
      "activation": "tanh",
      "rec_activation": "hard_sigmoid"
    },
    {
      "units": 128,
      "dropout": 0.20857652372682717,
      "activation": "tanh",
      "rec_activation": "hard_sigmoid"
    }
  ],
  "lr": 0.0010053829131721616,
  "decay": 0,
  "momentum": 0.9,
  "nesterov": True
}
