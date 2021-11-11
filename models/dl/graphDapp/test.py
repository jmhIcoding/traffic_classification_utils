__author__ = 'dk'
import  numpy as np
from  models.dl.graphDapp import logger_wrappers
import torch as th
from torch import nn
from torch import optim
from torch.nn import functional as F

from models.dl.graphDapp.model_seriealization import save,load
from models.dl.graphDapp.data_builder import Dataset_fgnet
from models.dl.graphDapp.DApp_Classifier import DApp_classifier
from models.dl.graphDapp.graphDapp_config import config
from sklearn.metrics import classification_report
use_gpu = th.cuda.is_available()
if use_gpu :
    device_id = config['device_id']
    device= device_id
else:
    device= "cpu"

def main(dataset_name, modelpath,max_epoch=0):
    data_loader = Dataset_fgnet(raw_dir=r'',dumpfile=dataset_name,renew=False)
    print(data_loader)
    model = DApp_classifier(nb_classes=len(data_loader.labelname),
                            gin_layer_num=config['gin_layer_num'],
                            gin_hidden_units=config['gin_hidden_units'],
                            iteration_nums=config['iteration_nums'],
                            device= device,use_gpu= use_gpu)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=5e-5)

    model = load(model,optimizer=optimizer,checkpoint_path=modelpath)
    if use_gpu:
        model = model.cuda(device)
    batch_size = config['batch_size']

    model.eval()
    acc_list =[]
    ground_truth = []
    predict_truth = []

    for subset in range(len(data_loader.test_set)//batch_size):
        graphs,labels = data_loader.next_test_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        ground_truth = ground_truth + labels.tolist()
        predict_truth = predict_truth + argmax_labels.tolist()
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        acc_list.append(acc)
        info='Accuracy of argmax predictions on the test subset{1}: {0:4f}%'.format(acc,subset)
    info = 'Average Accuracy on entire test set:{:0.4f}%'.format(np.mean(acc_list))
    logger_wrappers.info(info)
    print(classification_report(y_true=ground_truth,y_pred=predict_truth,digits=5))
