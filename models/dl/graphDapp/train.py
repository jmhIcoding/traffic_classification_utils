__author__ = 'dk'
import  numpy as np
from  models.dl.graphDapp import logger_wrappers
import torch as th
from torch import nn
from torch import optim
from torch.nn import functional as F
import tqdm

from sklearn.metrics import classification_report
from models.dl.graphDapp.model_seriealization import save,load
from models.dl.graphDapp.data_builder import Dataset_fgnet
from models.dl.graphDapp.DApp_Classifier import DApp_classifier
from models.dl.graphDapp.graphDapp_config import config
use_gpu = th.cuda.is_available()
if use_gpu :
    device_id = config['device_id']
    device = device_id
else:
    device= "cpu"

def main(dataset_name, modelpath, max_epoch=config['max_epoch']):
    data_loader = Dataset_fgnet(raw_dir='', dumpfile= dataset_name,renew=False)
    print(data_loader)
    model = DApp_classifier(nb_classes=len(data_loader.labelname),
                            gin_layer_num= config['gin_layer_num'],
                            gin_hidden_units=config['gin_hidden_units'],
                            iteration_nums=config['iteration_nums'],
							iteration_first=True,
                            device= device,use_gpu= use_gpu)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=config['learning_rate'])
    #model = load(model,optimizer=optimizer,checkpoint_path=modelpath)
    if use_gpu:
        model = model.cuda(device)
        loss_func = loss_func.cuda(device)

    #训练
    model.train()
    epoch_losses = []
    epoch_acces = []
    batch_size = config['batch_size']

    for epoch in tqdm.trange(max_epoch):
        epoch_loss = 0
        iter = 0
        while data_loader.epoch_num == epoch:
            graphs,labels= data_loader.next_train_batch(batch_size)
            if use_gpu :
                graphs = graphs.to(th.device(device))
                labels = labels.to(th.device(device))
            predict_label = model(graphs)
            #print(predict_label.size())
            #print(labels.size())
            loss = loss_func(predict_label,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_gpu:
                lv= loss.detach().item()
            else:
                lv = loss.detach().cpu().item()
            epoch_loss += lv
            iter +=1
            #print('Inner loss: {:.4f},Train Watch:{}'.format(lv,data_loader.train_watch))
            #epoch_losses.append(lv)
        epoch_loss /= (iter+0.0000001)
        info='Epoch {}, loss: {:.4f}'.format(epoch,epoch_loss)
        logger_wrappers.warning(info)
        epoch_losses.append(epoch_loss)
        #测试一下:
        graphs,labels = data_loader.next_valid_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        #print('pred:', argmax_labels)
        #print('real:', labels)
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        info='Accuracy of argmax predictions on the valid set: {:4f}%'.format(
            acc)
        epoch_acces.append(acc)
        logger_wrappers.info(info)
        ###保存一下模型
        save(model,optimizer,checkpoint_path=modelpath)
    model.eval()
    acc_list =[]
    y_pred=  []
    y_ture = []

    for subset in range(len(data_loader.test_set)//batch_size):
        graphs,labels = data_loader.next_test_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        y_pred += argmax_labels.tolist()
        y_ture += labels.tolist()
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        acc_list.append(acc)
        info='Accuracy of argmax predictions on the test subset{1}: {0:4f}%'.format(acc,subset)
        logger_wrappers.info(info)
    info = 'Average Accuracy on entire test set:{:0.4f}%'.format(np.mean(acc_list))
    logger_wrappers.info(info)
    print(classification_report(y_pred=y_pred,y_true=y_ture, digits=5))
