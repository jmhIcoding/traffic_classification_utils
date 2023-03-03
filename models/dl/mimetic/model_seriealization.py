__author__ = 'dk'
#模型的保存和加载
import torch
import json
import logger_wrappers
import os

def save(model,model_path):
    checkpoint_path = model_path
    path = model_path
    if os.path.exists(os.path.dirname(checkpoint_path)) == False:
        logger_wrappers.warning('create checkpoint path: {0}'.format(os.path.dirname(checkpoint_path)))
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok= True)
    torch.save(model.state_dict(),path)
    info = "Dump model to {0} well.".format(checkpoint_path)
    logger_wrappers.warning(info)

def load(model,model_path, use_gpu=True, device=None):
    #print(device)
    path = model_path
    if os.path.exists(path):
        if use_gpu == False:
            map_location= torch.device('cpu')
        else:
            map_location = lambda storage, loc: storage.cuda(int(device.split(":")[-1]))
        model_CKPT = torch.load(path, map_location=map_location)
        model.load_state_dict(model_CKPT)
        info ="Load model from {0} well.".format(path)
        logger_wrappers.warning(info)
    else:
        logger_wrappers.warning('Load empty model from {0}.'.format(path))
    return model#,optimizer
