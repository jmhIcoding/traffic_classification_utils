__author__ = 'dk'
import logging
import datetime
logger_name = 'unsupervised_adaption_for_traffic_classification'
scirpt_name = ''
logger=logging.Logger(logger_name)
logger.setLevel(logging.NOTSET)
_WARNING = 10
_INFO = 100
_ERROR= 0
level = _INFO
def warning(msg):
    if level < _WARNING :
        return
    msg =  "Time:{0}, [{2}-WARN]: {1}".format(datetime.datetime.now(),msg, scirpt_name)
    logger.warning(msg)
    with open(logger_name+'.log','a') as fp:
        fp.writelines(msg+'\n')

def info(msg):
    if level < _INFO :
        return
    msg="Time:{0}, [{2}-INFO]: {1}".format(datetime.datetime.now(),msg, scirpt_name)
    logger.warning(msg)
    with open(logger_name+'.log','a') as fp:
        fp.writelines(msg+'\n')
def error(msg):
    msg ="Time:{0}, [{2}-ERROR]: {1}".format(datetime.datetime.now(),msg, scirpt_name)
    logger.warning(msg)
    with open(logger_name+'.log','a') as fp:
        fp.writelines(msg+'\n')
