__author__ = 'dk'
import logging
import datetime
logger=logging.Logger('graph neural network')
logger.setLevel(logging.NOTSET)
_WARNING = 10
_INFO = 100
_ERROR= 0
level = _INFO
def warning(msg):
    if level < _WARNING :
        return
    logger.warning(msg="Time:{0}, [WARN]: {1}".format(datetime.datetime.now(),msg))

def info(msg):
    if level < _INFO :
        return
    logger.warning(msg="Time:{0}, [INFO]: {1}".format(datetime.datetime.now(),msg))

def error(msg):
    logger.warning(msg="Time:{0}, [ERROR]: {1}".format(datetime.datetime.now(),msg))