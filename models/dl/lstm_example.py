__author__ = 'dk'
import  os
#设置Tensorflow的日志等级
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

from attacks import  CNN_model,DF_model,SDAE_model,LSTM_model
from df import df_model_config

from cnn import cnn_model_config
from sdae import sdae_model_config
from lstm import  lstm_model_config

from data_utils import LoadDataNoDefCW100,LoadDataNoDefCW,LoadDataWalkieTalkieCW,LoadDataApp

#使用步骤
#1. 修改各个模型的超参数,xx_model_config.py,把里面的输入向量和标签数改成自己所需要的
#2. 读取数据,构造好训练集,验证集,测试集
#3. build_model()
#4. 调用fit()
#5. 测试一下
#6. 保存模型
def test_cnn(X_train,y_train,X_valid,y_valid,X_test,y_test):
    model = CNN_model()
    model.build_model()
    model.fit(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,
              batch_size=cnn_model_config.learning_params_template['batch_size'],
              epochs=cnn_model_config.learning_params_template['nb_epochs'])
    model.save_model()
    score = model.evaluate(X_test=X_test,y_test=y_test)
    print('simple CNN accuracy :{0}'.format(score))

def test_df(X_train,y_train,X_valid,y_valid,X_test,y_test):

    model =DF_model()
    model.build_model()

    model.fit(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,
              batch_size=df_model_config.learning_params_template['batch_size'],
              epochs=df_model_config.learning_params_template['epoch'])
    model.save_model()
    score = model.evaluate(X_test=X_test,y_test=y_test)
    print('Deep Fingerprinting accuracy :{0}'.format(score))



def test_lstm(X_train,y_train,X_valid,y_valid,X_test,y_test):

    model = LSTM_model()
    model.build_model()
    model.fit(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,
              batch_size=lstm_model_config.learn_params_template['batch_size'],
              epochs=lstm_model_config.learn_params_template['nb_epochs'])
    model.save_model()
    score = model.evaluate(X_test=X_test,y_test=y_test)
    print('lstm accuracy :{0}'.format(score))
def test_sdae(X_train,y_train,X_valid,y_valid,X_test,y_test):
    model = SDAE_model()
    model.build_model()
    model.pre_train(x_train=X_train,x_test=X_test)

    model.fit(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,
              batch_size=sdae_model_config.learning_params_template['batch_size'],
              epochs=sdae_model_config.learning_params_template['nb_epochs'])
    model.save_model()
    score = model.evaluate(X_test=X_test,y_test=y_test)
    print('sdae accuracy :{0}'.format(score))
if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test =  LoadDataApp()#LoadDataWalkieTalkieCW() # LoadDataNoDefCW()
    #test_df(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,X_test=X_test,y_test=y_test)
    #test_cnn(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,X_test=X_test,y_test=y_test)
    #test_sdae(X_train,y_train,X_valid,y_valid,X_test,y_test)
    if X_train.shape[1] > lstm_model_config.learn_params_template['maxlen']:
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1])
        X_valid = X_valid.reshape(X_valid.shape[0],X_valid.shape[1])
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])

        X_train = X_train[:,:lstm_model_config.learn_params_template['maxlen']]
        X_valid = X_valid[:,:lstm_model_config.learn_params_template['maxlen']]
        X_test = X_test[:,:lstm_model_config.learn_params_template['maxlen']]

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_valid = X_valid.reshape(X_valid.shape[0],X_valid.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    test_lstm(X_train,y_train,X_valid,y_valid,X_test,y_test)
