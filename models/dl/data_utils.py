__author__ = 'dk'
import sys
import numpy as np
import pickle
import gzip
from keras.utils import np_utils
def normalize(x):
    return x
    positive = 1
    negative = 1
    if x > 0:
        x = x / positive
    elif x < 0:
        x = x / negative
    return x
def Cluster_consecutive_bursts(X,normalized=True,padding=True,return_array=True):
    LENGTH=40
    samples,dimentions = X.shape
    rst=[]
    for i in range(samples):
        x=[]
        last = 0
        cnt = 1
        for j in range(dimentions):
            if j ==0:
                last = X[i][j]
            elif last==X[i][j]:
                cnt+=1
            else:
                if normalized:
                    x.append(normalize(cnt*last))
                else:
                    x.append(int(cnt*last))
                last = X[i][j]
                cnt =1
        #填充0
        if padding:
            x+=[0.0]*(LENGTH-len(x))
            rst.append(x[:LENGTH])
        else:
           rst.append(x)
    if return_array:
        return  np.array(rst)
    else:
        return  rst

def LoadDataNoDefCW():
    #来自walkie-talkie这些数据
    NB_CLASSES=100
    print ("Loading non-defended dataset from Walkie-Talkie for closed-world scenario 0518")
    # Point to the directory storing data
    if sys.platform !='linux':
        dataset_dir = r"E:\\NoDef\\"
    else:
        dataset_dir=r"/home3/jmh/wt_dataset/NoDef/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='bytes'))

    X_train = Cluster_consecutive_bursts(X_train,normalized=False)
    X_valid = Cluster_consecutive_bursts(X_valid,normalized=False)
    X_test = Cluster_consecutive_bursts(X_test,normalized=False)

    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test  = X_test[:, :,np.newaxis]

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
def LoadDataWalkieTalkieCW():
    #来自walkie-talkie这些数据
    NB_CLASSES=100
    print ("Loading defended dataset from Walkie-Talkie for closed-world scenario 0518")
    # Point to the directory storing data
    if sys.platform !='linux':
        dataset_dir = r"E:\\Def\\"
    else:
        dataset_dir=r"/home3/jmh/wt_dataset/Def/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_WalkieTalkie.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_test_WalkieTalkie.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='bytes'))

    X_train = Cluster_consecutive_bursts(X_train,normalized=False)
    X_valid = Cluster_consecutive_bursts(X_valid,normalized=False)
    X_test = Cluster_consecutive_bursts(X_test,normalized=False)

    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test  = X_test[:, :,np.newaxis]

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LoadDataNoDefCW100():
    #来自AWF 论文里面的CW100
    print ("Loading non-defended dataset from AWF for closed-world scenario 0518")
    # Point to the directory storing data
    if sys.platform !='linux':
        dataset_dir = r"E:\\awf_dataset\\cw100\\"
    else:
        dataset_dir=r"/home3/jmh/awf_dataset/cw100/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_cw100.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_train_cw100.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_cw100.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_valid_cw100.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_cw100.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='bytes'))
    with open(dataset_dir + 'y_test_cw100.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='bytes'))

    X_train = Cluster_consecutive_bursts(X_train,normalized=False)
    X_valid = Cluster_consecutive_bursts(X_valid,normalized=False)
    X_test = Cluster_consecutive_bursts(X_test,normalized=False)

    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test  = X_test[:, :,np.newaxis]

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
def LoadDataApp(dataset_dir=None):
    NB_CLASSES=55
    feature_name = "pkt_length"
    print ("Loading App Dataset")
    # Point to the directory storing data
    if dataset_dir==None:
        if sys.platform !='linux':
            dataset_dir = r"E:\\app_dataset\\"
        else:
            dataset_dir=r"./app_dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with gzip.GzipFile(dataset_dir + 'X_train_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_train_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='bytes'))

    # Load validation data
    with gzip.GzipFile(dataset_dir + 'X_valid_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_valid_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='bytes'))

    # Load testing data
    with gzip.GzipFile(dataset_dir + 'X_test_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_test_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='bytes'))

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
def LoadDataApp_crossversion(dataset_dir=None):
    NB_CLASSES=53
    feature_name = "pkt_length"
    print ("Loading App Cross Version Dataset")
    # Point to the directory storing data
    if dataset_dir==None:
        if sys.platform !='linux':
            dataset_dir = r"E:\\app_dataset_crossversion\\"
        else:
            dataset_dir=r"/home3/jmh/app_dataset_crossversion/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with gzip.GzipFile(dataset_dir + 'X_train_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_train_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='bytes'))

    # Load validation data
    with gzip.GzipFile(dataset_dir + 'X_valid_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_valid_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='bytes'))

    # Load testing data
    with gzip.GzipFile(dataset_dir + 'X_test_{0}.pkl'.format(feature_name), 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='bytes'))
    with gzip.GzipFile(dataset_dir + 'y_test_{0}.pkl'.format(feature_name), 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='bytes'))

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test=LoadDataApp()
    exit()
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1])
    X_valid = X_valid.reshape(X_valid.shape[0],X_valid.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])

    X= np.concatenate([X_train,X_valid,X_test])
    X = Cluster_consecutive_bursts(X,normalized=False,padding=False)
    lens = []
    min_=[]
    max_=[]
    for each in X:
        #print(each)
        lens.append(len(each))
        min_.append(np.min(each))
        max_.append(np.max(each))
    print(np.min(lens),np.max(lens),np.average(lens),np.std(lens))
    #聚合后向量长度：10 2305 329.312 272.7245666625271
    print(np.min(min_),np.max(max_))
    #-483.0 110.0
    #输出百分位数
    for i in range(0,101,1):

       print("percentile {0}:{1}".format(i,np.percentile(lens,i)))