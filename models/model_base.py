
import random
import os
import pickle
import numpy as np
import gzip
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, auc
class abs_model:
    def __init__(self, name, randseed):
        self.database = './data/'
        self.name = name
        self.rand = random.Random(x = randseed)
        self.data = None
        self.model = None
        self.full_rdata = []

    def data_exists(self):
        return  os.path.exists(self.data)
    def model_exist(self):
        return  os.path.exists(self.model)

    def train(self):
        pass

    def test(self):
        pass

    def parser_raw_data(self):
        ##从原始通用数据集获取自己所需格式数据集能力
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
    def save_data(self,X_train, y_train, X_valid, y_valid, X_test, y_test):
        fp = gzip.GzipFile(self.data + 'data.gzip','wb')
        pickle.dump({
            'X_train':X_train,
            'y_train':y_train,
            'X_valid':X_valid,
            'y_valid':y_valid,
            'X_test':X_test,
            'y_test':y_test
        },file=fp)
        fp.close()
    def load_data(self):
        fp = gzip.GzipFile(self.data + 'data.gzip','rb')
        data = pickle.load(fp)
        fp.close()
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_test = data['X_test']
        y_test = data['y_test']
        import random
        indexs = [x for x in range(len(y_test))]
        random.shuffle(indexs)
        return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test)[indexs], np.array(y_test)[indexs]
    def num_classes(self):
        for _root, _dir, _files in os.walk(self.full_rdata):
            classes = _files
        return len(classes)
    def fpr_tpr_auc(self,  y_pred, y_real,y_pred_logit=None):
        labels =set()
        for each in y_real:
            labels.add(each)
        labels =list(labels)
        mcm = multilabel_confusion_matrix(y_true=y_real,y_pred=y_pred,labels=labels)
        #print(mcm)
        fp ={}
        tp ={}
        fn ={}
        tn ={}
        for i in range(len(labels)):
            fp.setdefault(labels[i],mcm[i,0,1])
            tp.setdefault(labels[i],mcm[i,1,1])
            fn.setdefault(labels[i],mcm[i,1,0])
            tn.setdefault(labels[i],mcm[i,0,0])
        acc={}
        fpr={}
        tpr={}
        for each in fp:
            acc.setdefault(each,(tp[each]+tn[each])/(fp[each]+tn[each]+fn[each]+tp[each]))
            fpr.setdefault(each,fp[each]/(fp[each]+tn[each]))
            tpr.setdefault(each,tp[each]/(tp[each]+fn[each]))

        print('tpr:',tpr)

        print('fpr:',fpr)
        #auc = roc_auc_score(y_true=y_real, y_score=y_pred_logit[:,1])
        #print('auc (prob):', auc)

        auc = roc_auc_score(y_true=y_real, y_score=y_pred)
        print('auc (label):', auc)