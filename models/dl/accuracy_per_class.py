__author__ = 'dk'
#计算每个类别的准确率
import json
from sklearn import  metrics
from sklearn.metrics import classification_report
def accuracy_per_class(y_real,y_pred):
    right={}
    error={}
    for i in range(len(y_real)):
        if y_real[i] not in right:
            right.setdefault(y_real[i],0)
        if y_real[i] not in error:
            error.setdefault(y_real[i],0)
        if y_real[i]==y_pred[i]:
            right[y_real[i]] += 1
        else:
            error[y_real[i]] += 1
    acc={}
    for each in right:
        acc.setdefault(each,right[each]/(right[each]+error[each]))
    print('Accuracy of each class:')
    print(acc)
    #for i in range(len(right)):
    #    print("%0.2d\t%0.4f"%(i,acc[i]*100 if i in acc else 100))

    #计算各种率
    print(classification_report(y_true=y_real,y_pred=y_pred,digits=5))