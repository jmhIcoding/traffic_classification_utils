__author__ = 'dk'
#appscanner使用的特征提取方法,提取得到54维统计特征
import numpy as np
from scipy.stats import skew,kurtosis
_min=[1e9+1] * 54
_max=[0.0] * 54
def feature_trace(trace):
    feature = [0.0] * 18
    if len(trace)==0:
        return  feature
    feature[0] = np.min(trace)
    feature[1] = np.max(trace)
    feature[2] = np.mean(trace)
    feature[3] = np.median(np.absolute(trace-np.mean(trace)))
    feature[4] = np.std(trace)
    feature[5] = np.var(trace)
    feature[6] = skew(trace)
    feature[7] = kurtosis(trace)
    ##百分位数
    p=[10,20,30,40,50,60,70,80,90]
    percentile =np.percentile(trace,p)
    for i in range(9):
        feature[8+i] = percentile[i]
    feature[17]= len(trace)
    return  feature

def feature_extract(pkt_length_sequence):
    ingoing_trace =[]
    outgoing_trace =[]
    trace =[]
    pkt_length_sequence = np.array(pkt_length_sequence)
    pkt_length_sequence = pkt_length_sequence.reshape((-1))
    for i in range(pkt_length_sequence.shape[0]):
        if pkt_length_sequence[i] < 0 :
            ingoing_trace.append(pkt_length_sequence[i])
        if pkt_length_sequence[i] > 0 :
            outgoing_trace.append(pkt_length_sequence[i])
        if pkt_length_sequence[i]!=0:
            trace.append(pkt_length_sequence[i])
        if pkt_length_sequence[i]==0:
            break

    in_feature = feature_trace(ingoing_trace)
    out_feature = feature_trace(outgoing_trace)
    bi_feature = feature_trace(trace)

    feature =   in_feature+out_feature+bi_feature
    for i in range(54):
        if feature[i] > _max[i] :
            _max[i] = feature[i]
        if feature[i] < _min[i]:
            _min[i] = feature[i]
    return  feature
def normalize(feature,min=None,max=None):
    if type(min) == type(None):
        min = _min
    if type(max) == type(None):
        max = _max
    return  (feature-min)/(max-min)

if __name__ == '__main__':
    pkt_length_seq =[383, -290, 90, -165, 1448, 463, 929, 389, 1448, 976, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448, 717, 105, 1448, 1448, 1448, 1448, 1051, 144, 219, 196, 603, 113]
    x=feature_extract(pkt_length_seq)
    print(x)
    print(len(x))
