#coding:utf-8
#提取peak的统计特征
__author__ = 'jmh081701'
import json
import numpy as np
def moment(n,data,c=0):
    # n: 表示求的n阶矩
    # data: 一个list
    # c: 0 原点矩;   1 中心矩
    if not isinstance(data,list):
        raise  Exception('unknown moment')
    bin = 30
    distrubute  = length_distribution(data,bin)
    rst= 0
    if c!=0:
        avg= np.average(data)
    else:
        avg=0
    for i in range(0,distrubute.shape[0]):
        rst += distrubute[i] * (i*bin - avg) **n
    if rst < 0 :
        return -1 * abs(rst) ** (1/n)
    else:
        return rst **(1/n)

def length_percentile(packet_length,percentage):
    #包长百分位数
    bin=5
    distrbution = length_distribution(packet_length,bin)
    cdf=0
    for i in range(distrbution.shape[0]):
        cdf +=distrbution[i]
        if cdf >=percentage:
            return i*bin + bin/2.0


def length_distribution(packet_length,bin=30):
    distribution = np.zeros(shape =(1500//bin,))
    for each in packet_length:
        each = int(abs(each))
        if each >= 1500 :
            each = 1499
        distribution[each//bin] +=1
    distribution =distribution / len(packet_length)
    #print(distribution)
    return distribution
def poison_generate_function(data):
    rst = []
    for x in np.arange(1.1,10,2):
        sum =0
        for n in range(0,len(data)):
            sum =sum + data[n]*np.math.sin(2*x*np.math.pi /31 *n)
        rst.append(round(sum,3))
    return rst
def _poison_generate_function(data):
    rst =[]
    sum =0
    for n in range(0,len(data)):
        sum =sum + data[n]* np.math.pow(np.math.e,-x) * np.math.pow(x,1+n)/np.math.factorial(1+n)
    rst.append(round(sum,3))
    return rst
def L_generate_function(data):
    rst = []
    for x in np.arange(1.1,10,2):
        sum =0
        for n in range(0,len(data)):
            sum =sum + data[n]*np.math.cos(2*x*np.math.pi /91 *n)
        rst.append(round(sum,3))
    return rst
def _L_generate_function(data):
    rst = []
    for x in np.arange(1.5,3.5,0.5):
        sum =0
        for n in range(0,len(data)):
            fac=np.math.pow(x,n+1)
            sum =sum + data[n]*fac/(1-fac)
        rst.append(round(sum,3))
    return rst
def generate_function(data):
    rst=[]
    rst+=poison_generate_function(data)
    rst+=L_generate_function(data)
    return rst
def peak_pkt_length_feature(_peak):
    if len(_peak) == 0:
        return [0] * 10 + [0] *17
    peak = np.mat(_peak)
    packet_length_data = list(map(lambda  x : x[0],peak[:,1].tolist()))  #只取包长
    gen_features = generate_function(packet_length_data)

    mom_features = [0] * 17
    #0-4 阶中心矩
    mom_features[0] = moment(1,packet_length_data,c=1)
    mom_features[1] = moment(2,packet_length_data,c=1)
    mom_features[2] = moment(3,packet_length_data,c=1)
    mom_features[3] = moment(4,packet_length_data,c=1)
    mom_features[4] = moment(5,packet_length_data,c=1)
    #1-3阶 原点矩
    for each in packet_length_data:
        mom_features[5] +=abs(each)
        mom_features[6] +=abs(each)**2
        mom_features[7] +=abs(each)**3
    mom_features[6] =(mom_features[6]/len(packet_length_data)) **(1/2)
    mom_features[7] =(mom_features[7]/len(packet_length_data)) **(1/3)
    #10%-90% 百分位数

    for i in range(8,17):
        mom_features[i] = length_percentile(packet_length_data,percentage=(i-8+1)*0.1)
    return gen_features + mom_features
def peak_relative_arrive_time_feature(_peak):
    if len(_peak) == 0:
        return [0] * 5
    peak = np.mat(_peak)
    arrive_time_data = list(map(lambda  x : x[0],peak[:,0].tolist()) )
    #gen_features = generate_function(arrive_time_data)
    mom_features = [0] * 5
    for i in range(0,5):
        for each in arrive_time_data:
            mom_features[i] +=each** i
        if i != 0:
            mom_features[i]= round((mom_features[i] **(1/i) ).real,3)
    return  mom_features
def peak_feature(peak):
    up_peak=[]
    down_peak=[]
    total_peak=[peak[0]]
    if total_peak[0][1] >0 :
        up_peak.append((total_peak[0][0],total_peak[0][1]))
    else:
        down_peak.append((total_peak[0][0],-total_peak[0][1]))
    for i in range(1,len(peak)):
        total_peak.append((peak[i][0]-peak[0][0],peak[i][1]))
        #上下游的包   #注意,需要验证一下 total_peak本身是否需要带负号
        if peak[i][1]>0:
            #upload的包
            up_peak.append((total_peak[i][0],total_peak[i][1]))
        else:
            #download 的包
            down_peak.append((total_peak[i][0],-total_peak[i][1]))
    features=[]
    #pkt length
    features += peak_pkt_length_feature(total_peak)
    #print('total peak pkt length feature:',len(features))
    features += peak_pkt_length_feature(up_peak)
    #print('up peak pkt length feature:',len(features))
    features += peak_pkt_length_feature(down_peak)
    #print('down peak pkt length feature:',len(features))
    #relative arrive time
    features += peak_relative_arrive_time_feature(total_peak)
    #print('total peak pkt arrive time feature:',len(features))
    features += peak_relative_arrive_time_feature(up_peak)
    #print('up peak pkt arrive time feature:',len(features))
    features += peak_relative_arrive_time_feature(down_peak)
    #print('down peak pkt arrive time feature:',len(features))
    return features

if __name__ == '__main__':
    packet_lengths=[(0,40),(0,53),(0,53),(0,1074),(0,73),(0,40),(0,217),(0,131),(0,209),(0,73),(0,40),(0,254),(0,73)]
    print(peak_pkt_length_feature(packet_lengths))



