__author__ = 'jmh081701'
import sys
import os
import json
import numpy as np
from models.ml.rdp import statistic_tractor
def read_txt(filename):
    '''
    :param filename:  text file the record packet length and timestamp.
    :return:
        client to server : upload traffic, length > 0
        server to client : download traffic , length <0
    '''
    with open(filename,"r") as fp:
        while True:
            line=fp.readline()
            if line:
                rst=[]
                line = line.split(",")
                line = line[0:3]+line[-2:-1]
                rst.append(float(line[0]))
                rst.append(int(line[3],10))
                try:
                    if line[1].count('107c') > 0:
                        #client to server   > 0
                        rst[1]*=1
                    else:
                        #server to client   < 0
                        rst[1]*= -1
                except:
                    if str(line[1]).count("124.16.") > 0:
                        #client to server >0
                        rst[1] *=1
                    else:
                        #server to client < 0
                        rst[0] *=-1
                yield  rst
            else:
                break
def gather_peak_up_down(filename,gap=0.1):
    file_reader = read_txt(filename)
    peaks=[[[0,0]]]
    index = 0
    for each in file_reader:
        if abs(peaks[index][-1][0]-each[0])<gap:
            peaks[index].append(each)
        else:
            peaks.append([each])
            index+=1
    return peaks[1:]# delete the first one item
def gather_peak_up(filename,gap=0.1):
    file_reader = read_txt(filename)
    peaks=[[[0,0]]]
    index = 0
    for each in file_reader:
        if each[0]>0 and abs(peaks[index][-1][0]-each[0])<gap:
            peaks[index].append(each)
        else:
            peaks.append([each])
            index+=1
    return peaks[1:]# delete the first one item
def gather_peak_down(filename,gap=0.1):
    file_reader = read_txt(filename)
    peaks=[[[0,0]]]
    index = 0
    for each in file_reader:
        if each[0]<0 and abs(peaks[index][-1][0]-each[0])<gap:
            peaks[index].append(each)
        else:
            peaks.append([each])
            index+=1
    return peaks[1:]# delete the first one item
if __name__ == '__main__':
    #gather the max peak length
    appnames=['micrords','anydesk','realvnc','teamviewer']
    #gaps=[0.2,0.5,0.8,1,1.5]
    gaps=[0.5,0.2,0.8]
    last_length=-1
    for gap in gaps:
        for app in appnames:
            dir=r"E:\TempWorkStation\i-know-what-are-you-doing\pcap\%s"%app
            peaks_length=[]
            packet_number_peak_length=[]
            for root,subdirs,files in os.walk(dir):
                for sub in subdirs:
                    if sub not in ['watching_video','reading_doc','editing_doc','installing_software','surfing_web','transfering_file']:
                        #'watching_video','reading_doc','editing_doc','installing_software','surfing_web','transfering_file'
                        continue
                    for _root,_subdirs,_files in os.walk(dir+"\\"+sub):
                        features =[]
                        timestamps=[]
                        counter=0
                        flowids=[]
                        for each in _files:
                            if each.count(".txt"):
                                peaks=gather_peak_up_down(dir+"\\"+sub+"\\"+each,gap)
                                if len(peaks)==0:
                                    print(peaks,each)
                                    continue
                                flowid = int(each.split(".")[0])
                                peaks_length.append(len(peaks))
                                packet_number_peak_length+=map(lambda  x: len(x),peaks)
                                for peak in peaks:
                                    feature = statistic_tractor.peak_feature(peak)
                                    if last_length!=-1 and len(feature)!=last_length:
                                        print(feature,peak)
                                        exit(0)
                                    if last_length==-1:
                                        last_length =len(feature)
                                    features.append(feature)
                                    timestamps.append(peak[0][0])
                                    flowids.append(flowid)
                                    counter +=1
                            #exit(0)
                        with open("E:\\TempWorkStation\\i-know-what-are-you-doing\\dataset\\vector_flowid\\%s_%s.data.gap=%s"%(app,sub,str(gap)),"w") as fp:
                            print("E:\\TempWorkStation\\i-know-what-are-you-doing\\dataset\\vector_flowid\\%s_%s.data  : %d"%(app,sub,counter))
                            json.dump({'feature':features,'timestamps':timestamps,'counter':counter,'flowids':flowids},fp)
                            del features
                            del timestamps
                            del counter
                            del flowids
        #        print(np.min(peaks_length),np.max(peaks_length),np.std(peaks_length),np.average(peaks_length))
        #        print(np.min(packet_number_peak_length),np.max(packet_number_peak_length),np.std(packet_number_peak_length),np.average(packet_number_peak_length))
