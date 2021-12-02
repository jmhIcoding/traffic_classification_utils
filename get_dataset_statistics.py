__author__ = 'jmh081701'
import numpy as np
import json
import os

def statistic(lengths):
    mean = np.mean(lengths)
    min = np.min(lengths)
    max = np.max(lengths)
    std = np.std(lengths)
    median = np.median(lengths)
    print('\tmean:{0}, std:{1}, min:{2}, max:{3}, median:{4}'.format(mean, std, min, max, median))
    percent = [10,20,30,40,50,60,70,80,90,95,99]
    for each_percent in percent:
        print("\t\tP( v<={1}) = {0}".format(each_percent, np.percentile(lengths,each_percent)))
def parser_dataset(dataset_dir):
    total_length = []
    print(dataset_dir)
    for _root, _dirs, _files in os.walk(dataset_dir):
        if len(_files) == 0 :
            raise  ValueError('{0} empty!'.format(dataset_dir))

        for file in _files:
            length = []
            path = _root + '/' + file
            with open(path) as fp:
                data = json.load(fp)

            for flow in data:
                length.append(len(flow['packet_length']))
                total_length.append(length[-1])

            #print(file)
            #print(statistic(length))

    print('total:')
    statistic(total_length)

if __name__ == '__main__':
    dataset_dir = 'dataset/tifs2015'
    parser_dataset(dataset_dir)