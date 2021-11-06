__author__ = 'jmh081701'
from flowcontainer.extractor import extract
import os
import json
import requests
import tqdm
import threading
def payload2packet_length(payload):
    rst = []
    i = 0
    while i < len(payload):
        rst.append(int(payload[i:i+2],base=16))
        i+= 2
    return rst

def request_label(packet):
    url = 'http://172.31.251.82:8899/datacon'
    post = {"packet_length": packet}
    response = requests.post(url=url,json=post)
    #print(response.json())
    return response.json()['label']
def traversal_training(dir):
    dataset= {}
    for _root, _dirs, _files in os.walk(dir):
        if len(_files)==0 :
            continue
        for file in tqdm.tqdm(_files):
            if  '.pcap' not  in file:
                continue
            label = file.split('_')[0]

            if label not in dataset:
                dataset[label] = []
            path = _root + '/' + file
            flows = extract(path,extension=['tcp.payload','udp.payload'], filter='tcp or udp')
            for each in tqdm.tqdm(flows, desc=file):
                flow = flows[each]
                if 'tcp.payload' in flow.extension:
                    payloads = flow.extension['tcp.payload']
                else:
                    payloads = flow.extension['udp.payload']
                for payload, index in payloads:
                    pkt_size= payload2packet_length(payload)
                    dataset[label].append({
                        "packet_length": pkt_size
                    })

    for label in dataset:
        with open('datacon/'+label + '.json', 'w') as fp:
            json.dump(dataset[label],fp)
            print('dump ', label)

def traversal_test(dir):
    log_file = 'test.log'
    rst_file = 'result.txt'
    for _root, _dirs, _files in os.walk(dir):
        if len(_files)==0 :
            continue
        for file in tqdm.tqdm(_files):
            if  '.pcap' not  in file:
                continue

            path = _root + '/' + file
            flows = extract(path,extension=['tcp.payload','udp.payload'], filter='tcp or udp')
            max_counter = 4096
            counter = 0
            label_counter = {}
            for each in tqdm.tqdm(flows, desc=file):
                flow = flows[each]
                if 'tcp.payload' in flow.extension:
                    payloads = flow.extension['tcp.payload']
                else:
                    payloads = flow.extension['udp.payload']

                packet_length = []
                for payload, index in payloads[:256]:
                    ##一个batch,一个batch的测试
                    pkt_size= payload2packet_length(payload)

                    if counter < max_counter or len(packet_length) == 0:
                        packet_length.append(pkt_size)
                        counter += 1

                _labels = request_label(packet=packet_length)
                for label in _labels:
                    if label  not  in label_counter:
                        label_counter[label] = 0
                    label_counter[label] += 1

            label_counter = list(label_counter.items())
            label_counter= sorted(label_counter, key= lambda  x: x[1])
            print('file: {0}, label counter: {1}, vote:{2}\n'.format(file, label_counter, label_counter[-1][0]))

            with open(log_file, 'a') as fp:
                fp.writelines('file: {0}, label counter: {1}, vote:{2}\n'.format(file, label_counter, label_counter[-1][0]))

            with open(rst_file,'a') as fp:
                fp.writelines('{0} {1}\n'.format(file, label_counter[-1][0]))


if __name__ == '__main__':
    #traversal_training(r'G:\chromeDownload\datacon2021_traffic_eta_part1\part1\sample')
    traversal_test(dir=r'G:\chromeDownload\datacon2021_traffic_eta_part1\part1\real_data')
