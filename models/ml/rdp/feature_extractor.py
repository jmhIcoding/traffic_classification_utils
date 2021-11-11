__author__ = 'dk'
from models.ml.rdp.rdp_config import hyper_params
from models.ml.rdp.statistic_tractor import peak_feature

def feature_extract(pkt_size, timestamps, time_threshold= hyper_params['time_threshold'] ):
    assert len(pkt_size) == len(timestamps)
    timestamps = [0.0] + [timestamps[i]- timestamps[i-1] for i in range(1, len(timestamps)) ]
    total_peak = [(each[0], each[1]) for each in zip(timestamps, pkt_size)]
    peaks = [[]]
    for i in range(len(total_peak)):
        if total_peak[i][0] <= time_threshold :
            peaks[-1].append(total_peak[i])
        else:
            peaks.append([total_peak[i]])

    features = []
    for peak in peaks:
        feature =  peak_feature(peak)
        features.append(feature)

    return features[1:]