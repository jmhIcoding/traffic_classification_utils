import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json


PAD_KEY = 0
START_KEY = 1
END_KEY = 2


def read_file_generator(filename, max_len, keep_ratio=1):

    def gen():
        with open(filename) as fp:
            data = json.load(fp)
        data_all = []
        for exp in data:
            flow_length = len(exp['flow'])
            if flow_length <= max_len:
                flow = [START_KEY] + exp['flow'] + [END_KEY] + [PAD_KEY] * (max_len - flow_length)
                data_all.append((str.encode(exp['id']), exp['label'], flow))
        numx = 0
        total_num = min(int(keep_ratio * len(data_all)), len(data_all)-1)
        data_all = data_all[:total_num]
        #print('total_num',total_num)
        while True:
            if numx == 0:
                np.random.shuffle(data_all)
            #print('numx',numx)
            yield data_all[numx]
            numx = (numx + 1) % total_num
    return gen


def get_dataset_from_generator(file, config, max_len, keep_ratio=1):
    data_gen = read_file_generator(file, max_len, keep_ratio)
    dataset = tf.data.Dataset.from_generator(
        data_gen,
        (tf.string, tf.int32, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([max_len + 2]))
    ).shuffle(config.capacity).batch(config.batch_size).prefetch(4)
    return dataset


def _get_summary(metric):
    summ = []
    for met in metric:
        sx = tf.Summary(value=[tf.Summary.Value(tag=met, simple_value=metric[met])])
        summ.append(sx)
    return summ


def accuracy(model, val_num_batches, sess, handle, str_handle, name):
    pred_all, pred_right, losses, r_losses, c_losses = 0, 0, [], [], []
    metric = {}
    for _ in tqdm(range(val_num_batches), desc='eval', ascii=True):
        loss,\
        pred, label = sess.run(
            [model.loss,
             model.pred, model.label],
            feed_dict={handle: str_handle})
        losses.append(loss)
        #r_losses.append(r_loss)
        #c_losses.append(c_loss)
        pred_all += len(pred)
        pred_right += np.sum(pred == label)
    loss = np.mean(losses)
    metric[name + '/loss/all'] = loss
    #metric[name + '/loss/clf'] = np.mean(c_losses)
    #metric[name + '/loss/rec'] = np.mean(r_losses)
    metric[name + '/accuracy'] = pred_right / pred_all
    summ = _get_summary(metric)

    return loss, summ, metric