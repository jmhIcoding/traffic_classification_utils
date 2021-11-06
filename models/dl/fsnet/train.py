import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import model
import os
import functools

from dataset import accuracy, get_dataset_from_generator
import eval


def train(config):
    max_len = config.max_flow_length_train
    with open(config.train_meta) as fp:
        train_num = int(fp.read().strip())
    with open(config.test_meta) as fp:
        dev_num = int(fp.read().strip())
    dev_ratio = config.eval_batch * config.batch_size / dev_num
    if config.eval_batch == -1:
        config.eval_batch = dev_num // config.batch_size + 1
        dev_ratio = 1
    train_dataset = get_dataset_from_generator(config.train_json, config, max_len)
    dev_dataset = get_dataset_from_generator(config.test_json, config, max_len, dev_ratio)

    if config.decay_step == 'auto':
        config.decay_step = train_num * 2 // config.batch_size + 1
    print('[Decay Step]:', config.decay_step)
    print('[Length Num]:', config.length_num)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_app_iterator = dev_dataset.make_one_shot_iterator()
    rnn_classify = model.FSNet(config, iterator)

    for v in tf.trainable_variables():
        if v.shape.dims is None:
            print('%65s%5s' % (v.name, ' ' * 5), None)
        else:
            print('%65s%10d' % (v.name, functools.reduce(lambda x, y: x * y, v.shape)))

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    loss_step = config.loss_save
    lr = config.learning_rate

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

        train_handle = sess.run(train_iterator.string_handle())
        dev_app_handle = sess.run(dev_app_iterator.string_handle())

        sess.run(rnn_classify.train_false)
        sess.run(tf.assign(rnn_classify.lr, tf.constant(lr, dtype=tf.float32)))
        # writer.add_graph(sess.graph)

        for _ in tqdm(range(config.iter_num), ascii=True, desc='Training'):
            global_step = sess.run(rnn_classify.global_step) + 1

            loss, _, clr = sess.run([rnn_classify.loss, rnn_classify.train_op, rnn_classify.clr],
                                    feed_dict={handle: train_handle})
            if not (global_step % loss_step):  # save loss
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=loss)])
                writer.add_summary(loss_sum, global_step)

            if not (global_step % config.checkpoint):  # save model and compute train and test
                sess.run(rnn_classify.train_false)
                # compute train loss
                _, summary, metric = accuracy(rnn_classify, config.train_eval_batch, sess, handle, train_handle, 'train')
                tqdm.write('[Step={}] TRAIN batch: loss: {}, accuracy: {}'.format(
                    global_step, metric.get('train/loss/all'), metric.get('train/accuracy')))
                for s in summary:
                    writer.add_summary(s, global_step)
                # computer test loss
                loss_app, summary_app, metric = accuracy(rnn_classify, config.eval_batch, sess, handle, dev_app_handle, 'dev')
                tqdm.write('[Step={}] DEV batch: loss: {}, accuracy: {}'.format(
                    global_step, metric.get('dev/loss/all'), metric.get('dev/accuracy')))
                for s in summary_app:
                    writer.add_summary(s, global_step)
                sess.run(rnn_classify.train_true)

                lr_sum = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=clr)])
                writer.add_summary(lr_sum, global_step)
                writer.flush()

                # save model
                saver.save(sess, os.path.join(config.model_dir, 'model_%d.ckpt' % global_step))
        writer.close()


def _predict_test(sess, model, num, class_num):
    pred = [[] for _ in range(class_num)]
    real = [[] for _ in range(class_num)]
    sample_set = set()
    feature_set = {}
    for _ in tqdm(range(num), ascii=True, desc='Predict'):
        ids, preds, features = sess.run([model.ids, model.pred, model.logit])
        for idx, predx, feature in zip(ids.tolist(), preds.tolist(), features.tolist()):
            idx = idx.decode('utf-8')
            if idx in sample_set:
                continue
            sample_set.add(idx)
            real_app = int(idx.strip().split('-')[0])
            real[real_app].append(real_app)
            pred[real_app].append(predx)
            if real_app not in feature_set:
               feature_set[real_app] =[]
            feature_set[real_app].append(feature)
    import pickle
    with open('feature_set_FSNET.pkl','wb') as fp:
        pickle.dump(feature_set, fp)    
    return real, pred


def predict(config):
    test_dataset = get_dataset_from_generator(config.test_json, config, config.max_flow_length_test)
    test_dataset = test_dataset.make_one_shot_iterator()
    with open(config.test_meta) as fp:
        test_num = int(fp.read().strip())

    rnn_classify = model.FSNet(config, test_dataset, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, tf.train.latest_checkpoint(config.test_model_dir))

        sess.run(rnn_classify.train_false)
        num = test_num // config.batch_size + 1

        real, pred = _predict_test(sess, rnn_classify, num, config.class_num)
        res = eval.evaluate(real, pred)
        eval.save_res(res, os.path.join(config.pred_dir, 'FSNet.json'))
        print(json.dumps(res, indent=1, sort_keys=True))

def get_logit(config):
    test_dataset = get_dataset_from_generator(config.test_json, config, config.max_flow_length_test)
    test_dataset = test_dataset.make_one_shot_iterator()
    with open(config.test_meta) as fp:
        test_num = int(fp.read().strip())

    rnn_classify = model.FSNet(config, test_dataset, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    feature=[]
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, tf.train.latest_checkpoint(config.test_model_dir))

        sess.run(rnn_classify.train_false)
        num = test_num // config.batch_size + 1
        print(rnn_classify.flow)
        flow=sess.run(rnn_classify.flow)
        print(len(flow))
        return
        for _ in tqdm(range(num), ascii=True, desc='Predict'):
            _feature = sess.run([rnn_classify.feature])
            print(_feature)
            feature.append(_feature)
    return feature

inititial= False
rnn_classify = None
sess_config=None
sess = None
from dataset import START_KEY, PAD_KEY, END_KEY
import tensorflow as tf
def get_generator(config,flow):
    def gen():
        numx = 0
        total_num = len(flow)
        while True:
            yield ('0-0',0,flow[numx])
            numx = (numx + 1) % total_num
    return gen
def get_logit_online(config, flow):
    global  inititial,rnn_classify,sess,sess_config
    #id: label-index
    #label: 标签
    #flow: 包长序列
    for i in range(len(flow)):
        if len(flow[i]) <= config.max_packet_length:
            flow[i] = [START_KEY] + flow[i] + [END_KEY] + [PAD_KEY] * (config.max_packet_length-len(flow[i]))
            flow[i] = flow[i][:config.max_packet_length+2]
    if inititial == False:
        nums = len(flow)
        test_dataset = tf.data.Dataset.from_generator(get_generator(config,flow),
                (tf.string, tf.int32, tf.int32),
            (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([config.max_packet_length + 2]))
        ).batch(nums)
        test_dataset = test_dataset.make_one_shot_iterator()
        rnn_classify = model.FSNet(config, test_dataset, trainable=False)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        inititial=True
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, tf.train.latest_checkpoint(config.test_model_dir))
    sess.run(rnn_classify.train_false)
    feature= sess.run(rnn_classify.logit,feed_dict={"reshape/strided_slice:0": flow})
    return feature
