__author__ = 'dk'

from models.model_base import abs_model
from config import raw_dataset_base, min_flow_len
import tqdm,json
import tensorflow as tf
import os
import train
import preprocess
class model(abs_model):
    def __init__(self, dataset, randseed, splitrate ,max_len=200):
        super(model,self).__init__('fsnet',randseed= randseed)
        if os.path.exists(self.database) == False:
            os.makedirs(self.database,exist_ok=True)

        self.dataset = dataset
        self.model = self.database + '/'+ self.name + '_' + dataset + '_model'
        self.data = self.database + '/'+ self.name + '_' + dataset + '/'
        self.splitrate = splitrate
        #原始数据集目录
        full_rdata = raw_dataset_base + self.dataset
        self.full_rdata = full_rdata
        self.max_len = max_len
        if self.data_exists() == False:
            self.parser_raw_data()
    def parser_raw_data(self):
        def pad_sequence(x, max_len= self.max_len, pad_value=0):
            r =  x + [pad_value] * (max_len - len(x))
            return r[:max_len]
        full_rdata = self.full_rdata
        if os.path.exists(full_rdata) == False:
            raise OSError('Dataset {0} (full path: {1}) does not exist!'.format(self.dataset,full_rdata))
        os.makedirs(self.data, exist_ok=True)
        ##从原始数据集构建FS-Net 所需的数据集
        flow_dict ={}
        for _root, _dirs, _files in os.walk(full_rdata):
            for file in tqdm.trange(len(_files)):
                file = _files[file]
                label = file
                if label not  in flow_dict :
                    flow_dict[label] = []
                file = _root + '/' + file

                with open(file) as fp:
                    rdata = json.load(fp)

                for each in rdata :
                    pkt_size= each['packet_length']
                    if len(pkt_size) < min_flow_len :
                        continue
                    x = pad_sequence(pkt_size)
                    flow_dict[label].append(x)
        for each in flow_dict:
            with open(self.data + each + '.num','w') as fp:
                for flow in flow_dict[each]:
                    s=';{0}\t;\n'.format("\t".join([str(item) for item in flow]))
                    fp.writelines(s)

        self.fs_main(self.model, self.data, mode='prepro')
    def fs_main(self, home, data_dir, mode):
        record_dir = os.path.join(home, 'record')
        save_base = os.path.join(home, 'log')
        log_dir = os.path.join(save_base)
        pred_dir = os.path.join(home, 'result')

        for dirx in [save_base, record_dir, log_dir, pred_dir]:
            if not os.path.exists(dirx):
                os.makedirs(dirx)

        train_record = os.path.join(record_dir, 'train.json')
        test_record = os.path.join(record_dir, 'test.json')
        train_meta = os.path.join(record_dir, 'train.meta')
        test_meta = os.path.join(record_dir, 'test.meta')
        status_label = os.path.join(data_dir, 'status.label')

        flags = tf.flags

        flags.DEFINE_string('train_json', train_record, 'the processed train json file')
        flags.DEFINE_string('test_json', test_record, 'the processed test json file')
        flags.DEFINE_string('train_meta', train_meta, 'the processed train number')
        flags.DEFINE_string('test_meta', test_meta, 'the processed test number')
        flags.DEFINE_string('log_dir', log_dir, 'where to save the log')
        flags.DEFINE_string('model_dir', log_dir, 'where to save the model')
        flags.DEFINE_string('data_dir', data_dir, 'where to read data')
        flags.DEFINE_integer('class_num', self.num_classes(), 'the class number')
        flags.DEFINE_integer('length_block', 1, 'the length of a block')
        flags.DEFINE_integer('min_length', 2, 'the flow under this parameter will be filtered')
        flags.DEFINE_integer('max_packet_length', 1000, 'the largest packet length')
        flags.DEFINE_float('split_ratio', 0.8, 'ratio of train set of target app')
        flags.DEFINE_float('keep_ratio', 1, 'ratio of keeping the example (for small dataset test)')
        flags.DEFINE_integer('max_flow_length_train', 200, 'the max flow length, if larger, drop')
        flags.DEFINE_integer('max_flow_length_test', 1000, 'the max flow length, if larger, drop')
        flags.DEFINE_string('test_model_dir', log_dir, 'the model dir for test result')
        flags.DEFINE_string('pred_dir', pred_dir, 'the dir to save predict result')

        flags.DEFINE_integer('batch_size', 128, 'train batch size')
        flags.DEFINE_integer('hidden', 128, 'GRU dimension of hidden state')
        flags.DEFINE_integer('layer', 2, 'layer number of length RNN')
        flags.DEFINE_integer('length_dim', 16, 'dimension of length embedding')
        flags.DEFINE_string('length_num', 'auto', 'length_num')

        flags.DEFINE_float('keep_prob', 0.8, 'the keep probability for dropout')
        flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
        flags.DEFINE_integer('iter_num', int(0.7e5), 'iteration number')
        flags.DEFINE_integer('eval_batch', 77, 'evaluated train batches')
        flags.DEFINE_integer('train_eval_batch', 77, 'evaluated train batches')
        flags.DEFINE_string('decay_step', 'auto', 'the decay step')
        flags.DEFINE_float('decay_rate', 0.5, 'the decay rate')

        flags.DEFINE_string('mode', mode, 'model mode: train/prepro/test')
        flags.DEFINE_integer("capacity", int(1e3), "size of dataset shuffle")
        flags.DEFINE_integer("loss_save", 100, "step of saving loss")
        flags.DEFINE_integer("checkpoint", 5000, "checkpoint to save and evaluate the model")
        flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")

        flags.DEFINE_boolean('is_cudnn', True, 'whether take the cudnn gru')
        flags.DEFINE_float('rec_loss', 0.5, 'the parameter to control the reconstruction of length sequence')

        config = flags.FLAGS
        config.gpu_options.allow_growth = True
        if config.length_num == 'auto':
            config.length_num = config.max_packet_length // config.length_block + 4
        else:
            config.length_num = int(config.length_num)
        if config.decay_step != 'auto':
            config.decay_step = int(config.decay_step)
        if config.mode == 'train':
            train.train(config)
        elif config.mode == 'prepro':
            preprocess.preprocess(config)
        elif config.mode == 'test':
            print(config.test_model_dir)
            train.predict(config)
        else:
            print('unknown mode, only support train now')
            raise Exception



    def train(self):
        home = self.model
        self.fs_main(home,self.data,mode='train')
    def test(self):
        self.fs_main(home=self.model, data_dir=self.data,mode='test')
if __name__ == '__main__':
    fsnet_model = model('app60', randseed= 128, splitrate=0.1)
    fsnet_model.train()
    #df_model.test()