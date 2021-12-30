import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnGRU


class MultiBiGRU(object):

    def __init__(self, hidden, layer, keep_prob, is_train, is_cat=True):
        self._hidden = hidden
        self._layer = layer
        self._keep_prob = keep_prob
        self._is_train = is_train
        self._is_cat = is_cat
        self._make_multiple_layer()

    def _make_single_gru(self, hidden):
        gru_cell = tf.nn.rnn_cell.GRUCell(hidden)
        if self._is_train and self._keep_prob < 1:
            gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=self._keep_prob)
        return gru_cell

    def _make_multiple_layer(self):
        self.gru_bw = []
        self.gru_fw = []
        for layer in range(self._layer):
            self.gru_bw.append(self._make_single_gru(self._hidden))
            self.gru_fw.append(self._make_single_gru(self._hidden))

    def __call__(self, inputs, seq_len, init_fw=None, init_bw=None):
        batch_size = tf.shape(inputs)[0]
        hidden_s = inputs.shape.as_list()[-1]
        outputs = [inputs]
        output_states = []
        for layer in range(self._layer):
            gru_bw = self.gru_bw[layer]
            gru_fw = self.gru_fw[layer]
            with tf.variable_scope("bi_%d" % layer):
                if init_bw is None:
                    init_bw = tf.get_variable('init_bw', shape=[1, self._hidden], dtype=tf.float32)
                    init_bw = tf.tile(init_bw, [batch_size, 1])
                if init_fw is None:
                    init_fw = tf.get_variable('init_fw', shape=[1, self._hidden], dtype=tf.float32)
                    init_fw = tf.tile(init_fw, [batch_size, 1])
                output, output_state = tf.nn.bidirectional_dynamic_rnn(
                    gru_fw, gru_bw, outputs[-1], seq_len, dtype=tf.float32, time_major=False,
                    initial_state_bw=init_bw, initial_state_fw=init_fw
                )
            outputs.append(tf.concat(output, axis=2))
            output_states.append(tf.concat(output_state, axis=1))
        if self._is_cat:
            res = tf.concat(outputs[1:], axis=2)
            res_state = tf.concat(output_states, axis=1)
        else:
            res = outputs[-1]
            res_state = output_states[-1]

        return res_state, res


class CudaBiGRU(object):
    def __init__(self, hidden, layer, keep_prob, is_train, is_cat=True):
        self._hidden = hidden
        self._layer = layer
        self._keep_prob = keep_prob
        self._is_train = is_train
        self._is_cat = is_cat
        self._make_multiple_layer()

    def _make_single_gru(self):
        dropout = (1 - self._keep_prob) if self._is_train and self._keep_prob < 1 else 0
        gru_cell = CudnnGRU(1, self._hidden, direction='bidirectional', dropout=dropout)
        return gru_cell

    def _make_multiple_layer(self):
        self.gru = []
        for layer in range(self._layer):
            self.gru.append(self._make_single_gru())

    def __call__(self, inputs, seq_len):
        batch_size = tf.shape(inputs)[0]
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        output_states = []
        for layer in range(self._layer):
            with tf.variable_scope("bi_%d" % layer):
                init = tf.get_variable('init', shape=[2, 1, self._hidden], dtype=tf.float32)
                init = tf.tile(init, [1, batch_size, 1])
                gru = self.gru[layer]
                output, output_state = gru(outputs[-1], (init, ))
            outputs.append(output)
            output_states.append(output_state[0])
        if self._is_cat:
            res = tf.concat(outputs[1:], axis=2)
            res_state = tf.concat(output_states, axis=2)
        else:
            res = outputs[-1]
            res_state = output_states[-1]
        res = tf.transpose(res, [1, 0, 2])
        res_state = tf.reshape(tf.transpose(res_state, [1, 0, 2]), [batch_size, self._layer * 2 * self._hidden])
        return res_state, res


class FSNet(object):

    def __init__(self, config, batch_data, trainable=True):
        self.config = config
        self.batch_size = config.batch_size

        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.train_true = tf.assign(self.is_train, tf.constant(True, dtype=tf.bool))
        self.train_false = tf.assign(self.is_train, tf.constant(False, dtype=tf.bool))
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.ids, self.label, self.flow = batch_data.get_next()

        print(self.flow)

        self._gru = CudaBiGRU if config.is_cudnn else MultiBiGRU
        # get best batch shape
        with tf.variable_scope('reshape'):
            self.mask = tf.cast(self.flow, tf.bool)
            self.len = tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1)
            self.max_len = tf.reduce_max(self.len)

            self.flow = self.flow[:, 0: self.max_len]
            self.mask = self.mask[:, 0: self.max_len]

        self.loss, self.pred = self._make_graph()

        if trainable:
            self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            self.clr = tf.train.exponential_decay(self.lr, self.global_step,
                                                  self.config.decay_step, self.config.decay_rate, staircase=True)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                     global_step=self.global_step)

    def _embedding(self, emb_dim, vac_num, inputs, scope='embedding'):
        with tf.variable_scope(scope):
            embedding = tf.get_variable('embedding', dtype=tf.float32, shape=[vac_num, emb_dim])
            seq = tf.nn.embedding_lookup(embedding, inputs)
        return seq

    def _encoder(self, hidden, layer, seq, scope='encoder'):
        with tf.variable_scope(scope):
            if self.is_train and self.config.keep_prob < 1:
                seq = tf.nn.dropout(seq, self.config.keep_prob)
            gru = self._gru(hidden, layer, self.config.keep_prob, self.is_train)
            feature, outputs = gru(seq, self.len)
        return feature, outputs

    def _decoder_input(self, feature, seq):
        feature_input = tf.tile(tf.expand_dims(feature, axis=1), [1, self.max_len, 1])
        return feature_input

    def _decoder(self, hidden, layer, inputs, scope='decoder'):
        with tf.variable_scope(scope):
            gru = self._gru(hidden, layer, self.config.keep_prob, self.is_train)
            feature, outputs = gru(inputs, self.len)
        return feature, outputs

    def _fusion(self, e_fea, d_fea, scope='fusion'):
        hidden = e_fea.shape.as_list()[-1]
        with tf.variable_scope(scope):
            fea = tf.concat([e_fea, d_fea, e_fea * d_fea], 1)
            if self.is_train and self.config.keep_prob < 1:
                fea = tf.nn.dropout(fea, self.config.keep_prob)
            g = tf.layers.dense(fea, hidden, activation=tf.nn.sigmoid, name='gate')
            update_ = tf.layers.dense(fea, hidden, activation=tf.nn.tanh, name='update')
        return e_fea * g + (1 - g) * update_

    def _compress(self, feature, scope='compress'):
        with tf.variable_scope(scope):
            ff_ = tf.layers.dense(feature, 2 * self.config.hidden, use_bias=True, activation=tf.nn.selu, name='W1')
            if self.is_train and self.config.keep_prob < 1:
                ff_ = tf.nn.dropout(ff_, self.config.keep_prob)
        return ff_

    def _reconstruct(self, inputs, vac_num, label, mask, scope='rec'):
        with tf.variable_scope(scope):
            logits = tf.layers.dense(inputs, self.config.hidden, activation=tf.nn.selu)
            logits = tf.layers.dense(logits, vac_num)
            logits = tf.reshape(logits, [-1, vac_num])
            loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(label, [-1]))
            mask = tf.cast(tf.reshape(mask, [-1]), dtype=tf.float32)
            loss = tf.reduce_sum(loss_all * mask) / tf.reduce_sum(mask)
        return loss

    def _classify(self, feature):
        with tf.variable_scope('classify'):
            logit = tf.layers.dense(feature, self.config.class_num, use_bias=True)
            self.logit = logit
            pred = tf.argmax(logit, axis=1)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.label)
            loss = tf.reduce_mean(loss)
        return loss, pred

    def _make_graph(self):
        rec_loss = 0
        seq = self._embedding(self.config.length_dim, self.config.length_num, self.flow)
        e_fea, in_output = self._encoder(self.config.hidden, self.config.layer, seq)
        dec_input = self._decoder_input(e_fea, seq)
        d_fea, l_output = self._decoder(self.config.hidden, self.config.layer, dec_input)
        rec_loss += self._reconstruct(l_output, self.config.length_num, self.flow, self.mask)
        feature = tf.concat([e_fea, d_fea], axis=1)
        feature = self._compress(feature)
        self.feature = feature
        c_loss, pred = self._classify(feature)
        return c_loss, pred
        loss = c_loss + self.config.rec_loss * rec_loss
        self.c_loss = c_loss
        self.rec_loss = self.config.rec_loss * rec_loss
        return loss, pred
