#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    模型: bi-lstm + crf
"""
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import uniform_tensor, get_sequence_actual_length, \
    zero_nil_slot, shuffle_matrix
import tensorflow.contrib.seq2seq as tc_seq2seq


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def get_activation(activation=None):
    """
    Get activation function accord to the parameter 'activation'
    Args:
        activation: str: 激活函数的名称
    Return:
        激活函数
    """
    if activation is None:
        return None
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'softmax':
        return tf.nn.softmax
    elif activation == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception('Unknow activation function: %s' % activation)


class MultiConvolutional3D(object):

    def __init__(self, input_data, filter_length_list, nb_filter_list, padding='VALID',
                 activation='relu', pooling='max', name='Convolutional3D'):
        """3D卷积层
        Args:
            input_data: 4D tensor of shape=[batch_size, sent_len, word_len, char_dim]
                in_channels is set to 1 when use Convolutional3D.
            filter_length_list: list of int, 卷积核的长度，用于构造卷积核，在
                Convolutional1D中，卷积核shape=[filter_length, in_width, in_channels, nb_filters]
            nb_filter_list: list of int, 卷积核数量
            padding: 默认'VALID'，暂时不支持设成'SAME'
        """
        assert padding in ('VALID'), 'Unknow padding %s' % padding
        # assert padding in ('VALID', 'SAME'), 'Unknow padding %s' % padding

        # expand dim
        char_dim = int(input_data.get_shape()[-1])  # char的维度
        self._input_data = tf.expand_dims(input_data, -1)  # shape=[x, x, x, 1]
        self._filter_length_list = filter_length_list
        self._nb_filter_list = nb_filter_list
        self._padding = padding
        self._activation = get_activation(activation)
        self._name = name

        pooling_outpouts = []
        for i in range(len(self._filter_length_list)):
            filter_length = self._filter_length_list[i]
            nb_filter = self._nb_filter_list[i]
            with tf.variable_scope('%s_%d' % (name, filter_length)) as scope:
                # shape= [batch_size, sent_len-filter_length+1, word_len, 1, nb_filters]
                conv_output = tf.contrib.layers.conv3d(
                    inputs=self._input_data,
                    num_outputs=nb_filter,
                    kernel_size=[1, filter_length, char_dim],
                    padding=self._padding)
                # output's shape=[batch_size, new_height, 1, nb_filters]
                act_output = (
                    conv_output if activation is None
                    else self._activation(conv_output))
                # max pooling，shape = [batch_size, sent_len, nb_filters]
                if pooling == 'max':
                    pooling_output = tf.reduce_max(tf.squeeze(act_output, [-2]), 2)
                elif pooling == 'mean':
                    pooling_output = tf.reduce_mean(tf.squeeze(act_output, [-2]), 2)
                else:
                    raise Exception('pooling must in (max, mean)!')
                pooling_outpouts.append(pooling_output)

                scope.reuse_variables()
        # [batch_size, sent_len, sum(nb_filter_list]
        self._output = tf.concat(pooling_outpouts, axis=-1)

    @property
    def output(self):
        return self._output

    @property
    def output_dim(self):
        return sum(self._nb_filter_list)


class SequenceLabelingModel2(object):

    def __init__(self, sequence_length, nb_classes, nb_hidden=512, num_layers=1,
                 rnn_dropout=0., feature_names=None, feature_init_weight_dict=None,
                 feature_weight_shape_dict=None, feature_weight_dropout_dict=None,
                 dropout_rate=0., use_crf=True, path_model=None, nb_epoch=200,
                 batch_size=128, train_max_patience=10, l2_rate=0.01, rnn_unit='lstm',
                 learning_rate=0.001, clip=None, use_char_feature=False, word_length=None,
                 conv_filter_size_list=None, conv_filter_len_list=None, cnn_dropout_rate=0.):
        """
        Args:
          sequence_length: int, 输入序列的padding后的长度
          nb_classes: int, 标签类别数量
          nb_hidden: int, lstm/gru层的结点数
          num_layers: int, lstm/gru层数
          rnn_dropout: lstm层的dropout值

          feature_names: list of str, 特征名称集合
          feature_init_weight_dict: dict, 键:特征名称, 值:np,array, 特征的初始化权重字典
          feature_weight_shape_dict: dict，特征embedding权重的shape，键:特征名称, 值: shape(tuple)。
          feature_weight_dropout_dict: feature name to float, feature weights dropout rate

          dropout: float, dropout rate
          use_crf: bool, 标示是否使用crf层
          path_model: str, 模型保存的路径
          nb_epoch: int, 训练最大迭代次数
          batch_size: int
          train_max_patience: int, 在dev上的loss对于train_max_patience次没有提升，则early stopping

          l2_rate: float

          rnn_unit: str, lstm or gru
          learning_rate: float, default is 0.001
          clip: None or float, gradients clip

          use_char_feature: bool,是否使用字符特征
          word_length: int, 单词长度
        """
        self._sequence_length = sequence_length
        self._nb_classes = nb_classes
        self._nb_hidden = nb_hidden
        self._num_layers = num_layers
        self._rnn_dropout = rnn_dropout

        self._feature_names = feature_names


        self._feature_init_weight_dict = feature_init_weight_dict if \
            feature_init_weight_dict else dict()
        self._feature_weight_shape_dict = feature_weight_shape_dict
        self._feature_weight_dropout_dict = feature_weight_dropout_dict

        self._dropout_rate = dropout_rate
        self._use_crf = use_crf

        self._path_model = path_model
        self._nb_epoch = nb_epoch
        self._batch_size = batch_size
        self._train_max_patience = train_max_patience

        self._l2_rate = l2_rate
        self._rnn_unit = rnn_unit
        self._learning_rate = learning_rate
        self._clip = clip

        self._use_char_feature = use_char_feature
        self._word_length = word_length
        self._conv_filter_len_list = conv_filter_len_list
        self._conv_filter_size_list = conv_filter_size_list
        self._cnn_dropout_rate = cnn_dropout_rate

        assert len(feature_names) == len(list(set(feature_names))), \
            'duplication of feature names!'

        # init ph, weights and dropout rate
        self.input_feature_ph_dict = dict()
        self.weight_dropout_ph_dict = dict()
        self.feature_weight_dict = dict()
        self.nil_vars = set()
        self.dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate_ph')
        self.rnn_dropout_rate_ph = tf.placeholder(tf.float32, name='rnn_dropout_rate_ph')

        self.intent_weight_ph = tf.placeholder(tf.float32, name='intent_weight')

        # label ph
        self.input_label_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, self._sequence_length], name='input_label_ph')
        self.input_label_intent = tf.placeholder(
            dtype=tf.int32, shape=[None], name='input_label_intent'
        )
        if self._use_char_feature:
            self.cnn_dropout_rate_ph = tf.placeholder(tf.float32, name='cnn_dropout_rate_ph')

        self.build_model()

    @staticmethod
    def merge_bi_rnn_state(bi_state):
        encoder_fw_state = bi_state[0][0]
        encoder_bw_state = bi_state[1][0]
        if isinstance(encoder_fw_state, rnn.LSTMStateTuple):  # LstmCell
            state_c = tf.concat(
                (encoder_fw_state.c, encoder_bw_state.c), 1, name="bidirectional_concat_c")
            state_h = tf.concat(
                (encoder_fw_state.h, encoder_bw_state.h), 1, name="bidirectional_concat_h")
            final_state = rnn.LSTMStateTuple(c=state_c, h=state_h)
        else:
            raise ValueError("RNN state type error")
            # final_state = tf.concat(
            #     (encoder_fw_state, encoder_bw_state), 1, name="bidirectional_state_concat")
        return final_state

    def build_model(self):
        for feature_name in self._feature_names:

            # input ph
            self.input_feature_ph_dict[feature_name] = tf.placeholder(
                dtype=tf.int32, shape=[None, self._sequence_length],
                name='input_feature_ph_%s' % feature_name)

            # dropout rate ph
            self.weight_dropout_ph_dict[feature_name] = tf.placeholder(
                tf.float32, name='dropout_ph_%s' % feature_name)

            # init feature weights, 初始化未指定的
            if feature_name not in self._feature_init_weight_dict:
                feature_weight = uniform_tensor(
                    shape=self._feature_weight_shape_dict[feature_name],
                    name='f_w_%s' % feature_name)
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=feature_weight, name='feature_weigth_%s' % feature_name)
            else:
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=self._feature_init_weight_dict[feature_name],
                    name='feature_weight_%s' % feature_name)
                self.nil_vars.add(self.feature_weight_dict[feature_name].name)

            # init dropout rate, 初始化未指定的
            if feature_name not in self._feature_weight_dropout_dict:
                self._feature_weight_dropout_dict[feature_name] = 0.
        # char feature
        if self._use_char_feature:
            # char feature weights
            feature_weight = uniform_tensor(
                shape=self._feature_weight_shape_dict['char'], name='f_w_%s' % 'char')
            self.feature_weight_dict['char'] = tf.Variable(
                initial_value=feature_weight, name='feature_weigth_%s' % 'char')
            self.nil_vars.add(self.feature_weight_dict['char'].name)
            self.nil_vars.add(self.feature_weight_dict['char'].name)
            self.input_feature_ph_dict['char'] = tf.placeholder(
                dtype=tf.int32, shape=[None, self._sequence_length, self._word_length],
                name='input_feature_ph_%s' % 'char')

        # init embeddings
        self.embedding_features = []
        for feature_name in self._feature_names:
            print(self.input_feature_ph_dict[feature_name].shape)
            embedding_feature = tf.nn.dropout(tf.nn.embedding_lookup(
                self.feature_weight_dict[feature_name],
                ids=self.input_feature_ph_dict[feature_name],
                name='embedding_feature_%s' % feature_name),
                keep_prob=1. - self.weight_dropout_ph_dict[feature_name],
                name='embedding_feature_dropout_%s' % feature_name)
            self.embedding_features.append(embedding_feature)
            # print(feature_name +' shape',embedding_feature.shape)
            # print(feature_name + ' weight shape', self.feature_weight_dict[feature_name].shape)
            # print(feature_name + ' feature shape', self.input_feature_ph_dict[feature_name].shape)
        # char embedding
        if self._use_char_feature:
            char_embedding_feature = tf.nn.embedding_lookup(
                self.feature_weight_dict['char'],
                ids=self.input_feature_ph_dict['char'],
                name='embedding_feature_%s' % 'char')
            # conv
            couv_feature_char = MultiConvolutional3D(
                char_embedding_feature, filter_length_list=self._conv_filter_len_list,
                nb_filter_list=self._conv_filter_size_list).output
            couv_feature_char = tf.nn.dropout(
                couv_feature_char, keep_prob=1 - self.cnn_dropout_rate_ph)

        # concat all features
        print(self.embedding_features)
        print(self._feature_names)
        # input_features = self.embedding_features[0] if len(self.embedding_features) == 1 \
        #     else tf.concat(values=self.embedding_features, axis=len(self._feature_names), name='input_features')
        input_features = self.embedding_features[0] if len(self.embedding_features) == 1 \
            else tf.concat(values=self.embedding_features, axis=2, name='input_features')
        print('input features shape', input_features.shape)

        if self._use_char_feature:
            input_features = tf.concat([input_features, couv_feature_char], axis=-1)

        # multi bi-lstm layer
        _fw_cells = []
        _bw_cells = []
        for _ in range(self._num_layers):
            fw, bw = self._get_rnn_unit(self._rnn_unit)
            _fw_cells.append(tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=1 - self.rnn_dropout_rate_ph))
            _bw_cells.append(tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=1 - self.rnn_dropout_rate_ph))
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(_fw_cells)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(_bw_cells)

        # 计算self.input_features[feature_names[0]]的实际长度(0为padding值)
        self.sequence_actual_length = get_sequence_actual_length(  # 每个句子的实际长度
            self.input_feature_ph_dict[self._feature_names[0]], dim=1)

        print(self.sequence_actual_length.shape)
        input_size = input_features.shape[-1]
        print('input_features shape ',input_features.shape)
        rnn_inputs = tf.reshape(input_features, [-1, self._sequence_length, input_size])
        print('rnn inputs shape ', rnn_inputs.shape)
        rnn_lengths = tf.reshape(self.sequence_actual_length, [-1])

        # todo: add encoder output
        rnn_outputs, rnn_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, rnn_inputs, scope='bi-lstm',
            dtype=tf.float32, sequence_length=rnn_lengths)

        # shape = [batch_size, max_len, nb_hidden*2]
        rnn_outputs = tf.concat(rnn_outputs, axis=2, name='lstm_output')
        rnn_outputs = tf.nn.dropout(
            rnn_outputs,
            keep_prob=1. - self.dropout_rate_ph, name='lstm_output_dropout')
        rnn_hidden = self.merge_bi_rnn_state(rnn_state).h
        #
        batch_size = tf.shape(input_features)[0]
        #
        print('rnn outputs shape', rnn_outputs.shape)
        print('rnn hidden shape', rnn_hidden.shape)
        #
        # rnn_outputs = tf.reshape(rnn_outputs,
        #                          [batch_size, turn_size, self._sequence_length, self._nb_hidden * 2])
        #
        # rnn_hidden = tf.reshape(rnn_hidden, [batch_size, turn_size, self._nb_hidden * 2])
        # # rnn_hidden = tf.nn.dropout(rnn_hidden, keep_prob=1. - self.dropout_rate_ph)
        # print('rnn outputs shape', rnn_outputs.shape)
        # print('rnn hidden shape', rnn_hidden.shape)
        #
        # # context rnn
        # ctx_cell = rnn.BasicLSTMCell(self._nb_hidden * 2, forget_bias=1., state_is_tuple=True)
        # ctx_lengths = get_sequence_actual_length(self.input_feature_ph_dict[self._feature_names[0]], dim=[1, 2])
        # print("ctx inputs shape", rnn_hidden.shape)
        # print('ctx lengths shape', ctx_lengths.shape)
        #
        # ctx_outputs, _ = tf.nn.dynamic_rnn(cell=ctx_cell,
        #                                    inputs=rnn_hidden,
        #                                    sequence_length=ctx_lengths,
        #                                    dtype=tf.float32)
        # # predict intents
        intent_logits = tf.layers.dense(rnn_hidden, 24, activation=None) #!!!!
        label_intents = tf.reshape(self.input_label_intent, [-1])
        print('intent_logits shape', intent_logits.shape)
        print('input_label_intent shape',self.input_label_intent.shape)
        intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_intents,
            logits=intent_logits)
        intent_mask = tf.reshape(tf.sign(self.sequence_actual_length), [-1])
        intent_mask = tf.cast(intent_mask, dtype=tf.float32)
        print('intent_mask shape', intent_mask.shape)
        self.intent_loss = tf.reduce_sum(intent_loss * intent_mask) / tf.reduce_sum(intent_mask)

        pred_intents = tf.argmax(intent_logits, axis=1)
        self.pred_intents = tf.reshape(pred_intents, [-1])
        print('pred_intents shape', self.pred_intents.shape)
        correct_preds = tf.equal(tf.cast(pred_intents, dtype=tf.int32),
                                 tf.cast(label_intents, dtype=tf.int32))

        self.intent_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32) * intent_mask) \
                               / tf.reduce_sum(intent_mask)

        self.intent_count = tf.cast(tf.reduce_sum(intent_mask), tf.int32)

        self.intent_logits = intent_logits

        # predict slots
        # batch * turn * hidden
        # print('ctx outputs shape', ctx_outputs.shape)
        # ctx_outputs = tf.reshape(ctx_outputs, [batch_size, turn_size, self._nb_hidden * 2])
        # rnn_intent_outputs = [ctx_outputs for _ in range(self._sequence_length)]
        # rnn_intent_outputs = tf.stack(rnn_intent_outputs, axis=2)
        # print('rnn intent outputs', rnn_intent_outputs.shape)
        #
        # ctx_h = tf.reshape(ctx_outputs[:, :, :self._nb_hidden], [-1, self._nb_hidden])
        # ctx_c = tf.reshape(ctx_outputs[:, :, self._nb_hidden:], [-1, self._nb_hidden])
        #
        # init_fw_hidden = []
        # init_bw_hidden = []
        # for _ in range(self._num_layers):
        #     lstm_hidden = rnn.LSTMStateTuple(h=ctx_h, c=ctx_c)
        #     init_fw_hidden += [lstm_hidden]
        #     init_bw_hidden += [lstm_hidden]
        # init_fw_hidden = tuple(init_fw_hidden)
        # init_bw_hidden = tuple(init_bw_hidden)

        # slot_outputs = tf.concat([rnn_outputs, rnn_intent_outputs], axis=3)
        # slot_outputs = tf.reshape(slot_outputs, [-1, self._nb_hidden * 4])

        # run the rnn again with init state

        # slot_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        #     fw_cell, bw_cell, rnn_inputs,
        #     scope='bi-lstm',
        #     initial_state_fw=init_fw_hidden,
        #     initial_state_bw=init_bw_hidden,
        #     dtype=tf.float32,
        #     sequence_length=rnn_lengths)

        # slot_outputs = tf.concat(slot_outputs, axis=2, name='slot_output')

        slot_logits = tf.layers.dense(rnn_outputs, self._nb_classes, activation=None)
        self.slot_logits = tf.reshape(slot_logits, [batch_size, self._sequence_length, self._nb_classes])
        print('slot logits shape', self.slot_logits.shape)
        slot_labels = tf.reshape(self.input_label_ph, [-1, self._sequence_length])
        slot_logits = tf.reshape(self.slot_logits, [-1, self._sequence_length, self._nb_classes])
        slot_lengths = tf.reshape(self.sequence_actual_length, [-1])
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            slot_logits, slot_labels, slot_lengths)
        print('transition params shape', self.transition_params.shape)
        print('log likelihood loss', log_likelihood.shape)
        self.slot_loss = tf.reduce_sum(-log_likelihood * intent_mask) / tf.reduce_sum(intent_mask)
        self.total_loss = self.intent_loss + self.slot_loss
        self.train_loss = self.slot_loss + self.intent_loss * self.intent_weight_ph

        # train op
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self.train_op = optimizer.minimize(self.train_loss)
        grads_and_vars = optimizer.compute_gradients(self.train_loss)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._clip:
            # clip by global norm
            gradients, variables = zip(*nil_grads_and_vars)
            gradients, _ = tf.clip_by_global_norm(gradients, self._clip)
            self.train_op = optimizer.apply_gradients(
                zip(gradients, variables), name='train_op', global_step=global_step)
        else:
            self.train_op = optimizer.apply_gradients(
                nil_grads_and_vars, name='train_op', global_step=global_step)

        # TODO sess, visible_device_list待修改
        gpu_options = tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # init all variable
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _get_rnn_unit(self, rnn_unit):
        if rnn_unit == 'lstm':
            fw_cell = rnn.BasicLSTMCell(self._nb_hidden, forget_bias=1., state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(self._nb_hidden, forget_bias=1., state_is_tuple=True)
        elif rnn_unit == 'gru':
            fw_cell = rnn.GRUCell(self._nb_hidden)
            bw_cell = rnn.GRUCell(self._nb_hidden)
        else:
            raise ValueError('rnn_unit must in (lstm, gru)!')
        return fw_cell, bw_cell

    def evaluate_slot(self, pred_slots, gold_slots, slot_lengths):
        correct = 0
        count = 0
        batch_size = slot_lengths.shape[0]
        for i in range(batch_size):
            length = slot_lengths[i]
            if length <= 0:
                continue
            count += 1
            pred_slot = pred_slots[i]
            gold_slot = gold_slots[i][:length].tolist()
            flag = True
            assert len(pred_slot) == len(gold_slot)
            for pred_w, gold_w in zip(pred_slot, gold_slot):
                if pred_w != gold_w:
                    flag = False
                    break
            if flag:
                correct += 1
        return correct, count

        pass

    def fit(self, data_dict, dev_size=0.2, seed=1337):
        """
        训练
        Args:
            data_dict: dict, 键: 特征名(or 'label'), 值: np.array
            dev_size: float, 开发集所占的比例，default is 0.2

            batch_size: int
            seed: int, for shuffle data
        """
        data_train_dict, data_dev_dict = self.split_train_dev(data_dict, dev_size=dev_size)
        self.saver = tf.train.Saver()  # save model

        train_iter = Iterator(data_train_dict, self._feature_names, self._batch_size, self._sequence_length,
                              shuffle=True, infer=False)
        dev_iter = Iterator(data_dev_dict, self._feature_names, self._batch_size, self._sequence_length,
                            shuffle=False, infer=False)

        print('-' * 100)
        print('.....begin training')
        print('.....train samples', train_iter.n_samples)
        print('.....dev samples', dev_iter.n_samples)

        nb_train = int(math.ceil(train_iter.size / float(self._batch_size)))

        min_dev_loss = 0  # 全局最小dev loss, for early stopping)
        current_patience = 0  # for early stopping
        for step in range(self._nb_epoch):
            print('Epoch %d / %d:' % (step + 1, self._nb_epoch))
            # train
            train_loss, l2_loss, all_intent_loss = 0., 0., 0.
            with tqdm(total=train_iter.n_samples) as pbar:
                train_acc, train_samples = 0., 0
                train_slot_correct = 0
                for batch_iter in train_iter.next():
                    feed_dict = dict()
                    for feature_name in self._feature_names:  # features
                        # feature
                        batch_data = batch_iter[feature_name]
                        item = {self.input_feature_ph_dict[feature_name]: batch_data}
                        feed_dict.update(item)
                        # dropout
                        dropout_rate = self._feature_weight_dropout_dict[feature_name]
                        item = {self.weight_dropout_ph_dict[feature_name]: dropout_rate}
                        feed_dict.update(item)

                    feed_dict.update(
                        {
                            self.dropout_rate_ph: self._dropout_rate,
                            self.rnn_dropout_rate_ph: self._rnn_dropout,
                        })
                    # label feed
                    batch_intent = batch_iter['intent']
                    batch_label = batch_iter['slot']
                    feed_dict.update({
                        self.input_label_ph: batch_label,
                        self.input_label_intent: batch_intent,
                        self.intent_weight_ph: max(0.1, 1.0 - step / 10)
                    })
                    # print(batch_label.shape)
                    _, loss, intent_loss, acc, batch_count, intent_preds, slot_logits, trans_params, seq_lengths \
                        = self.sess.run([self.train_op,
                                         self.total_loss,
                                         self.intent_loss,
                                         self.intent_accuracy,
                                         self.intent_count,
                                         self.pred_intents,
                                         self.slot_logits,
                                         self.transition_params,
                                         self.sequence_actual_length],
                                        feed_dict=feed_dict)
                    pred_slots = self.crf_infer(slot_logits, seq_lengths, trans_params)
                    slot_correct, slot_count = self.evaluate_slot(pred_slots, batch_iter['slot'], seq_lengths)

                    train_loss += loss
                    all_intent_loss += intent_loss
                    train_acc += batch_count * acc
                    train_samples += batch_count
                    train_slot_correct += slot_correct
                    pbar.update(batch_count)

            train_loss /= float(nb_train)
            train_acc /= train_samples
            intent_loss = all_intent_loss / float(nb_train)
            train_slot_acc = train_slot_correct / train_samples

            # 计算在开发集上的loss
            dev_loss, dev_intent_loss, dev_acc, dev_samples, dev_slot_acc = self.evaluate(dev_iter)

            print('train loss: %f, train intent loss: %f, dev loss: %f, dev intent loss: %f' % (
                train_loss, intent_loss, dev_loss, dev_intent_loss))
            print('train intent accuracy: %f(%d), dev intent accuracy: %f(%d)'
                  % (train_acc, train_samples, dev_acc, dev_samples))
            print('train slot accuracy: %f(%d), dev slot accuracy: %f(%d)'
                  % (train_slot_acc, train_samples, dev_slot_acc, dev_samples))

            # 根据dev上的表现保存模型
            if not self._path_model:
                continue
            if dev_acc + dev_slot_acc > min_dev_loss:
                min_dev_loss = (dev_acc + dev_slot_acc)
                current_patience = 0
                # save model
                self.saver.save(self.sess, self._path_model)
                print('model has saved to %s!' % self._path_model)
            else:
                current_patience += 1
                print('no improvement, current patience: %d / %d' %
                      (current_patience, self._train_max_patience))
                if self._train_max_patience and current_patience >= self._train_max_patience:
                    print('\nfinished training! (early stopping, max patience: %d)'
                          % self._train_max_patience)
                    return
        print('\nfinished training!')
        return

    def split_train_dev(self, data_dict: dict, dev_size=0.2):
        """
        划分为开发集和测试集
        Args:
            data_dict: dict, 键: 特征名(or 'label'), 值: np.array
            dev_size: float, 开发集所占的比例，default is 0.2
        Returns:
            data_train_dict, data_dev_dict: same type as data_dict
            :type dev_size: float
        """
        import random
        data_train_dict, data_dev_dict = dict(), dict()
        for key in data_dict:
            if random.random() < dev_size:
                data_dev_dict[key] = data_dict[key]
            else:
                data_train_dict[key] = data_dict[key]
        print("train session samples", len(data_train_dict))
        print("dev session samples", len(data_dev_dict))
        return data_train_dict, data_dev_dict

    def evaluate(self, dev_iter):
        """
        计算loss
        Args:
            dev_iter(Iterator): Iterator
        Return:
            loss: float
        """
        eval_loss = 0.
        intent_loss = 0.0
        n_samples = 0
        accuracy = 0
        dev_correct = 0
        for batch_iter in dev_iter.next():
            feed_dict = dict()
            for feature_name in self._feature_names:  # features
                # feature
                batch_data = batch_iter[feature_name]
                item = {self.input_feature_ph_dict[feature_name]: batch_data}
                feed_dict.update(item)
                # dropout
                item = {self.weight_dropout_ph_dict[feature_name]: 0.0}
                feed_dict.update(item)

            feed_dict.update(
                {
                    self.dropout_rate_ph: 0.0,
                    self.rnn_dropout_rate_ph: 0.0,
                })
            # label feed
            batch_intent = batch_iter['intent']
            batch_label = batch_iter['slot']
            feed_dict.update({
                self.input_label_ph: batch_label,
                self.input_label_intent: batch_intent})

            loss, intent_loss, acc, batch_count, slot_logits, trans_params, seq_lengths = \
                self.sess.run([self.total_loss,
                               self.intent_loss,
                               self.intent_accuracy,
                               self.intent_count,
                               self.slot_logits,
                               self.transition_params,
                               self.sequence_actual_length],
                              feed_dict=feed_dict)
            pred_slots = self.crf_infer(slot_logits, seq_lengths, trans_params)
            correct, slot_count = self.evaluate_slot(pred_slots, batch_iter['slot'], seq_lengths)

            eval_loss += loss * batch_count
            intent_loss += intent_loss * batch_count
            accuracy += acc * batch_count
            n_samples += batch_count
            assert slot_count == batch_count
            dev_correct += correct
        eval_loss /= n_samples
        intent_loss /= n_samples
        accuracy /= n_samples
        dev_slot_acc = dev_correct / n_samples
        print("dev samples", n_samples)
        return eval_loss, intent_loss, accuracy, n_samples, dev_slot_acc

    def crf_infer(self, slot_logits, slot_lengths, trans_params):
        batch_size = slot_lengths.shape[0]
        # print('slot_logits shape',slot_logits.shape)
        # print('slot_lengths shape', slot_lengths.shape)
        pred_slots = []
        for i in range(batch_size):
            session_slots = []

            length = slot_lengths[i]
            if length <= 0:
                continue
            logit_actual = slot_logits[i][:length]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logit_actual, trans_params)
            session_slots =viterbi_sequence

            pred_slots .append(session_slots)
        # print('pred_slots shape', len(pred_slots))
        return pred_slots
        pass

    def predict(self, data_test_dict):
        """
        根据训练好的模型标记数据
        Args:
            data_test_dict: dict
        Return:
            pass
        """
        print('predicting...')
        infer_iter = Iterator(data_test_dict, self._feature_names, self._batch_size, self._sequence_length,
                              shuffle=False, infer=True)
        infer_samples = 0
        infer_results = []

        with tqdm(total=infer_iter.n_samples) as pbar:
            for batch_iter in infer_iter.next():
                feed_dict = dict()
                for feature_name in self._feature_names:  # features
                    # feature
                    batch_data = batch_iter[feature_name]
                    item = {self.input_feature_ph_dict[feature_name]: batch_data}
                    feed_dict.update(item)
                    # dropout
                    item = {self.weight_dropout_ph_dict[feature_name]: 0.0}
                    feed_dict.update(item)

                feed_dict.update(
                    {
                        self.dropout_rate_ph: self._dropout_rate,
                        self.rnn_dropout_rate_ph: self._rnn_dropout,
                    })
                pred_intents, batch_count, slot_logits, trans_params, seq_lengths = \
                    self.sess.run([self.pred_intents,
                                   self.intent_count,
                                   self.slot_logits,
                                   self.transition_params,
                                   self.sequence_actual_length],
                                  feed_dict=feed_dict)
                pred_slots = self.crf_infer(slot_logits, seq_lengths, trans_params)
                infer_samples += batch_count
                batch_keys = batch_iter['sid']
                for i, sid in enumerate(batch_keys):
                    length = len(pred_slots[i])
                    # infer_results += [[sid, pred_intents[i].tolist()[:length], pred_slots[i]]]
                    infer_results.append((sid, pred_intents[i], pred_slots[i]))

                pbar.update(batch_count)
        print("infer session count", len(infer_results))
        print('infer sentences count', infer_samples)
        return infer_results

    # def compute_loss(self):
    #     """
    #     计算loss
    #
    #     Return:
    #         loss: scalar
    #     """
    #     if not self._use_crf:
    #         labels = tf.reshape(
    #             tf.contrib.layers.one_hot_encoding(
    #                 tf.reshape(self.input_label_ph, [-1]), num_classes=self._nb_classes),
    #             shape=[-1, self._sequence_length, self._nb_classes])
    #         cross_entropy = -tf.reduce_sum(labels * tf.log(self.logits), axis=2)
    #         mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))
    #         cross_entropy_masked = tf.reduce_sum(
    #             cross_entropy * mask, axis=1) / tf.cast(self.sequence_actual_length, tf.float32)
    #         return tf.reduce_mean(cross_entropy_masked)
    #     else:
    #         slot_labels = tf.reshape(self.input_label_ph, [-1, self._sequence_length])
    #         slot_logits = tf.reshape(self.slot_logits, [-1, self._sequence_length, self._nb_hidden*3])
    #         slot_lengths = tf.reshape(self.sequence_actual_length, [-1])
    #         log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
    #            slot_logits, slot_labels, slot_lengths)
    #         return tf.reduce_mean(-log_likelihood)


class Iterator(object):
    def __init__(self, data_dict: dict, feat_names, batch_size, sequence_length, shuffle=False, infer=False):
        self._feat_names = feat_names
        self._data_dict = data_dict
        self.batch_size = batch_size
        self.seq_length = sequence_length
        self.shuffle = shuffle
        self.infer = infer
        samples = [len(data_dict[k]) for k in self._data_dict]
        self._n_samples = sum(samples)

    @property
    def size(self):
        return len(self._data_dict)

    @property
    def n_samples(self):
        return self._n_samples

    def _prepare_batch_data(self, batch_keys):
        batch_feat_dict = {}
        batch_samples = [self._data_dict[k] for k in batch_keys]
        # batch_turns = [len(x) for x in batch_samples]
        # max_turn = max(batch_turns)
        batch_size = len(batch_keys)
        for feat_name in self._feat_names:
            batch_feat_dict[feat_name] = np.zeros(shape=[batch_size, self.seq_length], dtype='int32')
        if not self.infer:
            batch_feat_dict['intent'] = np.zeros(shape=[batch_size], dtype='int32')
            batch_feat_dict['slot'] = np.zeros(shape=[batch_size,self.seq_length], dtype='int32')

        for i, session_sample in enumerate(batch_samples):
            feat, intent, slot = session_sample
            for feat_name in self._feat_names:
                batch_feat_dict[feat_name][i] = feat[feat_name]
            if not self.infer:
                batch_feat_dict['slot'][i] = slot
                batch_feat_dict['intent'][i] = intent
        batch_feat_dict['sid'] = batch_keys
        return batch_feat_dict

    def next(self):
        keys = list(self._data_dict.keys())
        if self.shuffle:
            np.random.shuffle(keys)
        start_idx = 0
        while start_idx < self.size:
            end_idx = start_idx + self.batch_size
            if end_idx > self.size:
                end_idx = self.size
            batch_keys = keys[start_idx:end_idx]
            yield self._prepare_batch_data(batch_keys)
            start_idx += self.batch_size
        pass
