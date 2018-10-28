#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    加载数据
"""
import sys
import codecs
import pickle
import numpy as np
from utils import map_item2id,map_domain2id
import collections


# INTENT_DIC = {'O': 1,
#  'atis_flight': 4,
#  'atis_airfare': 10,
#  'atis_ground_service': 21,
#  'atis_airline': 30,
#  'atis_abbreviation': 31,
#  'atis_aircraft': 41,
#  'atis_flight_time': 49,
#  'atis_quantity': 53,
#  'atis_flight#atis_airfare': 79,
#  'atis_airport': 80,
#  'atis_distance': 82,
#  'atis_city': 84,
#  'atis_ground_fare': 87,
#  'atis_capacity': 89,
#  'atis_flight_no': 94,
#  'atis_meal': 104,
#  'atis_restriction': 105,
#  'atis_airline#atis_flight_no': 127,
#  'atis_ground_service#atis_ground_fare': 137,
#  'atis_airfare#atis_flight_time': 138,
#  'atis_cheapest': 139,
#  'atis_aircraft#atis_flight#atis_flight_no': 140}
INTENT_DIC={'O': 1,
 'atis_flight': 2,
 'atis_airfare': 3,
 'atis_ground_service': 4,
 'atis_airline': 5,
 'atis_abbreviation': 6,
 'atis_aircraft': 7,
 'atis_flight_time': 8,
 'atis_quantity': 9,
 'atis_flight#atis_airfare': 10,
 'atis_airport': 11,
 'atis_distance': 12,
 'atis_city': 13,
 'atis_ground_fare': 14,
 'atis_capacity': 15,
 'atis_flight_no': 16,
 'atis_meal': 17,
 'atis_restriction': 18,
 'atis_airline#atis_flight_no': 19,
 'atis_ground_service#atis_ground_fare': 20,
 'atis_airfare#atis_flight_time': 21,
 'atis_cheapest': 22,
 'atis_aircraft#atis_flight#atis_flight_no': 23,
 'OTHERS': 0}
DOMAIN_DIC = {
    'music': 1,
    'phone_call': 2,
    'navigation': 3,
    'OTHERS': 0
}


def load_vocs(paths):
    """
    加载vocs
    Args:
        paths: list of str, voc路径
    Returns:
        vocs: list of dict
    """
    vocs = []
    for path in paths:
        with open(path, 'rb') as file_r:
            vocs.append(pickle.load(file_r))
    return vocs


def load_lookup_tables(paths):
    """
    加载lookup tables
    Args:
        paths: list of str, emb路径
    Returns:
        lookup_tables: list of dict
    """
    lookup_tables = []
    for path in paths:
        with open(path, 'rb', encoding='utf-8') as file_r:
            lookup_tables.append(pickle.load(file_r))
    return lookup_tables


def load_session_data(path, feature_names, vocs, max_len, model='train'):
    assert model in ['train', 'test']
    fr = open(path, 'r', encoding='utf-8')
    samples = fr.read().strip().split('\n\n')
    print('number of samples', len(samples))
    data_dict = collections.defaultdict(list)

    for i, sample in enumerate(samples):
        sentences = sample.split('\n')
        ss = sentences[0].split('\t')
        if model == 'train':
            assert len(ss) == 4
        else:
            assert len(ss) == 3

        sid = ss[0]
        intent = None
        if model == 'train':
            intent = ss[3]
        feat_dict = {}
        for feature_name in feature_names:
            feat_dict[feature_name] = []
        slot = []
        for sentence in sentences[1:]:
            ss = sentence.split('\t')
            for i, feat_name in enumerate(feature_names):
                feat_dict[feat_name].append(ss[i])
            if model == 'train':
                slot += [ss[-1]]

        # data_dict[sid].append((intent, slot, feat_dict))
        data_dict[sid]=(intent, slot, feat_dict)

    # index all features
    # max_turn = max([len(data_dict[x]) for x in data_dict])
    print('number of sessions', len(data_dict))
    # print('max turn of sessions', max_turn)

    idx_dict = dict()
    for sid in data_dict:
        session_list = data_dict[sid]
        # session_x = []
        # for label, slot, feat_dict in session_list:
        #     label_idx = INTENT_DIC.get(label, 0)
        #     slot_idx = map_item2id(slot, vocs[-1], max_len)
        #     length = len(slot)
        #     feat_idx_dict = dict()
        #     for i, feat_name in enumerate(feature_names):
        #         assert length == len(feat_dict[feat_name])
        #         feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        #     session_x += [[feat_idx_dict, label_idx, slot_idx]]
        # idx_dict[sid] = session_x
        label, slot, feat_dict = session_list
        label_idx = INTENT_DIC.get(label, 0)
        slot_idx = map_item2id(slot, vocs[-1], max_len)
        length = len(slot)
        feat_idx_dict = dict()
        for i, feat_name in enumerate(feature_names):
            # assert length == len(feat_dict[feat_name])
            feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        session_x = [feat_idx_dict, label_idx, slot_idx]
        idx_dict[sid] = session_x
    return idx_dict


def load_session_infer_data(path, feature_names, vocs, max_len, model='test'):
    assert model in ['train', 'test']
    assert model == 'test'

    fr = open(path, 'r', encoding='utf-8')
    samples = fr.read().strip().split('\n\n')
    print('number of samples', len(samples))
    data_dict = collections.OrderedDict()

    for i, sample in enumerate(samples):
        sentences = sample.split('\n')
        ss = sentences[0].split('\t')
        assert len(ss) == 3

        sid = ss[0]
        intent = None
        feat_dict = {}
        for feature_name in feature_names:
            feat_dict[feature_name] = []
        slot = []
        for sentence in sentences[1:]:
            ss = sentence.split('\t')
            for i, feat_name in enumerate(feature_names):
                feat_dict[feat_name].append(ss[i])
            if model == 'train':
                slot += [ss[-1]]
        if sid not in data_dict:
            data_dict[sid] = []
        data_dict[sid].append((intent, slot, feat_dict))

    # index all features
    max_turn = max([len(data_dict[x]) for x in data_dict])
    print('number of sessions', len(data_dict))
    print('max turn of sessions', max_turn)

    idx_dict = collections.OrderedDict()
    for sid in data_dict:
        session_list = data_dict[sid]
        session_x = []
        for label, slot, feat_dict in session_list:
            feat_idx_dict = dict()
            for i, feat_name in enumerate(feature_names):
                feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
            session_x += [[feat_idx_dict, None, None]]
        idx_dict[sid] = session_x
    return idx_dict

def load_session_data_domain(path, feature_names, vocs, max_len, model='train'):
    assert model in ['train', 'test']
    fr = open(path, 'r', encoding='utf-8')
    samples = fr.read().strip().split('\n\n')
    print('number of samples', len(samples))
    data_dict = collections.defaultdict(list)

    for i, sample in enumerate(samples):
        sentences = sample.split('\n')
        ss = sentences[0].split('\t')
        if model == 'train':
            assert len(ss) == 4
        else:
            assert len(ss) == 3

        sid = ss[0]
        intent = None
        if model == 'train':
            intent = ss[3]
        feat_dict = {}
        for feature_name in feature_names:
            feat_dict[feature_name] = []
        slot = []
        for sentence in sentences[1:]:
            ss = sentence.split('\t')
            for i, feat_name in enumerate(feature_names):
                feat_dict[feat_name].append(ss[i])
            if model == 'train':
                slot += [ss[-1]]

        # data_dict[sid].append((intent, slot, feat_dict))
        data_dict[sid]=(intent, slot, feat_dict)

    # index all features
    # max_turn = max([len(data_dict[x]) for x in data_dict])
    print('number of sessions', len(data_dict))
    # print('max turn of sessions', max_turn)

    idx_dict = dict()
    for sid in data_dict:
        session_list = data_dict[sid]
        # session_x = []
        # for label, slot, feat_dict in session_list:
        #     label_idx = INTENT_DIC.get(label, 0)
        #     slot_idx = map_item2id(slot, vocs[-1], max_len)
        #     length = len(slot)
        #     feat_idx_dict = dict()
        #     for i, feat_name in enumerate(feature_names):
        #         assert length == len(feat_dict[feat_name])
        #         feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        #     session_x += [[feat_idx_dict, label_idx, slot_idx]]
        # idx_dict[sid] = session_x
        label, slot, feat_dict = session_list
        label_idx = INTENT_DIC.get(label, 0)
        slot_idx = map_item2id(slot, vocs[-1], max_len)
        length = len(slot)
        feat_idx_dict = dict()
        for i, feat_name in enumerate(feature_names[:-1]):
            # assert length == len(feat_dict[feat_name])
            feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        arr=map_domain2id(feat_dict['f3'],vocs[2],max_len)
        for feature_id in range(arr.shape[0]):
            feat_name = 'domain'+str(feature_id)
            feat_idx_dict[feat_name] = arr[feature_id]
        session_x = [feat_idx_dict, label_idx, slot_idx]
        idx_dict[sid] = session_x
    return idx_dict

def init_data(path, feature_names, vocs, max_len, model='train', intent_path=None,
              use_char_feature=False, word_len=None, sep='\t'):
    """
    加载数据(待优化，目前是一次性加载整个数据集)
    Args:
        path: str, 数据路径
        feature_names: list of str, 特征名称
        vocs: list of dict
        max_len: int, 句子最大长度
        model: str, in ('train', 'test')
        use_char_feature: bool，是否使用char特征
        word_len: None or int，单词最大长度
        sep: str, 特征之间的分割符, default is '\t'
    Returns:
        data_dict: dict
    """
    assert model in ('train', 'test')
    file_r = codecs.open(path, 'r', encoding='utf-8')
    if intent_path:
        intent_file_r = codecs.open(intent_path, 'r', encoding='utf-8')
        intents = intent_file_r.read().strip().split('\n')
    sentences = file_r.read().strip().split('\n\n')

    sentence_count = len(sentences)
    feature_count = len(feature_names)
    data_dict = dict()
    for feature_name in feature_names:
        data_dict[feature_name] = np.zeros((sentence_count, max_len), dtype='int32')
    # char feature
    if use_char_feature:
        data_dict['char'] = np.zeros(
            (sentence_count, max_len, word_len), dtype='int32')
        char_voc = vocs.pop(0)
    if model == 'train':
        data_dict['label'] = np.zeros((len(sentences), max_len), dtype='int32')
        data_dict['intent'] = np.zeros((len(sentences)), dtype='int32')
        data_dict['prev_domain'] = np.zeros((len(sentences)), dtype='int32')
    for index, sentence in enumerate(sentences):
        items = sentence.split('\n')
        one_instance_items = []
        [one_instance_items.append([]) for _ in range(len(feature_names) + 1)]
        for item in items[1:]:
            feature_tokens = item.split(sep)
            for j in range(feature_count):
                one_instance_items[j].append(feature_tokens[j])
            if model == 'train':
                one_instance_items[-1].append(feature_tokens[-1])
        for i in range(len(feature_names)):
            data_dict[feature_names[i]][index, :] = map_item2id(
                one_instance_items[i], vocs[i], max_len)
        if use_char_feature:
            for i, word in enumerate(one_instance_items[0]):
                if i >= max_len:
                    break
                data_dict['char'][index][i, :] = map_item2id(
                    word, char_voc, word_len)
        if model == 'train':
            data_dict['intent'][index] = INTENT_DIC.get(items[0].split(sep)[-1].rstrip('\n'), 0)
            data_dict['prev_domain'][index] = DOMAIN_DIC.get(items[0].split(sep)[1], 0)
            data_dict['label'][index, :] = map_item2id(
                one_instance_items[-1], vocs[-1], max_len)
        sys.stdout.write('loading data: %d\r' % index)
    file_r.close()
    return data_dict

def label2one_hot(slot_idx,nb_classes):
    # label = np.array([0, 3, 2, 8, 9, 1])  ##标签数据，标签从0开始
    # classes = max(label) + 1  ##类别数为最大数加1
    one_hot_label = np.zeros(shape=(slot_idx.shape[0], nb_classes))  ##生成全0矩阵
    one_hot_label[np.arange(0, slot_idx.shape[0]), slot_idx] = 1  ##相应标签位置置1
    return one_hot_label

def load_session_data_one_hot_labels(path, feature_names, vocs, max_len, nb_classes,model='train'):
    assert model in ['train', 'test']
    fr = open(path, 'r', encoding='utf-8')
    samples = fr.read().strip().split('\n\n')
    print('number of samples', len(samples))
    data_dict = collections.defaultdict(list)

    for i, sample in enumerate(samples):
        sentences = sample.split('\n')
        ss = sentences[0].split('\t')
        if model == 'train':
            assert len(ss) == 4
        else:
            assert len(ss) == 3

        sid = ss[0]
        intent = None
        if model == 'train':
            intent = ss[3]
        feat_dict = {}
        for feature_name in feature_names:
            feat_dict[feature_name] = []
        slot = []
        for sentence in sentences[1:]:
            ss = sentence.split('\t')
            for i, feat_name in enumerate(feature_names):
                feat_dict[feat_name].append(ss[i])
            if model == 'train':
                slot += [ss[-1]]

        # data_dict[sid].append((intent, slot, feat_dict))
        data_dict[sid]=(intent, slot, feat_dict)

    # index all features
    # max_turn = max([len(data_dict[x]) for x in data_dict])
    print('number of sessions', len(data_dict))
    # print('max turn of sessions', max_turn)

    idx_dict = dict()
    for sid in data_dict:
        session_list = data_dict[sid]
        # session_x = []
        # for label, slot, feat_dict in session_list:
        #     label_idx = INTENT_DIC.get(label, 0)
        #     slot_idx = map_item2id(slot, vocs[-1], max_len)
        #     length = len(slot)
        #     feat_idx_dict = dict()
        #     for i, feat_name in enumerate(feature_names):
        #         assert length == len(feat_dict[feat_name])
        #         feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        #     session_x += [[feat_idx_dict, label_idx, slot_idx]]
        # idx_dict[sid] = session_x
        label, slot, feat_dict = session_list
        label_idx = INTENT_DIC.get(label, 0)
        slot_idx = map_item2id(slot, vocs[-1], max_len)
        one_hot_slot = label2one_hot(slot_idx,nb_classes)
        length = len(slot)
        feat_idx_dict = dict()
        for i, feat_name in enumerate(feature_names):
            # assert length == len(feat_dict[feat_name])
            feat_idx_dict[feat_name] = map_item2id(feat_dict[feat_name], vocs[i], max_len)
        session_x = [feat_idx_dict, label_idx, one_hot_slot]
        idx_dict[sid] = session_x
    return idx_dict
