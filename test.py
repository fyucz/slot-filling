#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    标记文件
"""
import codecs
import yaml
import pickle
import tensorflow as tf
from load_data import load_vocs, init_data, load_session_data, load_session_infer_data,load_session_data_domain
from model import SequenceLabelingModel2
import numpy as np

# INTENT_DIC={
#     'music.play' : 1,
#     'music.pause': 2,
#     'music.prev': 3,
#     'music.next': 4,
#     'navigation.navigation': 5,
#     'navigation.open':6,
#     'navigation.start_navigation':7,
#     'navigation.cancel_navigation':8,
#     'phone_call.make_a_phone_call':9,
#     'phone_call.cancel':10,
#     'OTHERS':0
# }

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


def main():
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)


    feature_names = config['model_params']['feature_names'][:2]
    for id in range(0, 83):
        feature_names.append('domain' + str(id))

    use_char_feature = config['model_params']['use_char_feature']

    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)

    feature_weight_shape_dict, feature_weight_dropout_dict, \
    feature_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names[:2]:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['dropout_rate']
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)

    for feature_name in feature_names[2:]:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params']['default']['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params']['default']['dropout_rate']
        path_pre_train = config['model_params']['embed_params']['default']['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)

    # char embedding shape
    if use_char_feature:
        feature_weight_shape_dict['char'] = \
            config['model_params']['embed_params']['char']['shape']
        conv_filter_len_list = config['model_params']['conv_filter_len_list']
        conv_filter_size_list = config['model_params']['conv_filter_size_list']
    else:
        conv_filter_len_list = None
        conv_filter_size_list = None
    # 加载数据

    # 加载vocs
    path_vocs = []
    if use_char_feature:
        path_vocs.append(config['data_params']['voc_params']['char']['path'])
    for feature_name in config['model_params']['feature_names']:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    # for feature_name in feature_names[2:]:
    #     path_vocs.append(config['data_params']['voc_params']['default']['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)


    # 加载数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    max_len = config['model_params']['sequence_length']
    word_len = config['model_params']['word_length']
    session_data_dict = load_session_data_domain(config['data_params']['path_test'], config['model_params']['feature_names'], vocs, max_len,model='test')
    # print(session_data_dict)
    # 加载模型
    model = SequenceLabelingModel2(
        sequence_length=config['model_params']['sequence_length'],
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        num_layers=config['model_params']['bilstm_params']['num_layers'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        use_char_feature=use_char_feature,
        conv_filter_size_list=conv_filter_size_list,
        conv_filter_len_list=conv_filter_len_list,
        word_length=word_len,
        path_model=config['model_params']['path_model'])

    saver = tf.train.Saver()
    saver.restore(model.sess, config['model_params']['path_model'])

    # 标记
    infer_results = model.predict(session_data_dict)
    # print(infer_results)
    # 写入文件
    slot_voc = dict()
    for key in vocs[-1]:
        slot_voc[vocs[-1][key]] = key
    intent_voc = dict()
    for key in INTENT_DIC:
        intent_voc[INTENT_DIC[key]] = key

    with codecs.open(config['data_params']['path_test'], 'r', encoding='utf-8') as file_r:
        sentences = file_r.readlines()
    file_result = codecs.open(
        config['data_params']['path_result'], 'w', encoding='utf-8')
    infer_sentence_results = []
    # for sid, pred_intents in infer_results:
    #     for x in pred_intents:
    #         infer_sentence_results += [(sid, x)]
    # ['10000', [2], [[1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 5, 1, 1, 1, 1, 1, 22, 59]]]
    for sid, pred_intents,pre_slots in infer_results:
        pre_label = []
        pre_label.append(pred_intents)
        for slot in pre_slots:
            pre_label.append(slot)
        infer_sentence_results.append((sid, pre_label))
    print('predict sentences count', len(infer_sentence_results), len(sentences))
    # for i, sentence in enumerate(sentences):
    #     sid, pred_intent = infer_sentence_results[i]
    #     for j, item in enumerate(sentence.split('\n')):
    #         if j == 0:
    #             for key in INTENT_DIC.keys():
    #                 if pred_intent == INTENT_DIC.get(key):
    #                     file_result.write('%s\t%s\n' % (item, key))
    #             continue
    #         else:
    #             file_result.write('%s\tO\n' % item)
    #     file_result.write('\n')
    infer_session = []
    session_temp = []
    for ss in sentences:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            infer_session.append(session_temp)
            session_temp = []
    if session_temp:
        infer_session.append(session_temp)
    assert len(infer_session) == len(infer_sentence_results)
    for sessions,results in zip(infer_session,infer_sentence_results):
        labels = results[1]
        # print(len(sessions) , len(labels))
        # print(sessions)
        assert len(sessions) == len(labels)
        first = True
        for sess,lab in zip(sessions,labels):
            if first:
                file_result.write(sess.strip()+'\t'+intent_voc[lab]+'\n')
                first = False
            else:
                file_result.write(sess.strip() + '\t' + slot_voc[lab] + '\n')
        file_result.write('\n')
    file_result.close()

def test_acc():
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    test_path = config['data_params']['path_result']
    true_path = config['data_params']['path_test_true']
    with open(test_path,'r') as test_file:
        sentences = test_file.readlines()
    with open(true_path,'r') as true_file:
        truths = true_file.readlines()
    assert len(sentences) == len(truths)
    test_session = []
    truth_session = []
    session_temp = []
    for ss in sentences:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            test_session.append(session_temp)
            session_temp = []
    if session_temp:
        test_session.append(session_temp)
    session_temp = []
    for ss in truths:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            truth_session.append(session_temp)
            session_temp = []
    if session_temp:
        truth_session.append(session_temp)
    assert len(truth_session) == len(test_session)
    sum = len(truth_session)
    intent_acc,intent_count = 0,0
    slot_acc,slot_count,slot_sum = 0,0,0
    acc,count = 0,0
    for test_ss,true_ss in zip(test_session,truth_session):
        assert len(test_ss) == len(true_ss)
        test_intent = test_ss[0].strip().split()[-1]
        true_intent = true_ss[0].strip().split()[-1]
        if test_intent == true_intent:
            intent_count += 1
        # slot_flag = True
        # for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
        #     if test_slot.strip().split()[-1] == true_slot.strip().split()[-1]:
        #         continue
        #     else:
        #         slot_flag = False
        #         break
        # if slot_flag:
        #     slot_count += 1
        # if test_intent == true_intent and slot_flag:
        #     count +=1
        for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
            slot_sum += 1
            if test_slot.strip().split()[-1] == true_slot.strip().split()[-1]:
                slot_count += 1
    print("slot_acc:" + str(float(slot_count) / slot_sum))
    print("intent_acc:"+str(float(intent_count)/sum))
    # print("slot_acc:" + str(float(slot_count) / sum))
    print("acc:" + str(float(count) / sum))


def test_f1():
    # slot_set = set()
    SLOT_DIC = {'I-toloc.airport_name': 0, 'B-toloc.city_name': 1, 'B-day_name': 2, 'B-arrive_time.time': 3, 'B-depart_time.time': 4, 'B-return_date.date_relative': 5, 'B-economy': 6, 'B-depart_date.month_name': 7, 'I-state_name': 8, 'I-flight_time': 9, 'B-state_name': 10, 'B-depart_time.time_relative': 11, 'B-arrive_date.day_number': 12, 'B-flight_mod': 13, 'B-toloc.state_name': 14, 'I-class_type': 15, 'B-arrive_time.time_relative': 16, 'B-flight_time': 17, 'B-toloc.airport_name': 18, 'B-transport_type': 19, 'B-depart_time.end_time': 20, 'I-return_date.date_relative': 21, 'B-meal_code': 22, 'I-round_trip': 23, 'B-flight_stop': 24, 'B-depart_date.day_name': 25, 'B-return_date.day_name': 26, 'I-arrive_time.time_relative': 27, 'B-city_name': 28, 'B-depart_date.date_relative': 29, 'I-transport_type': 30, 'B-stoploc.airport_code': 31, 'B-meal': 32, 'B-round_trip': 33, 'B-meal_description': 34, 'I-airline_name': 35, 'B-arrive_time.period_of_day': 36, 'B-flight_number': 37, 'B-depart_time.period_mod': 38, 'B-fromloc.airport_name': 39, 'B-period_of_day': 40, 'B-arrive_date.day_name': 41, 'B-airport_code': 42, 'B-arrive_date.date_relative': 43, 'B-cost_relative': 44, 'B-state_code': 45, 'I-arrive_time.end_time': 46, 'I-city_name': 47, 'I-restriction_code': 48, 'B-stoploc.city_name': 49, 'I-stoploc.city_name': 50, 'B-flight': 51, 'B-mod': 52, 'B-fare_amount': 53, 'B-arrive_time.start_time': 54, 'B-aircraft_code': 55, 'I-fromloc.airport_name': 56, 'B-depart_date.today_relative': 57, 'B-fromloc.state_name': 58, 'I-toloc.city_name': 59, 'B-toloc.airport_code': 60, 'B-flight_days': 61, 'B-toloc.state_code': 62, 'B-arrive_date.month_name': 63, 'I-fare_amount': 64, 'I-cost_relative': 65, 'B-connect': 66, 'B-fromloc.city_name': 67, 'B-depart_date.day_number': 68, 'I-depart_time.period_of_day': 69, 'I-fromloc.city_name': 70, 'B-compartment': 71, 'I-toloc.state_name': 72, 'B-depart_time.period_of_day': 73, 'B-toloc.country_name': 74, 'B-or': 75, 'I-arrive_time.time': 76, 'I-airport_name': 77, 'I-flight_number': 78, 'I-flight_mod': 79, 'B-booking_class': 80, 'I-depart_time.time': 81, 'B-class_type': 82, 'B-days_code': 83, 'O': 84, 'I-depart_time.time_relative': 85, 'B-depart_time.start_time': 86, 'B-airline_code': 87, 'B-fare_basis_code': 88, 'I-depart_time.start_time': 89, 'I-depart_time.end_time': 90, 'B-fromloc.state_code': 91, 'B-depart_date.year': 92, 'B-airport_name': 93, 'B-airline_name': 94, 'B-restriction_code': 95, 'I-arrive_time.start_time': 96, 'I-depart_date.day_number': 97, 'I-fromloc.state_name': 98, 'B-fromloc.airport_code': 99, 'B-arrive_time.end_time': 100,'I-depart_date.today_relative':101}
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    test_path = config['data_params']['path_result']
    true_path = config['data_params']['path_test_true']
    with open(test_path,'r') as test_file:
        sentences = test_file.readlines()
    with open(true_path,'r') as true_file:
        truths = true_file.readlines()
    assert len(sentences) == len(truths)
    test_session = []
    truth_session = []
    session_temp = []
    for ss in sentences:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            test_session.append(session_temp)
            session_temp = []
    if session_temp:
        test_session.append(session_temp)
    session_temp = []
    for ss in truths:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            truth_session.append(session_temp)
            session_temp = []
    if session_temp:
        truth_session.append(session_temp)
    assert len(truth_session) == len(test_session)
    intent_acc,intent_count = 0,0
    matrix = np.zeros((len(SLOT_DIC),len(SLOT_DIC)),dtype=np.int)
    for test_ss,true_ss in zip(test_session,truth_session):
        assert len(test_ss) == len(true_ss)
        test_intent = test_ss[0].strip().split()[-1]
        true_intent = true_ss[0].strip().split()[-1]
        if test_intent == true_intent:
            intent_count += 1
        # slot_flag = True
        # for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
        #     if test_slot.strip().split()[-1] == true_slot.strip().split()[-1]:
        #         continue
        #     else:
        #         slot_flag = False
        #         break
        # if slot_flag:
        #     slot_count += 1
        # if test_intent == true_intent and slot_flag:
        #     count +=1
        for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
            # if true_slot.strip().split()[-1] not in slot_set:
            #     slot_set.add(true_slot.strip().split()[-1])
            # if test_slot.strip().split()[-1] not in slot_set:
            #     slot_set.add(true_slot.strip().split()[-1])
            i = int(SLOT_DIC[true_slot.strip().split()[-1]])
            j = int(SLOT_DIC[test_slot.strip().split()[-1]])
            matrix[i][j] += 1
    # for id,slot in enumerate(slot_set):
    #     slot_dic[slot] = id
    # print(slot_dic)
    precision = np.zeros(len(SLOT_DIC),dtype=np.float)
    recall = np.zeros(len(SLOT_DIC),dtype=np.float)
    f1 = np.zeros(len(SLOT_DIC),dtype=np.float)
    sum_row = np.sum(matrix,axis=1) #行
    sum_column = np.sum(matrix,axis=0)
    TP = np.zeros(len(SLOT_DIC),dtype=np.float)
    # print(sum_column)
    for class_id in range(len(SLOT_DIC)):
        if matrix[class_id][class_id] > 0 and sum_row[class_id]>0:
            recall[class_id] = matrix[class_id][class_id]/sum_row[class_id]
        TP[class_id] = matrix[class_id][class_id]
        if matrix[class_id][class_id]>0 and sum_column[class_id]>0:
            precision[class_id] = matrix[class_id][class_id]/sum_column[class_id]
        if precision[class_id]>0 and recall[class_id] > 0:
            f1[class_id] = 2 * precision[class_id] *recall[class_id] /(recall[class_id]+precision[class_id])
        else:
            f1[class_id] = 0

    Macro_p = np.mean(precision[:-1])
    Macro_r = np.mean(recall[:-1])
    Macro_f1 = 2*Macro_p*Macro_r/(Macro_p+Macro_r)
    # print(np.mean(recall[:-1]))
    print(Macro_f1)
    Micro_p = np.sum(TP)/np.sum(sum_column)
    Micro_r = np.sum(TP)/np.sum(sum_row)
    Micro_f1 = 2*Micro_p*Micro_r/(Micro_p+Micro_r)
    print(Micro_p,Micro_r,Micro_f1)


def test_f1_with_sklearn(SLOT_DIC):
    # slot_set = set()
    # SLOT_DIC = {'I-toloc.airport_name': 0, 'B-toloc.city_name': 1, 'B-day_name': 2, 'B-arrive_time.time': 3, 'B-depart_time.time': 4, 'B-return_date.date_relative': 5, 'B-economy': 6, 'B-depart_date.month_name': 7, 'I-state_name': 8, 'I-flight_time': 9, 'B-state_name': 10, 'B-depart_time.time_relative': 11, 'B-arrive_date.day_number': 12, 'B-flight_mod': 13, 'B-toloc.state_name': 14, 'I-class_type': 15, 'B-arrive_time.time_relative': 16, 'B-flight_time': 17, 'B-toloc.airport_name': 18, 'B-transport_type': 19, 'B-depart_time.end_time': 20, 'I-return_date.date_relative': 21, 'B-meal_code': 22, 'I-round_trip': 23, 'B-flight_stop': 24, 'B-depart_date.day_name': 25, 'B-return_date.day_name': 26, 'I-arrive_time.time_relative': 27, 'B-city_name': 28, 'B-depart_date.date_relative': 29, 'I-transport_type': 30, 'B-stoploc.airport_code': 31, 'B-meal': 32, 'B-round_trip': 33, 'B-meal_description': 34, 'I-airline_name': 35, 'B-arrive_time.period_of_day': 36, 'B-flight_number': 37, 'B-depart_time.period_mod': 38, 'B-fromloc.airport_name': 39, 'B-period_of_day': 40, 'B-arrive_date.day_name': 41, 'B-airport_code': 42, 'B-arrive_date.date_relative': 43, 'B-cost_relative': 44, 'B-state_code': 45, 'I-arrive_time.end_time': 46, 'I-city_name': 47, 'I-restriction_code': 48, 'B-stoploc.city_name': 49, 'I-stoploc.city_name': 50, 'B-flight': 51, 'B-mod': 52, 'B-fare_amount': 53, 'B-arrive_time.start_time': 54, 'B-aircraft_code': 55, 'I-fromloc.airport_name': 56, 'B-depart_date.today_relative': 57, 'B-fromloc.state_name': 58, 'I-toloc.city_name': 59, 'B-toloc.airport_code': 60, 'B-flight_days': 61, 'B-toloc.state_code': 62, 'B-arrive_date.month_name': 63, 'I-fare_amount': 64, 'I-cost_relative': 65, 'B-connect': 66, 'B-fromloc.city_name': 67, 'B-depart_date.day_number': 68, 'I-depart_time.period_of_day': 69, 'I-fromloc.city_name': 70, 'B-compartment': 71, 'I-toloc.state_name': 72, 'B-depart_time.period_of_day': 73, 'B-toloc.country_name': 74, 'B-or': 75, 'I-arrive_time.time': 76, 'I-airport_name': 77, 'I-flight_number': 78, 'I-flight_mod': 79, 'B-booking_class': 80, 'I-depart_time.time': 81, 'B-class_type': 82, 'B-days_code': 83, 'O': 84, 'I-depart_time.time_relative': 85, 'B-depart_time.start_time': 86, 'B-airline_code': 87, 'B-fare_basis_code': 88, 'I-depart_time.start_time': 89, 'I-depart_time.end_time': 90, 'B-fromloc.state_code': 91, 'B-depart_date.year': 92, 'B-airport_name': 93, 'B-airline_name': 94, 'B-restriction_code': 95, 'I-arrive_time.start_time': 96, 'I-depart_date.day_number': 97, 'I-fromloc.state_name': 98, 'B-fromloc.airport_code': 99, 'B-arrive_time.end_time': 100,'I-depart_date.today_relative':101}
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)
    test_slots , true_slots= [],[]
    test_path = config['data_params']['path_result']
    true_path = config['data_params']['path_test_true']
    print('path_result:',test_path)
    print('path_test_true:', true_path)
    with open(test_path,'r') as test_file:
        sentences = test_file.readlines()
    with open(true_path,'r') as true_file:
        truths = true_file.readlines()
    assert len(sentences) == len(truths)
    test_session = []
    truth_session = []
    session_temp = []
    for ss in sentences:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            test_session.append(session_temp)
            session_temp = []
    if session_temp:
        test_session.append(session_temp)
    session_temp = []
    for ss in truths:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            truth_session.append(session_temp)
            session_temp = []
    if session_temp:
        truth_session.append(session_temp)
    assert len(truth_session) == len(test_session)
    se = set()
    for test_ss,true_ss in zip(test_session,truth_session):
        assert len(test_ss) == len(true_ss)

        # slot_flag = True
        # for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
        #     if test_slot.strip().split()[-1] == true_slot.strip().split()[-1]:
        #         continue
        #     else:
        #         slot_flag = False
        #         break
        # if slot_flag:
        #     slot_count += 1
        # if test_intent == true_intent and slot_flag:
        #     count +=1
        for test_slot,true_slot in zip(test_ss[1:],true_ss[1:]):
            if true_slot.strip().split()[-1]  in SLOT_DIC:
                i = int(SLOT_DIC[true_slot.strip().split()[-1]])
            else:
                se.add(true_slot.strip().split()[-1])
            j = int(SLOT_DIC[test_slot.strip().split()[-1]])
            test_slots.append(j)
            true_slots.append(i)

    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    Macro_f1 = f1_score(true_slots, test_slots, average='macro')
    print('Macro_f1:',Macro_f1)
    Micro_f1 = f1_score(true_slots, test_slots, average='micro')
    print('Micro_f1:',Micro_f1)
    print(se)
if __name__ == '__main__':

    main()

    #bio
    # SLOT_DIC = {'I-toloc.airport_name': 0, 'B-toloc.city_name': 1, 'B-day_name': 2, 'B-arrive_time.time': 3, 'B-depart_time.time': 4, 'B-return_date.date_relative': 5, 'B-economy': 6, 'B-depart_date.month_name': 7, 'I-state_name': 8, 'I-flight_time': 9, 'B-state_name': 10, 'B-depart_time.time_relative': 11, 'B-arrive_date.day_number': 12, 'B-flight_mod': 13, 'B-toloc.state_name': 14, 'I-class_type': 15, 'B-arrive_time.time_relative': 16, 'B-flight_time': 17, 'B-toloc.airport_name': 18, 'B-transport_type': 19, 'B-depart_time.end_time': 20, 'I-return_date.date_relative': 21, 'B-meal_code': 22, 'I-round_trip': 23, 'B-flight_stop': 24, 'B-depart_date.day_name': 25, 'B-return_date.day_name': 26, 'I-arrive_time.time_relative': 27, 'B-city_name': 28, 'B-depart_date.date_relative': 29, 'I-transport_type': 30, 'B-stoploc.airport_code': 31, 'B-meal': 32, 'B-round_trip': 33, 'B-meal_description': 34, 'I-airline_name': 35, 'B-arrive_time.period_of_day': 36, 'B-flight_number': 37, 'B-depart_time.period_mod': 38, 'B-fromloc.airport_name': 39, 'B-period_of_day': 40, 'B-arrive_date.day_name': 41, 'B-airport_code': 42, 'B-arrive_date.date_relative': 43, 'B-cost_relative': 44, 'B-state_code': 45, 'I-arrive_time.end_time': 46, 'I-city_name': 47, 'I-restriction_code': 48, 'B-stoploc.city_name': 49, 'I-stoploc.city_name': 50, 'B-flight': 51, 'B-mod': 52, 'B-fare_amount': 53, 'B-arrive_time.start_time': 54, 'B-aircraft_code': 55, 'I-fromloc.airport_name': 56, 'B-depart_date.today_relative': 57, 'B-fromloc.state_name': 58, 'I-toloc.city_name': 59, 'B-toloc.airport_code': 60, 'B-flight_days': 61, 'B-toloc.state_code': 62, 'B-arrive_date.month_name': 63, 'I-fare_amount': 64, 'I-cost_relative': 65, 'B-connect': 66, 'B-fromloc.city_name': 67, 'B-depart_date.day_number': 68, 'I-depart_time.period_of_day': 69, 'I-fromloc.city_name': 70, 'B-compartment': 71, 'I-toloc.state_name': 72, 'B-depart_time.period_of_day': 73, 'B-toloc.country_name': 74, 'B-or': 75, 'I-arrive_time.time': 76, 'I-airport_name': 77, 'I-flight_number': 78, 'I-flight_mod': 79, 'B-booking_class': 80, 'I-depart_time.time': 81, 'B-class_type': 82, 'B-days_code': 83, 'O': 84, 'I-depart_time.time_relative': 85, 'B-depart_time.start_time': 86, 'B-airline_code': 87, 'B-fare_basis_code': 88, 'I-depart_time.start_time': 89, 'I-depart_time.end_time': 90, 'B-fromloc.state_code': 91, 'B-depart_date.year': 92, 'B-airport_name': 93, 'B-airline_name': 94, 'B-restriction_code': 95, 'I-arrive_time.start_time': 96, 'I-depart_date.day_number': 97, 'I-fromloc.state_name': 98, 'B-fromloc.airport_code': 99, 'B-arrive_time.end_time': 100,'I-depart_date.today_relative':101}

    #bieos
    # SLOT_DIC_DIEOS = {'S-fromloc.city_name': 0, 'B-depart_time.time': 1, 'E-depart_time.time': 2, 'S-toloc.city_name': 3,
    #  'S-arrive_time.time': 4, 'S-arrive_time.period_of_day': 5, 'S-depart_date.day_name': 6,
    #  'S-depart_time.period_of_day': 7, 'B-flight_time': 8, 'E-flight_time': 9, 'B-fromloc.city_name': 10,
    #  'E-fromloc.city_name': 11, 'S-cost_relative': 12, 'B-round_trip': 13, 'E-round_trip': 14, 'B-fare_amount': 15,
    #  'E-fare_amount': 16, 'S-depart_date.today_relative': 17, 'B-toloc.city_name': 18, 'E-toloc.city_name': 19,
    #  'S-city_name': 20, 'S-stoploc.city_name': 21, 'S-toloc.airport_code': 22, 'S-depart_time.time_relative': 23,
    #  'B-class_type': 24, 'E-class_type': 25, 'S-depart_date.date_relative': 26, 'B-airline_name': 27,
    #  'E-airline_name': 28, 'S-airline_name': 29, 'I-depart_time.time': 30, 'S-arrive_time.time_relative': 31,
    #  'B-depart_time.start_time': 32, 'E-depart_time.start_time': 33, 'B-depart_time.end_time': 34,
    #  'E-depart_time.end_time': 35, 'B-fromloc.airport_name': 36, 'E-fromloc.airport_name': 37, 'S-toloc.state_name': 38,
    #  'B-depart_date.day_number': 39, 'E-depart_date.day_number': 40, 'S-depart_date.month_name': 41, 'S-mod': 42,
    #  'S-fare_basis_code': 43, 'I-toloc.city_name': 44, 'S-flight_time': 45, 'S-depart_date.day_number': 46,
    #  'S-transport_type': 47, 'S-flight_mod': 48, 'B-cost_relative': 49, 'E-cost_relative': 50,
    #  'S-arrive_date.month_name': 51, 'S-arrive_date.day_number': 52, 'S-meal': 53, 'S-toloc.state_code': 54,
    #  'S-meal_description': 55, 'S-return_date.month_name': 56, 'S-return_date.day_number': 57,
    #  'I-fromloc.airport_name': 58, 'S-airline_code': 59, 'S-depart_time.period_mod': 60, 'B-arrive_time.time': 61,
    #  'E-arrive_time.time': 62, 'S-flight_stop': 63, 'I-arrive_time.time': 64, 'B-city_name': 65, 'E-city_name': 66,
    #  'I-airline_name': 67, 'S-fromloc.airport_code': 68, 'S-arrive_date.day_name': 69, 'S-time': 70, 'S-or': 71,
    #  'B-arrive_date.day_number': 72, 'E-arrive_date.day_number': 73, 'S-depart_time.start_time': 74,
    #  'S-depart_time.end_time': 75, 'I-city_name': 76, 'B-economy': 77, 'E-economy': 78, 'S-class_type': 79,
    #  'S-flight_number': 80, 'S-economy': 81, 'S-arrive_time.period_mod': 82, 'I-fromloc.city_name': 83,
    #  'S-depart_time.time': 84, 'B-transport_type': 85, 'E-transport_type': 86, 'S-fare_amount': 87, 'S-flight_days': 88,
    #  'S-fromloc.airport_name': 89, 'B-stoploc.city_name': 90, 'I-stoploc.city_name': 91, 'E-stoploc.city_name': 92,
    #  'B-toloc.airport_name': 93, 'E-toloc.airport_name': 94, 'S-state_code': 95, 'B-flight_stop': 96,
    #  'E-flight_stop': 97, 'S-arrive_time.start_time': 98, 'B-arrive_time.end_time': 99, 'E-arrive_time.end_time': 100,
    #  'B-fromloc.state_name': 101, 'E-fromloc.state_name': 102, 'I-toloc.airport_name': 103,
    #  'S-arrive_date.date_relative': 104, 'S-depart_date.year': 105, 'S-return_date.date_relative': 106,
    #  'S-airport_code': 107, 'S-aircraft_code': 108, 'S-fromloc.state_code': 109, 'S-connect': 110, 'I-flight_stop': 111,
    #  'S-fromloc.state_name': 112, 'B-arrive_time.start_time': 113, 'E-arrive_time.start_time': 114,
    #  'B-restriction_code': 115, 'E-restriction_code': 116, 'B-toloc.state_name': 117, 'E-toloc.state_name': 118,
    #  'B-airport_name': 119, 'I-airport_name': 120, 'E-airport_name': 121, 'S-toloc.country_name': 122,
    #  'S-days_code': 123, 'B-fare_basis_code': 124, 'E-fare_basis_code': 125, 'B-arrive_time.time_relative': 126,
    #  'I-arrive_time.time_relative': 127, 'E-arrive_time.time_relative': 128, 'B-arrive_time.period_of_day': 129,
    #  'E-arrive_time.period_of_day': 130, 'B-depart_time.time_relative': 131, 'I-depart_time.time_relative': 132,
    #  'E-depart_time.time_relative': 133, 'S-day_name': 134, 'S-period_of_day': 135, 'S-toloc.airport_name': 136,
    #  'S-restriction_code': 137, 'S-round_trip': 138, 'S-today_relative': 139, 'S-stoploc.state_code': 140,
    #  'I-transport_type': 141, 'B-meal_code': 142, 'E-meal_code': 143, 'B-today_relative': 144, 'I-today_relative': 145,
    #  'E-today_relative': 146, 'S-meal_code': 147, 'B-flight_mod': 148, 'I-flight_mod': 149, 'E-flight_mod': 150,
    #  'S-state_name': 151, 'S-stoploc.airport_name': 152, 'S-arrive_date.today_relative': 153, 'I-flight_time': 154,
    #  'S-time_relative': 155, 'B-time': 156, 'E-time': 157, 'S-return_time.period_of_day': 158,
    #  'B-return_date.day_number': 159, 'E-return_date.day_number': 160, 'B-depart_time.period_of_day': 161,
    #  'E-depart_time.period_of_day': 162, 'I-meal_code': 163, 'S-return_time.period_mod': 164,
    #  'B-depart_date.today_relative': 165, 'I-depart_date.today_relative': 166, 'E-depart_date.today_relative': 167,
    #  'S-month_name': 168, 'S-day_number': 169, 'B-return_date.date_relative': 170, 'I-return_date.date_relative': 171,
    #  'E-return_date.date_relative': 172, 'B-meal_description': 173, 'E-meal_description': 174,
    #  'S-arrive_time.end_time': 175, 'B-return_date.today_relative': 176, 'I-return_date.today_relative': 177,
    #  'E-return_date.today_relative': 178, 'S-return_date.day_name': 179,'O':180,
    # 'S-compartment':181, 'S-flight':182, 'E-state_name':183, 'S-booking_class':184, 'B-state_name':185, 'B-flight_number':186, 'S-stoploc.airport_code':187, 'E-flight_number':188}

    # test_f1_with_sklearn(SLOT_DIC)
