#!/usr/bin/env python
# -*- encoding: utf-8 -*-

def bieos(slot_list,word_list):
    assert len(slot_list) == len(word_list)
    if len(slot_list) == 0:
        return {}
    slots = {} #slot:[word1,word2]
    temp_word = ''
    temp_slot = ''
    for slot,word in zip(slot_list,word_list):
        if slot == 'O':
            continue
        if 'B-' in slot:
            temp_word = word
            temp_slot = slot.strip().split('-')[-1]
        if 'I-' in slot:
            if slot.strip().split('-')[-1] == temp_slot:
                temp_word = temp_word+' '+word
        if 'E-' in slot:
            if slot.strip().split('-')[-1] == temp_slot:
                temp_word = temp_word+' '+word
            if temp_word != '' and temp_slot!= '':
                if temp_slot in slots:
                    slots[temp_slot].append(temp_word)
                else:
                    slots[temp_slot]=[]
                    slots[temp_slot].append(temp_word)
            temp_word = ''
            temp_slot = ''
        if 'S-' in slot:
            temp_word = word
            temp_slot = slot.strip().split('-')[-1]
            if temp_word != '' and temp_slot!= '':
                if temp_slot in slots:
                    slots[temp_slot].append(temp_word)
                else:
                    slots[temp_slot]=[]
                    slots[temp_slot].append(temp_word)
            temp_word = ''
            temp_slot = ''

    return slots


def bio(slot_list,word_list):
    assert len(slot_list) == len(word_list)
    if len(slot_list) == 0:
        return {}
    slots = {} #slot:[word1,word2]
    temp_word = ''
    temp_slot = ''
    for slot,word in zip(slot_list,word_list):
        if slot == 'O':
            if temp_word != '' and temp_slot!= '':
                if temp_slot in slots:
                    slots[temp_slot].append(temp_word)
                else:
                    slots[temp_slot]=[]
                    slots[temp_slot].append(temp_word)
                temp_word = ''
                temp_slot = ''
        if 'B-' in slot:
            if temp_word != '' and temp_slot!= '':
                if temp_slot in slots:
                    slots[temp_slot].append(temp_word)
                else:
                    slots[temp_slot]=[]
                    slots[temp_slot].append(temp_word)

            temp_word = word
            temp_slot = slot.strip().split('-')[-1]
        if 'I-' in slot:
            if slot.strip().split('-')[-1] == temp_slot:
                temp_word = temp_word+' '+word

    if temp_word != '' and temp_slot != '':
        if temp_slot in slots:
            slots[temp_slot].append(temp_word)
        else:
            slots[temp_slot] = []
            slots[temp_slot].append(temp_word)
    return slots


def f1(test_path,true_path):
    print('path_result:', test_path)
    print('path_test_true:', true_path)
    matrix={} #key:(TP,FP,FN)
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

    for test_ss,true_ss in zip(test_session,truth_session):
        assert len(test_ss) == len(true_ss)
        test_slot_list,true_slot_list=[],[]
        test_word_list, true_word_list = [], []
        for test,true in zip(test_ss[1:],true_ss[1:]):
            true_slot_list.append(true.strip().split()[-1])
            test_slot_list.append(test.strip().split()[-1])
            true_word_list.append(true.strip().split()[0])
            test_word_list.append(test.strip().split()[0])
        true_dic = bieos(true_slot_list,true_word_list)
        test_dic = bieos(test_slot_list,test_word_list)
        keys = set()
        for key in true_dic.keys():
            keys.add(key)
        for key in test_dic.keys():
            keys.add(key)
        for key in keys:
            if key not in matrix:
                matrix[key] = [0,0,0] #TP,FP,FN
            true_words,test_words = [],[]
            if key in true_dic:
                true_words = true_dic[key]
            if key in test_dic:
                test_words = test_dic[key]
            for word in true_words:
                if word in test_words:
                    matrix[key][0] += 1
                else:
                    matrix[key][2] += 1
            for word in test_words:
                if word not in true_words:
                    matrix[key][1] += 1

    TP_sum,FP_sum,FN_sum = 0,0,0
    for key in matrix:
        TP,FP,FN = matrix[key]
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN

    # Macro_f1 = 0
    # print('Macro_f1:',Macro_f1)
    Micro_p = float(TP_sum)/(float(TP_sum+FP_sum))
    Micro_r = float(TP_sum) / (float(TP_sum + FN_sum))
    Micro_f1 = 2*Micro_p*Micro_r/(Micro_r+Micro_p)
    print('Micro_p:', Micro_p)
    print('Micro_r:', Micro_r)
    print('Micro_f1:',Micro_f1)


if __name__=='__main__':
    f1('./atis/data/bieos/atis.session.result','./atis/data/bieos/atis.session.test_true')
    # print(bio(['O','B-1','I-1','O','B-2','O','B-3','B-1'],['wo', 'd', 'fe', 'fe', 'gr', 'gt', 's', 'fsd']))
