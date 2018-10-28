from jieba import posseg as pseg
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pickle
import os

from tqdm import tqdm


def lstm_voc(inputpath):
    domain_p = 2
    domain_dic = {'Z': 1}
    ENR_p = 1
    ENR_dic = {}
    word_p = 2
    word_dic = {'UNK': 1}
    label_p = 3
    label_dic = {'OTHERS':1,'O':2}
    char_p = 2
    char_dic = {'UNK': 1}
    dic_intent = {}
    with open(inputpath,'r') as fr:
        lines = fr.readlines()
    for id, line in enumerate(lines):
        temp = line.strip('\n').split(' EOS\t')
        query = ' '.join(temp[0].split(' ')[1:])
        labels = temp[1].split(' ')[1:-1]  # label
        intent = temp[1].split(' ')[-1]
        if intent not in label_dic:
            label_dic[intent] = label_p
            dic_intent[intent] = label_p
            label_p += 1
        for label in labels:
            if label not in label_dic:
                label_dic[label] = label_p
                label_p += 1
        psegs = pos_tag(word_tokenize(query.replace("'", '')))
        ENRs = []  # 词性
        segs = query.split(' ')  # word
        for seg, ENR in psegs:
            if seg.strip():
                ENRs.append(ENR)
        for seg in segs:
            if seg not in word_dic:
                word_dic[seg] = word_p
                word_p += 1
            for char in seg:
                if char not in char_dic:
                    char_dic[char] = char_p
                    char_p +=1
        for ENR in ENRs:
            if ENR not in ENR_dic:
                ENR_dic[ENR] = ENR_p
                ENR_p += 1
        for label in labels:
            domain = label.split('-')[1].strip() if label != 'O' else 'o'
            if domain not in domain_dic:
                domain_dic
                [domain] = domain_p
                domain_p += 1

    f1 = open('f1.voc.pkl', 'wb')
    pickle.dump(word_dic, f1)
    f1.close()
    f2 = open('f2.voc.pkl', 'wb')
    pickle.dump(ENR_dic, f2)
    f2.close()
    f3 = open('f3.voc.pkl', 'wb')
    pickle.dump(domain_dic, f3)
    f3.close()
    f4 = open('label.voc.pkl', 'wb')
    pickle.dump(label_dic, f4)
    f4.close()
    f5 = open('char.voc.pkl', 'wb')
    pickle.dump(char_dic, f5)
    f5.close()

def lstm_session(inputpath,outputpath,train=True):
    with open(inputpath,'r') as fr:
        lines = fr.readlines()
    with open(outputpath,'w') as fw:
        for id,line in enumerate(lines):
            temp = line.strip('\n').split(' EOS\t')
            query = ' '.join(temp[0].split(' ')[1:])
            labels = temp[1].split(' ')[1:-1] #label
            intent = temp[1].split(' ')[-1]
            # psegs = pseg.cut(query)
            psegs = pos_tag(word_tokenize(query.replace("'",'')))
            ENRs = [] #词性
            segs = query.split(' ') #word
            for seg,ENR in psegs:
                if seg.strip():
                    # segs.append(seg)
                    ENRs.append(ENR)
            print(id)
            # print(len(segs))
            assert len(segs) == len(labels)
            if train:
                fw.write(str(id)+'\t'+'Z'+'\t'+'Z'+'\t'+intent+'\n')
                for index,label in enumerate(labels):
                    domain = label.split('-')[1].strip() if label != 'O' else 'o'
                    fw.write(segs[index]+'\t'+ENRs[index]+'\t'+domain+'\t'+label+'\n')
                fw.write('\n')
            else:
                fw.write(str(10000+id) + '\t' + 'OTHERS' + '\t' + 'Z' + '\n')
                for index, label in enumerate(labels):
                    domain =  'o'
                    fw.write(segs[index] + '\t' + ENRs[index] + '\t' + domain + '\n')
                fw.write('\n')

def slot_dic(filename,outpath):
    with open(filename,'r') as fr:
        lines = fr.readlines()
    import os
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    slot_dic = {}
    for line in lines:
        temp = line.strip('\n').split(' EOS\t')
        query = temp[0].split(' ')[1:]
        labels = temp[1].split(' ')[1:-1]  # label
        assert len(query) == len(labels)
        temp = ''
        slot_label = ''
        for i in range(len(query)):
            if labels[i] != 'O':
                temp = temp+query[i]+' '
                slot_label = labels[i].split('-')[1]
                slot_dic[query[i]] = slot_label
            elif len(temp) != 0:
                fw = open(outpath+'/'+slot_label+'.txt','a+')
                fw.write(temp+'\n')
                fw.close()
                temp = ''
                slot_label = ''
    pkl = open('slot_dic.voc.pkl', 'wb')
    pickle.dump(slot_dic, pkl)
def slot_dic_set():
    source_dir = 'slot_dic'
    target_dir = 'slot_dic_set'
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)
        lines = open(filepath, 'r').readlines()
        lines = set(lines)
        fw_path = os.path.join(target_dir, filename)
        with open(fw_path,'w') as fw:
            for line in lines:
                fw.write(line)

def countSet(target_file):
    dir = 'slot_dic_set'
    filepath = os.path.join(dir, target_file)
    lines = open(filepath, 'r').readlines()
    target_set = set(lines)
    source_dic = {}
    for filename in os.listdir(dir):
        if filename != target_file:
            filepath = os.path.join(dir, filename)
            lines = open(filepath, 'r').readlines()
            temp_set = set(lines)
            source_dic[filename] = temp_set

    for key in source_dic:
        if len(target_set & source_dic[key]) != 0:
            print(key)

def bulid_ahocorasick():
    import ahocorasick
    a = ahocorasick.Automaton()
    slot_dic = {}
    dirpath = 'slot_dic'
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        tag = '.'.join(filename.split('.')[:-1])
        lines = open(filepath, 'r').readlines()
        lines = set(lines)
        for line in lines:
            if line.strip() not in slot_dic:
                slot_dic[line.strip()] = tag
            else:
                slot_dic[line.strip()] = 'UNK'
    # slot_dic = load_dic('slot-dictionaries')
    counter = 1
    for key in slot_dic.keys():
        a.add_word(key, (counter, key))
        counter += 1
    a.make_automaton()
    return a,slot_dic

def bulid_domain_dic(dirpath = 'slot_dic'):
    """
    创建domain字典
    key : set(words)
    :return:
    """
    domain_dic = {}

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        tag = '.'.join(filename.split('.')[:-1])
        lines = open(filepath, 'r').readlines()
        words = set()
        for line in lines:
            words.add(line.strip())
        domain_dic[tag] = words

    return domain_dic

def word2domain(domain_dic,word):
    """
    查找一个word的domains
    返回一个set
    :param word:
    :return:
    """
    # domain_dic = bulid_domain_dic()
    tags = domain_dic.keys()
    domains = set()
    for tag in tags:
        if word in domain_dic[tag]:
            domains.add(tag)
    return domains


def gen_copurs_with_domain(input,output):
    domain_dic = bulid_domain_dic()

    with open(input,'r') as fr:
        sentences = fr.readlines()
    fw = open(output,'w')
    session = []
    session_temp = []
    for ss in sentences:
        if ss.strip():
            session_temp.append(ss.strip('\n'))
        else:
            session.append(session_temp)
            session_temp = []
    if session_temp:
        session.append(session_temp)

    for ss in tqdm(session):
        fw.write(ss[0].strip()+'\n')
        # sentence_list = []
        # ENRs = []
        # slots = []
        for s in ss[1:]:
            list_from_s = s.strip().split()
            word = list_from_s[0]
            domains = word2domain(domain_dic,word)
            if domains:
                domain = ','.join(domains)
            else:
                domain = 'o'
            list_from_s[2] = domain
            fw.write('\t'.join(list_from_s)+'\n')
        fw.write('\n')
    fw.close()
    #         ENRs.append(list_from_s[1])
    #         slots.append(list_from_s[3])
    #         sentence_list.append(word)
    #     sentence = ' '.join(sentence_list)
    #     dict_label = sen2domain(sentence)
    #     for word,ENR,slot in zip(sentence_list,ENRs,slots):
    #         label = 'o'
    #         if word in dict_label:
    #             label = dict_label[word]
    #         fw.write(word+'\t'+ENR+'\t'+label+'\t'+slot+'\n')
    #     fw.write('\n')
    # fw.close()

def domain2voc():
    domain_dic = bulid_domain_dic()
    pkl_dic = {'o':0}
    p = 1
    for key in domain_dic:
        pkl_dic[key] = p
        p += 1
    pkl = open('domain/f3.voc.pkl', 'wb')
    pickle.dump(pkl_dic, pkl)

def bieos2voc(bieos_filename,output):
    """
    将Bieos的atis.session.train的label编码
    :param bieos_filename:
    :param output:
    :return:
    """
    label_p = 3
    label_dic = {'OTHERS': 1, 'O': 2}
    with open(bieos_filename,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        if not line.strip('\n'):
            continue
        line = line.strip('\n')
        list_from_line = line.split('\t')
        assert len(list_from_line) == 4
        label = list_from_line[-1]
        if label not in label_dic:
            label_dic[label] = label_p
            label_p += 1
    print('label num : ',len(label_dic))
    fw = open(output, 'wb')
    pickle.dump(label_dic, fw)

if __name__=='__main__':
    # dir = 'slot_dic_set'
    # for filename in os.listdir(dir):
    #     print(filename)
    #     countSet(filename)
    #     print('-------------------------------------')

    # import nltk
    # nltk.download()
    # lstm_voc('atis.train.w-intent.iob')
    # slot_dic('atis.train.w-intent.iob', 'slot_dic')
    # slot_dic_set()

    # gen_train_with_domain()
    # gen_test_with_domain()

    # lstm_session('atis.test.w-intent.iob','atis.session.test',False)
    # line = "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis"
    # psegs = pos_tag(word_tokenize(line.replace("'",'')))
    # ENRs = []  # 词性
    # segs = line.split(' ')  # word
    # for seg, ENR in psegs:
    #     if seg.strip():
    #         # segs.append(seg)
    #         ENRs.append(ENR)
    # print(segs)
    # print(ENRs)

    # gen_copurs_with_domain('bio/atis.session.train', 'domain/atis.session.train')
    # gen_copurs_with_domain('bio/atis.session.test', 'domain/atis.session.test')
    domain2voc()
    # bieos2voc('bieos/atis.session.train', 'bieos/label.voc.pkl')