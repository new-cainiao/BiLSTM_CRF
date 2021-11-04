#!/usr/bin/python
# -*- coding: UTF-8 -*-

import codecs
import pandas as pd
import numpy as np
import re
import os
import collections
from sklearn.model_selection import train_test_split
import pickle

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def data2pkl(file_path, data_pkl_name):
    """数据处理。拆分训练测试集，并产生相关id2字典，并保存pkl文件

    Args:
        file_path (str): 
        data_pkl_name (str): 
    """
    datas = list()
    all_datas = list()
    labels = list()
    linedata=list()
    linelabel=list()
    tags = set()

    input_data = codecs.open('{}/wordtagsplit.txt'.format(file_path),'r','utf-8')
    for line in input_data.readlines():
        line = line.split()
        linedata=[]
        linelabel=[]
        numNotO=0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] and word[1]!='O':
                numNotO+=1
        if numNotO!=0:
            datas.append(linedata)
            labels.append(linelabel)
        all_datas.append(linedata)
    input_data.close()
    print(len(datas),tags)
    print(len(labels))
    
    # all_words = flatten(datas)
    # 加载完整的data数据，创建word2id
    all_words = flatten(all_datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = list(range(1, len(set_words)+1))

    tags = [i for i in tags]
    tag_ids = list(range(len(tags)))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"] = len(word2id)+1
    print(word2id)
    max_len = 60
    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:  
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len: 
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=list(range(len(datas))))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)

    
    # 创建全部数据多pkl
    with open('{}/../pkl_model/{}.pkl'.format(file_path, data_pkl_name), 'wb') as outp:
	    pickle.dump(word2id, outp)
	    pickle.dump(id2word, outp)
	    pickle.dump(tag2id, outp)
	    pickle.dump(id2tag, outp)
	    pickle.dump(x_train, outp)
	    pickle.dump(y_train, outp)
	    pickle.dump(x_test, outp)
	    pickle.dump(y_test, outp)
	    pickle.dump(x_valid, outp)
	    pickle.dump(y_valid, outp)
    print('** Finished saving the data.')
    # 创建对应字典的pkl
    with open('{}/../pkl_model/{}_id2d.pkl'.format(file_path, data_pkl_name), 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(id2tag, outp)
    print('** Finished saving the id2d.')
    
def origin2tag(file_path, ori_file_name):
    """处理标注文件，拆分成字粒度标注

    Args:
        file_path (str): 目标文件路径
        ori_file_name (str): 原文件名
    """
    input_data = codecs.open('{}/{}'.format(file_path, ori_file_name),'r','utf-8')
    output_data = codecs.open('{}/wordtag.txt'.format(file_path),'w','utf-8')
    for line in input_data.readlines():
        line=line.strip()
        i=0
        while i <len(line):
	        if line[i] == '{':
		        i+=2
		        temp=""
		        while line[i]!='}':
			        temp+=line[i]
			        i+=1
		        i+=2
		        word=temp.split(':')
		        sen = word[1]
		        output_data.write(sen[0]+"/B_"+word[0]+" ")
		        for j in sen[1:len(sen)-1]:
			        output_data.write(j+"/M_"+word[0]+" ")
		        output_data.write(sen[-1]+"/E_"+word[0]+" ")
	        else:
		        output_data.write(line[i]+"/O ")
		        i+=1
        output_data.write('\n')
    input_data.close()
    output_data.close()

def tagsplit(file_path):
    """句子拆分

    Args:
        file_path (str): 目标路径
    """
    with open('{}/wordtag.txt'.format(file_path),'rb') as inp:
	    texts = inp.read().decode('utf-8')
    sentences = re.split('[，。！？、‘’“”（）]/[O]'.encode('utf-8').decode('utf-8'), texts)
    # sentences = re.split('[。！？；]/[O]'.encode('utf-8').decode('utf-8'), texts)
    output_data = codecs.open('{}/wordtagsplit.txt'.format(file_path),'w','utf-8')
    for sentence in sentences:
	    if sentence != " ":
		    output_data.write(sentence.strip()+'\n')
    output_data.close()


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    origin2tag(file_path, "deal_text.txt")
    tagsplit(file_path)
    data2pkl(file_path, "Mydata_04_all")
