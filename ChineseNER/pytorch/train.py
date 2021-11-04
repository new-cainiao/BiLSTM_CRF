# coding=utf-8
import os
import re
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import codecs
import copy
from tqdm import tqdm
from BiLSTM_CRF import BiLSTM_CRF, BiLSTM_CRF_FAST, BiLSTM_CRF_MODIFY_PARALLEL, prepare_sequence, prepare_sequence_batch
from resultCal import calculate, calculate3

# 参数
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 5
lr = 0.005
weight_decay = 1e-4
use_gpu = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_data_pkl(model_path='./pkl_model/Mydata_01.pkl'):
    with open(model_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    print("train len:",len(x_train))
    print("test len:",len(x_test))
    print("valid len", len(x_valid))
    return word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_valid, y_valid

def read_data_pkl_id2d(model_path='./pkl_model/Mydata_01_id2d.pkl'):
    with open(model_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        id2tag = pickle.load(inp)
    return word2id, id2word, id2tag

def train(word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_valid, y_valid):
    tag2id[START_TAG]=len(tag2id)
    tag2id[STOP_TAG]=len(tag2id)

    # model = BiLSTM_CRF_FAST(len(id2word)+1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)
    model = BiLSTM_CRF_FAST(len(id2word)+1, tag2id, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    # model = nn.DataParallel(model).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # # Check predictions before training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #     print(model(precheck_sent))

    # 统一向量长度
    sentence_in_pad, targets_pad = prepare_sequence_batch(x_train, y_train)
    # sentence_in_pad, targets_pad = sentence_in_pad.to(DEVICE), targets_pad.to(DEVICE)
    for epoch in range(EPOCHS):
        # single
        index=0
        # for sentence, tags in zip(x_train,y_train):
        for sentence, tags in tqdm(zip(sentence_in_pad,targets_pad), desc="epoch_{}:".format(epoch)):
            sentence, tags = sentence.to(DEVICE), tags.to(DEVICE)
            index+=1
            model.zero_grad()
            # sentence=torch.tensor(sentence, dtype=torch.long)
            # tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence, tags)
            loss.backward()
            optimizer.step()
            
        ## parallel  并行代码无预测模块
        # model.zero_grad()
        # # Step 2. Get our batch inputs ready for the network, that is,
        # # turn them into Tensors of word indices.
        # # If training_data can't be included in one batch, you need to sample them to build a batch
        # sentence_in_pad, targets_pad = prepare_sequence_batch(x_train, y_train)
        # # Step 3. Run our forward pass.
        # loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
        # # Step 4. Compute the loss, gradients, and update the parameters by
        # # calling optimizer.step()
        # loss.backward()
        # optimizer.step() 
        
        computing_rate(x_test, y_test, model, id2word, id2tag)
        path_name = "./model_GPU/model"+str(epoch)+".pkl"
        print(path_name)
        torch.save(model, path_name)
        print("model has been saved")
    # 使用验证集计算最后的准召率
    computing_rate(x_valid, y_valid, model, id2word, id2tag)
        
def computing_rate(x_test, y_test, model, id2word, id2tag):
    entityres=[]
    entityall=[]
    for sentence, tags in zip(x_test,y_test):
        sentence = torch.tensor(sentence, dtype=torch.long).to(DEVICE)
        score, predict = model(sentence)
        entityres = calculate(sentence,predict,id2word,id2tag,entityres)
        entityall = calculate(sentence,tags,id2word,id2tag,entityall)
        
    jiaoji = [i for i in entityres if i in entityall]
    if len(jiaoji)!=0:
        zhun = float(len(jiaoji))/len(entityres)
        zhao = float(len(jiaoji))/len(entityall)
        print("test:")
        print("zhun:", zhun)
        print("zhao:", zhao)
        print("f:", (2*zhun*zhao)/(zhun+zhao))
    else:
        print("zhun:",0)

def deal_word(sentences):
    word2id = {}
    for sentence in sentences:
        for word in list(sentence):
            if word not in word2id:
                word2id[word] = len(word2id)  
    id2word = dict([val,key] for key,val in word2id.items())
    return word2id, id2word

def deal_sentence(sentence, word2id):
    def X_padding(words):
        max_len = len(word2id)
        ids = []
        for word in words:
            ids.append(word2id[word])
        if len(ids) >= max_len:  
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids
    sentence = list(sentence)
    ids = X_padding(sentence)
    return  ids

def deal_entityres(ori_entityres, result_l):
    key = ""
    value = ""
    temporary_d = {}
    location = {}
    for entityres in ori_entityres:
        for sign_word in entityres:
            if sign_word.split("_")[-1] == "key":
                key += sign_word.split("/")[0]
            elif sign_word.split("_")[-1] == "value":
                value += sign_word.split("/")[0]
            else:
                if key and not value:
                    location['key'] = sign_word
                elif value:
                    location['value'] = sign_word
        if key and value:
            temporary_d[key] = value
            temporary_d['location'] = location
            result_l.append(copy.deepcopy(temporary_d))
            key = ""
            value = ""
            temporary_d = {}
            location = {}
        # 目前一个key只对应一个value，且存在value之前必须存在key
        if value and not key:
            value = ""

def forecast(model_path, file_path, sentences):
    # 预测使用CPU
    # word2id, id2word = deal_word(sentences)
    word2id, id2word, id2tag = read_data_pkl_id2d(file_path)
    result_l = []
    for sentence_ori in sentences:
        start_idx = 0
        # 拆分句子
        sentence_l = re.split(r'[，。！？、‘’“”（）]', sentence_ori)
        result = []
        for sentence in sentence_l:
            ori_sentence = sentence
            sentence = deal_sentence(sentence, word2id)
            if not use_gpu:
                model = torch.load(model_path, map_location='cpu')
            else:
                model = torch.load(model_path, map_location='cuda:0')
            sentence = torch.tensor(sentence, dtype=torch.long).to(DEVICE)
            score, predict = model(sentence)
            res = calculate3(sentence, predict, id2word, id2tag, start_idx)
            deal_entityres(res, result)
            start_idx += len(ori_sentence) + 1
            # print(res)
        result_l.append({"ori_text":sentence_ori, "result":result})
    print(result_l)

def begin_train(pkl_file_path):
    # word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_valid, y_valid = read_data_pkl(pkl_file_path)
    # train(word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_valid, y_valid)
    data_l = read_data_pkl(pkl_file_path)
    train(*data_l)
    
def begin_forecast(model_path, id2d_model_path, data_l):
    forecast(model_path, id2d_model_path, data_l)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    # begin_train("{}/pkl_model/Mydata_04_all.pkl".format(file_path))
    begin_forecast("./model/model9.pkl", "{}/pkl_model/Mydata_04_all_id2d.pkl".format(file_path), ["经营活动现金流量净额为174,85万元，较2017年增加125,65万元，良好的现金流量状况为的偿债能力提供了强有力的支撑，控股总装机容量1,91120万千瓦。电话号码：18736141159", "小明生日11月在北京举办晚会共花销4000余元"])
