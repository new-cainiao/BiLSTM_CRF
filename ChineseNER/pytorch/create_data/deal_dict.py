#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
def read_file(file_path):
    data_list = []
    tar_path = "{}/key_dict.txt".format(file_path)
    if os.path.exists(tar_path):
        with open(tar_path, "r") as f:
            data_list = f.readlines()
    return data_list

def deal_data(data_list):
    data_s = set()
    for data in data_list:
        data_l = []
        # 去除最后的\n
        data = data.rstrip("\n")
        # 空格分词后取0
        data = data.split(" ")[0]
        # 去除（）
        data = data.split("（")[0]
        if "）" in data:
            data = ""
        # 根据/拆分
        if "/" in data:
            data_temp = data.split("/")
            data = ""
            if len(data_temp) == 1:
                data_l.append(data_temp[0])
            elif len(data_temp) == 2:
                if abs(len(data_temp[0]) - len(data_temp[1])) <= 1:
                    data_l.extend(data_temp)
                else:
                    data_l.append(data_temp[0])
        # 根据、拆分
        if data and "、" in data:
            data_temp = data.split("、")
            data = ""
            data_l.extend(data_temp)
        if data and len(data) > 2:
           data_s.add(data)
        else: 
            for data in data_l:
                if data and len(data) > 2:
                    data_s.add(data)
    return list(data_s)

def write_file(file_path, data_list):
    tar_path = "{}/deal_key_dict.txt".format(file_path)
    with open(tar_path, "w") as f:
        for data in data_list:
            f.write(data + "\n")

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_list = read_file(file_path)
    data_list = deal_data(data_list)
    write_file(file_path, data_list)