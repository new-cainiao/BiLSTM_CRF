#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: lixiaobing
Date: 2021/01/29
Desc:
"""
import re
import copy
import os
from tqdm import tqdm
import json
from typing import Any, Dict, List

origintext_file = "OriginText_all"
class DocAnalyzer:

    def __init__(self, jobj:Dict[str, Any]) -> None:
        if 'tree' not in jobj:
            raise Exception('输入json未包含树形结构')
        self.content = jobj 
        self.tree = jobj['tree']
        self.title_nodes:Dict[int, Dict] = {
            0: self.tree['root']
        }
        self.page_size_dict:Dict = {}
        self.node_obj:List = []

    def json_flunter(self, node) -> Dict:
        """
        将tree中的节点打平
        将所有嵌套children提取到统一层次下
        """
        ret:Dict[int, Dict] =  {}
        for child in node['children']:
            ret[child['id']] = child
            ret.update(self.json_flunter(child))
        return ret
    
    def get_page_size(self, pages):
        # 得到page的尺寸
        for page in pages:
            self.page_size_dict[page['number']] = page['crop_bbox']
        
    def read_data(self, node_d, text_list):
        textlines = node_d.get('data', {}).get('textlines', "")
        for textline in textlines:
            title_text = textline.get("text", "")
            if title_text and ".............." not in title_text:
                text_list.append(self.strip_title(title_text))
    
    def strip_title(self, title):
        """
        将title进行清洗，使得更适合利用字符串相似度的的方式来判定同一性
        """
        title = re.sub("[【】 ]", "", title)
        title = re.sub(
            "(?:^(第[一二三四五六七八九十零〇百]+[节章])|^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬]|[0-9a-zA-Z一二三四五六七八九十零〇百]+[)）]|[0-9a-zA-Z一二三四五六七八九十零〇百]+[、. ]|[（（][0-9a-zA-Z一二三四五六七八九十零〇百]+[)）])",
            "", title)
        title = re.sub("的", "", title)
        # title = re.sub("[;。；:： ]$", "", title)
        # title = re.sub("的", "", title)
        # title = re.sub("[【】 ]", "", title)
        # title = re.sub("(?:本公司|关于|本次|发行人|公司)", "", title)
        return title
    
    def deal_text(self, nodes_dict):
        # 处理section_kv
        text_list = []
        for _id, node_d in nodes_dict.items():
            if node_d.get('type') != "title":
                continue
            self.read_data(node_d, text_list)
            children_l = node_d.get('children', [])
            children_text_l = []
            for children_node in children_l:
                if children_node.get('type', "") == "section":
                    self.read_data(children_node, children_text_l)
            if children_text_l:
                text_list.append("".join(children_text_l))
        return "".join(text_list)
    
    def write_text(self, data_text, file_path, field_name):
        dirs = "{}/{}/".format(file_path, origintext_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open("{}/{}.txt".format(dirs, field_name), 'w', encoding='utf-8') as f:
            f.write(data_text)
    
    def analyze(self, file_path, field_name):
        """
        章节树解析入口
        """
        self.get_page_size(self.content['pages'])
        nodes_dict = self.json_flunter(self.tree['root'])
        data_text = self.deal_text(nodes_dict)
        self.write_text(data_text, file_path, field_name)
    
def get_json_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.loads(f.read())
    return content

def read_comfile(file_path):
    tar_list = "{}/{}/".format(file_path, origintext_file)
    file_list = []
    if os.path.exists(tar_list):
        file_list = os.listdir(tar_list)
        for idx in range(len(file_list)):
            file_list[idx] = file_list[idx].strip('.txt')
    return file_list

def deal_pdf(uuid_list, file_path, com_file_list):
    for field_name in uuid_list:
        if field_name in ['.DS_Store'] or field_name in com_file_list:
            continue
        json_content = get_json_content('{}/{}/解析/{}/doc.json'.format(file_path, path_name, field_name))
        docanalyzer = DocAnalyzer(json_content)
        # 分析doc的内容，格式化提取出有需要的字段
        docanalyzer.analyze(file_path, field_name)


def read_keydict_file(file_path):
    data_list = []
    tar_path = "{}/deal_key_dict.txt".format(file_path)
    if os.path.exists(tar_path):
        with open(tar_path, "r") as f:
            data_list = f.readlines()
    return data_list

def read_text_file(file_path):
    file_data = ""
    with open(file_path, "r") as f:
        file_data = f.read()
    return file_data
    
    
def deal_taginfo(tar_list):
    deal_tar_list = []
    for data in tar_list:
        # 去除最后的\n
        data = data.rstrip("\n")
        deal_tar_list.append(data)
    # 标注字典排序，从长到短
    def cmp(x):
        return len(x)
    deal_tar_list = sorted(deal_tar_list, key=cmp, reverse=True)
    return deal_tar_list

def pdftext_add_sign(text_content, tar_list):
    """文本化的pdf，添加对应的标签

    Args:
        text_content ([str]): 
        tar_list (list): 
    """
    key_value_interval = 10
    # company_re = r'[\u4e00-\u9fa5%]{0,10}[0-9+]+[,|，]?[0-9+]+[%千万亿M兆人小时qwW元瓦/吨]*'
    company_re = r'[\u4e00-\u9fa5%]{0,' + r'{}'.format(key_value_interval) + r'}[0-9,，]*[0-9]+[%十百千万亿兆M人小时qwW瓦元吨/及]+(以[上下])?'
    company_value_re = r'[0-9,，]*[0-9]+[%十百千万亿兆M人小时qwW瓦元吨/及]+(以[上下])?'
    for tar_data in tar_list:
        tar_data_re = re.compile(r"{}".format(tar_data) + company_re)
        tar_value_re = re.compile(company_value_re)
        while True:
            tar_data_search = tar_data_re.search(text_content)
            if tar_data_search:
                # 修改key，填充为"{{independent_key:{0}".format(tar_data)}}"
                tar_data_search_x = tar_data_search.span()[0]
                tar_data_search_y = tar_data_search_x + len(tar_data)
                # text_content = text_content[:tar_data_search_x] + "{{" + "independent_key:{0}".format(tar_data) + "}}" + text_content[tar_data_search_y:]
                # 修改value
                key_value_str = tar_data_search.group().replace(tar_data, "")
                tar_value_search = tar_value_re.search(key_value_str)
                if tar_value_search:
                    value_str = tar_value_search.group()
                    tar_value_search_x = tar_data_search_y + tar_value_search.span()[0]
                    tar_value_search_y = tar_value_search_x + len(value_str)
                    text_content = text_content[:tar_data_search_x] + "{{" + "independent_key:{0}".format(tar_data) + "}}" + text_content[tar_data_search_y:tar_value_search_x] + "{{" + "independent_value:{0}".format(value_str) + "}}" + text_content[tar_value_search_y:]
            else:
                break
    return text_content
                

def load_taginfo(file_path):
    tar_list = read_keydict_file(file_path)
    deal_tar_list = deal_taginfo(tar_list)
    # 读取文本化的pdf
    tar_path = "{}/{}/".format(file_path, origintext_file)
    all_deal_text_l = []
    if os.path.exists(tar_path):
        file_list = os.listdir(tar_path)
        for path_name in tqdm(file_list, desc="文件处理中："):
            if path_name in [".DS_Store"]:
                continue
            text_content = read_text_file('{}/{}'.format(tar_path, path_name))
            if text_content:
                all_deal_text_l.append(pdftext_add_sign(text_content, deal_tar_list))
    return all_deal_text_l

def write_deal_text(file_path, data):
    dirs = "{}/deal_text.txt".format(file_path)
    with open(dirs, 'w', encoding='utf-8') as f:
        f.write(data)

if __name__ == "__main__":
    path_name = "全部"
    # path_name = "1"
    file_path = os.path.dirname(os.path.abspath(__file__))
    uuid_list = os.listdir('{}/{}/解析'.format(file_path, path_name))
    com_file_list = read_comfile(file_path)
    deal_pdf(uuid_list, file_path, com_file_list)
    all_deal_text_l = load_taginfo(file_path)
    write_deal_text(file_path, "。".join(all_deal_text_l))



    
