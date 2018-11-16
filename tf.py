import heapq
import re
from pyvi import ViTokenizer
import numpy as np
from operator import itemgetter
import pandas as pd
pt= re.compile(r"_")

# def preprocess(text):
#     sentence= re.sub(pt," ", text)
#     return sentence
# print(preprocess("Việt_nam"))
def segmentation(text):
    return ViTokenizer.tokenize(text)
def split_words(text):
    text = segmentation(text)
    try:
        return [x.strip("0123456789%@$.,=+-!;/()*\"&^:#|\n\t\'").lower() for x in text.split()]
    except TypeError:
        return []

def get_words_feature(text):
    split_word = split_words(text)
    return [word for word in split_word if word.encode('utf-8')]
def get_normal_word(text):
    get_words_featur=get_words_feature(text)
    list=[]
        # print(get_words_feature)
    for item in get_words_featur:
        sentence = re.sub(pt, " ", item)
        list.append(sentence)
    return list
print(get_normal_word("Cộng hòa xã hội chủ nghĩa việt nam"))
def remove_stop_word(list):
    f = open("D:\\studying\\20172\\ML\\Data\\stop.txt", 'r', encoding="UTF-8")
    str1=str(f.read()).splitlines()
    for item in list:
        if item in str1: list.remove(item)
    str2=[u'năm',u'công_ty',u'đồng','việt_nam',u'và',u'là','usd',u'trong',u'số',u'các',u'đó',u'trong',u'vang',u'tỷ']
    for item in list:
        if item in str2:list.remove(item)
    # list.remove('đồng')
    df = pd.read_excel('stopword.xlsx', sheetname='Sheet1', encoding="UTF-8")
    list2 = []
    for item in df.index:
        list2.append(df['stop'][item])
    for item in list:
        if item in list2:list.remove(item)
    # print(list2)

    return list
def remove_stop_word(list):
    f = open("D:\\studying\\20172\\ML\\Data\\stop.txt", 'r', encoding="UTF-8")
    str1=str(f.read()).splitlines()
    for item in list:
        if item in str1: list.remove(item)
    str2=[u'năm',u'công_ty',u'đồng','việt_nam',u'và',u'là','usd',u'trong',u'số',u'các',u'đó',u'trong',u'vang',u'tỷ']
    for item in list:
        if item in str2:list.remove(item)
    # list.remove('đồng')
    df = pd.read_excel('stopword.xlsx', sheetname='Sheet1', encoding="UTF-8")
    list2 = []
    for item in df.index:
        list2.append(df['stop'][item])
    for item in list:
        if item in list2:list.remove(item)
    # print(list2)

    return list
def select_feature(list_sentences):
    vocabulary={}

#     for sentence in list_sentences:
#         sentence_tokenized=remove_stop_word(get_words_feature(sentence))
#         for term in sentence_tokenized:
#             if term not in vocabulary:
#                 vocabulary[term]=1
#             else:
#                 vocabulary[term]+=1
#     features= [(key, value) for (key, value) in heapq.nlargest(10,vocabulary.items(),key=itemgetter(1))]
#     return features
# df = pd.read_excel('data.xlsx', sheetname='Sheet1',encoding="UTF-8")
# list=[]
# for item in df.index:
#     if int(df['label'][item])==1:
#         list.append(df['content'][item])
# # print(heapq.nlargest(1000,select_feature(list)['value']))
# # print([[word.value]for word in select_feature(list)])
# vocal=[]
# for item in select_feature(list):
#     vocal.append(get_normal_word(item[0])[0])
# print(type(vocal))

tags={1:'positive', 0:'neutral', -1:'negative'}
def load_excel_training_data(excel_file, excel_sheet):
    df = pd.read_excel(excel_file, sheetname=excel_sheet, encoding="UTF-8")
    training_data = {}
    for index in tags:
        training_data[tags[index]] = []

    for item in df.index:
        training_data[tags[df['label'][item]]].append(df['content'][item])
    return training_data

def select_feature(trainning_set, tag):
    vocabulary = {}
    for sentence in trainning_set[tag]:
        sentence_tokenized = remove_stop_word(get_normal_word(sentence))
        for term in sentence_tokenized:
            if term not in vocabulary:
                vocabulary[term] = 1
            else:
                vocabulary[term] += 1
    number = 0
    if tag == 'positive': number = 50
    if tag == 'negative': number = 30
    if tag == 'neutral': number = 20
    features = [(key, value) for (key, value) in heapq.nlargest(number, vocabulary.items(), key=itemgetter(1))]
    return features
trainning_set=load_excel_training_data("data.xlsx","Sheet1")
list = []
for tag in trainning_set:
    for item in select_feature(trainning_set,tag):
        if list.count(item[0])==0:list.append(item[0])
    print(len(list))

a=[[1,3,2,4,5,6],[1,2,3,34,3,54]]
list=[]
for i in a:
    for item in i:
        if list.count(item)==0:list.append(item)
print(list)
# print(a.count(2))
