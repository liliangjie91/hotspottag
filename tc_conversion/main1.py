# encoding=utf-8
import tablib
import os
import pandas as pd
from pandas import ExcelWriter
from numpy import *
import numpy
import numpy as np
import jieba.analyse
import jieba
import sys
# reload(sys)
import codecs
import re
import math
from langconv import *
import zh_wiki
import math

# sys.setdefaultencoding('utf8')
def cutSentence(doc):
    # 对文章分句
    sents = []
    pattern = re.compile('？|。|！')
    line_list = pattern.split(doc)
    l = len(line_list)
    for i in xrange(l):
        if len(line_list[i]) == 0:
            continue
        else:
            sents.append(line_list[i])
    return sents

def calcVect(list, word):
    # 计算句子的向量值
    value = []
    for i in range(len(list)):
        if list[i] in word:
            value.append(word.index(list[i]))
        else:
            value.append(-1)
    return value

def calcSum(list_1, list_2):
    # 计算句子的值
    sum = 0
    for i in range(len(list_1)):
        if list_1[i] in list_2:
            num = 60 - list_1[i]
            sum += num
        else:
            continue
    sum = sum*1.0/30
    return sum

def calcSenLen(sen, avi):
    # 计算句子长度特征
    return 1-abs(len(sen) - avi)/ai

def calcSimilar(sen, topic):
    # 计算句子和标题的相似度
    m = len(sen)
    sum = 0
    for i in range(m):
        if sen[i] in topic:
            sum += 1
        else:
            continue
    l = round((float)(sum*sum)/m, 4)
    return l


if __name__ == '__main__':
    # sys.path.append(r'E:\Novelty\keywords DES')
    sys.path.append(os.getcwd())
    stopwords = []
    with codecs.open('stopwords.txt', encoding='utf_8_sig') as f:  # 停用词处理
        for line in f.readlines():
            stopwords.append(line.strip('\r\n'))
    txt = ''
    with codecs.open('test.txt', encoding='utf_8_sig') as f:  # 繁体字转化为简体字
        for line in f.readlines():
            line = Converter('zh-hans').convert(line.decode('utf-8'))
            line = line.encode('utf-8')
            txt += line

    txt_cut = cutSentence(txt)   # 文章分句
    txt_cutWord = []
    txt_cutWord1 = []
    s = []
    n = ''
    txt_cutWord.append(jieba.analyse.extract_tags(txt.strip(' \t\n\r'), topK = 50))  # 文章提取权重最大的50个词
        # txt_cutWord += (jieba.analyse.textrank(txt[i].strip(' \t\n\r')))
    m = txt_cutWord[0]
    for i in range(len(m)):  # 把关键字以分号分开并以字符串形式返回
        n += m[i]
        n += ','
    # print n

    for i in range(len(txt_cutWord[0])):   # 对keyWords 进行停用词处理， 并返回过滤后的停用词
        if txt_cutWord[0][i] in stopwords:
            continue
        else:
            s.append(txt_cutWord[0][i])

    s = s[::-1]
    s.append(u'为此')
    s.append(u'因此')
    s.append(u'本文')
    s.append(u'总之')
    s.append(u'本实验')
    s.append(u'本研究')
    s = s[::-1]

    t = []
    b = []
    for i in range(len(txt_cut)):  # 对文章内容进行分词
        t.append(jieba.cut(txt_cut[i].strip('\t\r\n')))

    x = []
    topic = '基于新的广义粒子群方法的发电机组轴心轨迹提纯'    # 输入文本标题
    topic_cutword = jieba.cut(topic.strip('\t\r\n'))
    topic_list = list(topic_cutword)
    for i in range(len(topic_list)):
        if topic_list[i] in stopwords:
            continue
        else:
            x.append(topic_list[i])

    for i in range(len(t)):  # 返回文章分词以后以list的形式返回
        b.append(list(t[i]))

    for i in range(len(b)):
        if len(b[i]) == 0:
            b.pop()

    sum = 0
    for i in range(len(b)):
        sum += len(b[i])
    avi = sum/len(b)
    weigth = range(51)
    # result_list = []       # 保存文章句子，句子序号，文本相似度结果
    mylist = []    # 保存文章句子，句子序号，文本相似度结果
    for i in range(len(b)):
        # sum = 0
        pos = 0
        L = 0
        value_ = calcVect(b[i], s)
        sum_ = calcSum(value_, weigth)
        pos = round((float)(1+(numpy.log(len(b)-i + 1))/(numpy.log(len(b)*1.0)*(-1))), 4)
        L = round(1+(float)(abs(len(b[i]) - avi))/avi, 4)
        topic_value = calcSimilar(b[i], x)
        # result_list.append([txt_cut[i], i, round(10*(float)(numpy.log(sum_*sum_+1))/(numpy.log(len(b[i])+1)) + 5*pos + L + 5*topic_value,4)])
        # df = pd.DataFrame(result_list, columns=[u'句子', u'序号', u'Text similarity'])

        ts_algorithm = (str(txt_cut[i].encode('utf8')), str(i), str(round(10*(float)(numpy.log(sum_*sum_+1))/(numpy.log(len(b[i])+1)) + 5*pos + L + 5*topic_value,4)))
        # str_ret = ''.join(list(ts_algorithm))
        mylist.append(ts_algorithm)

        headers = ('句子', '序号', '打分结果')  # excel的columns值
        ret_list = []
        for i in xrange(len(mylist)):  # 将句子切片分割成句子、序号、算法结果三部分
            mylist_article = mylist[i][0:-2]
            mylist_number = mylist[i][-2:-1]
            mylist_result = mylist[i][-1:]
            tmp_data = tuple((mylist_article, mylist_number,mylist_result))
            ret_list.append(tmp_data)

        ret_list = tablib.Dataset(*ret_list, headers=headers)  # 输出到指定路径下的excel文件
        path = os.getcwd()  # 输出为项目所在路径下的文件夹
        out_file_path = path + '/text_similarity_excel.xlsx'
        with open(out_file_path, 'wb') as f:
            f.write(ret_list.xlsx)
