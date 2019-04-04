#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by liliangjie on 2018/11/10 
# Email llj : laiangnaduo91@gmail.com
from collections import Counter
import os,codecs,json
import util_common as util

'''
对分词后的文件做简单统计，不常用
'''


def getword4file(filepath,start=0):
    '''
    获取每行split后的每个词，返回词的列表
    :param filepath: 
    :return: 
    :rtype:list
    '''
    res=[]
    with codecs.open(filepath,'rU',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            res.extend(line.split()[start:])
    return res

def getword4folder(path, ifcount=0):
    '''
    结合上一函数，统计文件夹内所有符合要求的文件
    :param path: 
    :return:
    :rtype: list
    '''
    filepaths = util.getfileinfolder(path, '.txt')
    res = []
    for filep in filepaths:
        if os.path.isfile(filep):
            res.extend(getword4file(filep))
    print("get totaly %d words" % len(res))
    if ifcount:
        c = Counter(res)
        print("get totaly %d distinct words" % len(c))
        print('常用词频度统计结果')
        for (k, v) in c.most_common(50):
            print('%s%s %s  %d' % ('  ' * (5 - len(k)), k, '*' * int(v / 10000), v))
    return res

def getlinelength4json(path,top=20):
    '''
    功能同getlinelength4file，只处理json文件
    :param path: 
    :type path: 
    :return: 
    :rtype: 
    '''
    f=open(path)
    d=json.load(f)
    cnter=[]
    for k in d.keys():
            cnter.append(len(d[k]))
    c=Counter(cnter)
    for k,v in c.most_common(top):
            print("%02d %s %d" %(k,'*'*int(v/20000),v))

def getlinelength4file(filepath,start=0,ifcount=0):
    '''
    获取文件中每行split后的长度
    :param filepath: 
    :type filepath:str 
    :param start: 
    :type start:int 
    :return: 
    :rtype: list
    '''
    res = []
    print("loading file: %s" %filepath)
    with codecs.open(filepath, 'rU', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            res.append(len(line[start:]))
    if ifcount:
        c = Counter(res)
        print("get totaly %d distinct words" % len(c))
        print('每行长度统计结果')
        for (k, v) in c.most_common(50):
            print('%02d %s  %d' % (k, '*' * int(v / 500000), v))
    return res

def getfiledtop(cnter,filedfile,top=50):
    '''
    按filedfile里词的词频排序
    :param cnter: counter of all words
    :type cnter: Counter
    :param filedfile: 
    :param top: 
    :return: 
    '''
    worddic={}
    inwords = util.load2list(filedfile)
    for i in inwords:
        if cnter.has_key(i):
            worddic[i]=cnter[i]
    newcnter=Counter(worddic)
    top = min(len(newcnter),top)
    topnwords=[ "%s %d" %(i,c) for (i , c) in newcnter.most_common(top)]
    respath= "%s_top%d.txt" %(os.path.splitext(filedfile)[0],top)
    util.list2txt(topnwords,respath)
    return topnwords

def main():
    basepath=r'./data'
    segpath=basepath+r'/data_seg'
    filedpath=r'./source/other_jiebadicts'
    filedfiles=util.getfileinfolder(filedpath,'.txt')
    # wordl=getword4folder(segpath)
    wordl = getlinelength4file(segpath+r'/log_b_18.txt',start=1,ifcount=1)
    # for f in filedfiles:
        # getfiledtop(c,f,200)

if __name__ == '__main__':
    main()