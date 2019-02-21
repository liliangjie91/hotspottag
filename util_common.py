#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by liliangjie on 2018/11/10 
# Email llj : laiangnaduo91@gmail.com
import codecs
import os,json,re
import logging
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
bianma='utf8'


def get_code_field(code,dic_codefield):
    '''
    获取专题子栏目代码对应的中文解释
    :param code: 专题子栏目代码，可能是由分号隔开的多个
    :type code: str
    :param dic_codefield: 专题子栏目代码解释字典
    :type dic_codefield:dict 
    :return: 
    :rtype: 
    '''
    codes = code.strip(';').split(';')
    l0, l1 = [], []
    for c in codes:
        tmp0=dic_codefield[c] if c in dic_codefield else "NULL"
        if '_' in c and c[:c.find('_')] in dic_codefield:
            tmp1=dic_codefield[c[:c.find('_')]]
        else:
            tmp1=tmp0
        l0.append(tmp0)
        l1.append(tmp1)
    res0 = ';'.join(l0)
    res1 = ';'.join(l1)
    return res0,res1

def getfileinfolder(folderpath, prefix=None, recurse=False, sort=True, maxdepth=3, curdepth=0):
    '''
    返回文件夹内符合prefix前缀的文件名
    :param folderpath: 文件夹
    :type folderpath: str
    :param prefix: 前缀匹配模式
    :type prefix: str
    :return: 
    :rtype: list
    '''
    res=[]
    if recurse and curdepth>=maxdepth:
        return []
    if os.path.exists(folderpath):
        for file in os.listdir(folderpath):
            file_p = os.path.join(folderpath,file)
            if os.path.isfile(file_p):
                if prefix and re.search(prefix,file_p):
                    res.append(file_p)
                if not prefix:
                    res.append(file_p)
            elif os.path.isdir(file_p):
                if recurse:
                    res.extend(getfileinfolder(file_p, prefix=prefix, recurse=recurse, sort=False, maxdepth=maxdepth, curdepth=curdepth + 1))
    if sort:
        res=sorted(res,key=lambda x:os.path.getmtime(x))
    return res

def get_FileSize(filePath):
    '''
    获取文件大小MB
    :param filePath: 
    :return: 
    '''
    filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)

def list2txt(l,path):
    '''
    写list成txt
    :param l: 
    :param path: 
    :return: 
    '''
    logger.info("saving list2file : %s\nlength of list %d" %(path,len(l)))
    if os.path.exists(path):
        logger.info("file %s exists del it.." %path)
        os.remove(path)
    with open(path,'w') as f:
        for i in l:
            if isinstance(i,(int,float,long)):
                i = str(i)
            f.write(i.strip().encode('utf-8', 'ignore')+'\n')

def load2list(path,to1column=False,separator=None,get1column=-1,start=0):
    '''
    读取文本返回成list，
    默认一行一元素---进输入path即可
    也可按分隔符分割后每个词为一个元素---to1column=True
    也可按分隔符分割后每个词为一个元素(可选从哪一列开始，默认全部，即从0开始)---to1column=True，start=1
    也可单独获取分割后的某列词为一元素---get1column=需要的那一列从0开始
    
    :param path: 
    :type path: str
    :param to1column: 
    :type to1column:bool 
    :param separator: 
    :type separator: bool
    :param get1column: 
    :type get1column: 
    :return: 
    :rtype: list
    '''
    logger.info("loading file2list : %s" %path)
    res=[]
    if os.path.isfile(path):
        with codecs.open(path,'rU',encoding=bianma,errors='replace') as f:
            for l in f:
                l=l.strip()
                if l:
                    if to1column :
                        res.extend(l.split(separator)[start:])
                    elif get1column>=0:
                        res.append(l.split(separator)[get1column])
                    else:
                        res.append(l)
    logger.info("length of list %d" %len(res))
    return res

def savejson(path,j):
    logger.info("saving file2json : %s" % path)
    with codecs.open(path,"w",encoding=bianma) as f:
        json.dump(j,f)

def loadjson(path):
    '''
    
    :param path: 
    :type path: 
    :return: 
    :rtype:dict 
    '''
    logger.info("loading jsonfile : %s" % path)
    with codecs.open(path, "rU") as f:
        res=json.load(f)
    return res

def json2txt(jfile, respath, interactor=':'):
    #json文件转为txt，json的value默认是list形式
    if isinstance(jfile, str):
        logger.info("loading json file : %s" % jfile)
        f = codecs.open(jfile)
        d=json.load(f)
        f.close()
    else:
        d=jfile
    logger.info("writing txt file : %s" % respath)
    with codecs.open(respath,'a+',encoding=bianma) as ff:
        for k in d.keys():
            v=d[k]
            if isinstance(v,list):
                l=k+interactor+' '.join(v)+'\n'
            else:
                l = k + interactor + v + '\n'
            ff.write(l)

def printjson(j,limit=10):
    cnt=0
    for k in j.keys():
        if limit and cnt>=limit:
            break
        l = k + ' : ' + ' '.join(j[k]) + '\n'
        print(l)
        cnt+=1

def fieldcode_precess(s):
    '''
    对专题子栏目代码做规范化，例如缺失下划线A00258(A002_58)或有.或长度小于4或下划线在最后
    :param s: 
    :type s:str 
    :return: 
    :rtype: str
    '''

    if '.' in s:
        s = s.replace('.','')
    if s[-1] is '_':
        s=s[:-1]
    if len(s) < 4:
        return 'null'
    if len(s) == 4 or '_' in s:
        return s
    return s[:4]+'_'+s[4:]


def load2dic_02(path,separator=None):
    #把文件加载成字典。文件中每行第一个元素是value，后续元素分别是key
    #例：abcd AA BB   则输出 {AA:abcd,BB:abcd}
    logger.info("loading file2dic_02 : %s" % path)
    res = {}
    cnt=0
    if os.path.isfile(path):
        with codecs.open(path, 'rU', encoding=bianma, errors='replace') as f:
            for l in f:
                if cnt%1000000==0:
                    print(cnt)
                cnt+=1
                ll = l.strip().split()
                if len(ll) > 1:
                    value=ll[0]
                    keys=ll[1:]
                    for k in keys:
                        if k:
                            k = fieldcode_precess(k)
                            if res.has_key(k):
                                tmpl=res[k]
                                tmpl.append(value)
                                res[k]=tmpl
                            else:
                                res[k]=[value]
    print("all keys : %d all fn : %d" % (len(res),cnt))
    return res

def load2dic(path, separator=None,value2list=False,interactor=';'):
    '''
    把文件加载成字典。其中，文件每行第一个元素是key，后续元素组成列表或字符串做value
    例：abcd AA BB   则输出 {abcd:[AA,BB]} or {abcd:'AA;BB'}
    :param path: 
    :param separator: 
    :return: 
    :rtype: dict
    '''
    logger.info("loading file2dic : %s" % path)
    res = {}
    if os.path.isfile(path):
        with codecs.open(path, 'rU', encoding=bianma, errors='replace') as f:
            for l in f:
                ll = l.strip().split(separator)
                if len(ll)<2:
                    continue
                else:
                    if value2list:
                        res[ll[0]]=ll[1:]
                    else:
                        res[ll[0]]=interactor.join(ll[1:])
    print("all lines : %d" %len(res))
    return res

def load2dic_wc(path,separator=None):
    '''
    把文件加载成字典，文件每行每个元素是key，词频是value
    :param path: 
    :param separator: 
    :return: 
    :rtype: dict
    '''
    logger.info("loading file2dic_wc : %s" % path)
    res={}
    if os.path.isfile(path):
        with codecs.open(path, 'rU', encoding=bianma, errors='replace') as f:
            for l in f:
                ll=l.strip().split(separator)
                for i in ll:
                    res[i]=res.get(i,0)+1
    print("all lines : %d" % len(res))
    return res

if __name__ == '__main__':
    json2txt('./data/cluster/w2vkw1811_sgns_code/data_wv/A001/dic_center2words_A001_21_00107.json', 'tmp.txt')
