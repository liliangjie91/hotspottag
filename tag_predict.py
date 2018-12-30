#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/27
import util_path as path
import util_common as util
import os,time,json,re
import util_dbkbase
import ml_prepare as mlpre
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("./logs/tag_predict.log")
ch = logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
# logger.addHandler(ch)
CUR_FNCODE_DIC=util.load2dic(path.path_dataraw+'/fn_code_1811_seg')
CUR_FNKWS_DIC=util.load2dic(path.path_dataraw+'/fn_kws_1811')
CUR_FIELDCODE_DIC=util.load2dic('./source/field_dic.txt')

def get_code_name(code):
    res=''
    if code in CUR_FIELDCODE_DIC:
        res=CUR_FIELDCODE_DIC[code][0]
        # if '_' in code:
        #     subcode=code[:code.find('_')]
        #     if subcode in CUR_FIELDCODE_DIC:
        #         res=CUR_FIELDCODE_DIC[subcode][0]+'_'+res
    return res

def predict_tag_byword4code(words, codes, dic_word2center_all, savecodename=True):
    # 预测关键词对应的标签，给一组关键词，输出对应标签
    res = []
    for code in codes:
        code = code.strip()
        codename = get_code_name(code)
        if code in dic_word2center_all:
            for w in words:
                if w in dic_word2center_all[code]:
                    if savecodename:
                        res.append(codename +'_' + dic_word2center_all[code][w])
                    else:
                        res.append(dic_word2center_all[code][w])
    return list(set(res))

def predict_tag_byfn4code(fns,dic_word2center_all):
    dic_res={}
    needkbasefn = []
    for fn in fns:
        if fn in CUR_FNCODE_DIC and fn in CUR_FNKWS_DIC:
            words = CUR_FNKWS_DIC[fn]
            codes = CUR_FNCODE_DIC[fn]
            dic_res[fn] = predict_tag_byword4code(words, codes, dic_word2center_all)
        else:
            needkbasefn.append(fn)
    kbasekws, kbasecodes = get_kwords_kbase(needkbasefn)
    for i, fn in enumerate(needkbasefn):
        dic_res[fn] = predict_tag_byword4code(kbasekws[i], kbasecodes[i], dic_word2center_all)
    return dic_res

def predict_tag_byword4all(words, dic_word2center):
    # 预测关键词对应的标签，给一组关键词，输出对应标签
    res = []
    for w in words:
        if w in dic_word2center:
            res.append(dic_word2center[w])
    return list(set(res))

def predict_tag_byfn4all(fns, dic_word2center):
    # 对一组文件名，预测其标签，先从离线的fn_kwords查询文件名对应的关键词，查不到的再查kbase
    dic_res = {}  # {filename:[tag1,tag2...]}
    needkbasefn = []
    for fn in fns:
        if fn in CUR_FNKWS_DIC:
            words = CUR_FNKWS_DIC[fn]
            dic_res[fn] = predict_tag_byword4all(words, dic_word2center)
        else:
            needkbasefn.append(fn)
    kbasekws,codes = get_kwords_kbase(needkbasefn)
    for i, fn in enumerate(needkbasefn):
        dic_res[fn] = predict_tag_byword4all(kbasekws[i], dic_word2center)
    return dic_res

def get_kwords_kbase(inputlist):
    # 根据文件名去kbase查关键词
    if type(inputlist) is list:
        fns = inputlist
    else:
        fns = util.load2list(inputlist)
    kws = []  # 对应的关键词列表
    codes=[]
    kbpk = util_dbkbase.PooledKBase(max_connections_cnt=50, mode='cur')
    for i in fns:
        fn = i.strip()
        sql01 = u"SELECT 机标关键词,专题子栏目代码 from CJFDTOTAL,CDFDTOTAL,CMFDTOTAL WHERE 文件名='" + fn + u"'"
        thread_cur = kbpk.get_connection()
        ret = kbpk.cursor_execute_ksql(sql01, thread_cur, mode='all')
        if len(ret) == 0:
            kws.append([])
            codes.append([])
        else:
            kw = re.split(r'[,，;；]+', ret[0][0].strip())
            code = re.split(r';',ret[0][1].strip())
            kws.append(kw)
            codes.append(code)
    kbpk.close_connection()
    assert len(kws) == len(fns)
    return kws,codes

def run01():
    basefolder = path.path_dataroot + '/cluster/w2vkw1811_sgns_code/data_wv'
    aim_pattern = r'data_wv.*dic_word2center_.*json'
    dic_word2center_all = mlpre.load_allcresjson(basefolder,aim_pattern)
    fns=[]
    predict_tag_byfn4code(fns,dic_word2center_all)

if __name__ == '__main__':
    run01()