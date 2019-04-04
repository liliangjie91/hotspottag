#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/27
import util_path as path
import util_common as uc
import os,time,json,re,codecs
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
CUR_FNCODE_DIC=uc.load2dic(path.path_dataraw + '/fn_code_1811_seg')
# CUR_FNKWS_DIC=uc.load2dic(path.path_dataraw + '/fn_kws_1811')
# CUR_FNKWS_DIC_HIGHQ=uc.load2dic(path.path_dataraw + '/highqpaper/fn_kws_1811')
CUR_FNKWSRAW_BIGRAM=uc.load2dic(path.path_dataraw + '/fn_kwsraw1811bycode/tmp/fnKws_I_all.txt')
# CUR_FNKWSRAW_DIC=uc.load2dic(path.path_dataraw + '/fn_kwsraw_1811')
CUR_FNKWSRAW_DIC_HIGHQ=uc.load2dic(path.path_dataraw + '/highqpaper/fn_kwsraw_1811')
CUR_FIELDCODE_DIC=uc.load2dic('./source/field_dic.txt')

'''
标签预测：
1，为输入的词预测标签
2，为输入的文件名预测标签

'''
def predict_tag_byword4code(words, codes, word2center_dic, savecodename=True):
    '''
    预测关键词对应的标签，给一组关键词，输出对应标签 4code:按专题子栏目代码分类过
    :param words: 
    :type words: list
    :param codes: 
    :type codes: list
    :param word2center_dic: 
    :type word2center_dic: dict
    :return: 
    :rtype: 
    '''
    res = []
    if codes == [] or codes is None:
        codes = word2center_dic.keys()
    for code in codes:
        code = code.strip()
        codename , a = uc.get_code_field(code,dic_codefield=CUR_FIELDCODE_DIC)
        if code in word2center_dic:
            for w in words:
                if w in word2center_dic[code]:
                    if savecodename:
                        res.append(codename +'_' + word2center_dic[code][w])
                    else:
                        res.append(word2center_dic[code][w])
    return list(set(res))

def predict_tag_byfn4code(fns, word2center_dic, fnkwsdic, querrytokbase=True, savecodename=True):
    '''
    预测文献标签，输入文件名列表，以及关键词及中心字典 4code:按专题子栏目代码分类过
    :param fns: 
    :type fns: list
    :param word2center_dic: 
    :type word2center_dic: dict
    :return: 
    :rtype: dict
    '''
    dic_res={}
    needkbasefn = []
    cnt=-1
    alllen=len(fns)
    for fn in fns:
        cnt+=1
        if cnt%50000==0:
            logger.info("predict tag by fn 4 code %d/%d" %(cnt,alllen))
        if fn in CUR_FNCODE_DIC and fn in fnkwsdic:
            words = fnkwsdic[fn].strip(';').split(';')
            codes = CUR_FNCODE_DIC[fn].strip(';').split(';')
            dic_res[fn] = predict_tag_byword4code(words, codes, word2center_dic, savecodename=savecodename)
        else:
            needkbasefn.append(fn)
    if querrytokbase:
        kbasekws, kbasecodes = get_kwords_kbase(needkbasefn)
        for i, fn in enumerate(needkbasefn):
            dic_res[fn] = predict_tag_byword4code(kbasekws[i], kbasecodes[i], word2center_dic, savecodename=savecodename)
    return dic_res

def predict_tag_byword4all(words, dic_word2center):
    # 预测关键词对应的标签，给一组关键词，输出对应标签
    res = []
    for w in words:
        if w in dic_word2center:
            res.append(dic_word2center[w])
    return list(set(res))

def predict_tag_byfn4all(fns, fnkwsdic ,dic_word2center):
    # 对一组文件名，预测其标签，先从离线的fn_kwords查询文件名对应的关键词，查不到的再查kbase
    dic_res = {}  # {filename:[tag1,tag2...]}
    needkbasefn = []
    for fn in fns:
        if fn in fnkwsdic:
            words = fnkwsdic[fn]
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
        fns = uc.load2list(inputlist)
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
            code = re.split(r';',ret[0][1].strip().strip(';'))
            kws.append(kw)
            codes.append(code)
    kbpk.close_connection()
    assert len(kws) == len(fns)
    return kws,codes  #2d list

def tmp_mergeres(tagdic, kwdic, respath='predicttag_02.txt'):
    '''
    把标签结果写入文件
    :param tagdic: 
    :type tagdic:dict 
    :param kwdic: 
    :type kwdic: 
    :return: 
    :rtype: 
    '''
    logger.info("writing fn-tag res to file : %s ..." %respath)
    with codecs.open(respath, 'a+', encoding='utf8') as ff:
        for fn in tagdic.keys():
            tags = ';'.join(tagdic[fn])
            kw = kwdic[fn]
            l = '%s\t%s\t%s\n' % (fn, tags, kw)
            ff.write(l)

def predict_tag_run(basefolder, fnkwsdic, resprefix="test", getsubres=1000):
    '''
    分大类聚类完毕之后，对每个文件名获取其标签。
    :param basefolder: 存放聚类结果的文件夹
    :type basefolder: str
    :param fnkwsdic: 文件名-关键词 字典
    :type fnkwsdic: dict
    :param resprefix: 结果文件名前缀
    :type resprefix: str
    :param getsubres: 是否只获取部分结果
    :type getsubres: int
    :return: 
    :rtype: 
    '''
    aim_pattern = r'dic_word2center_.*json'
    logger.info("get fntag res , basefolder is : %s" %basefolder)
    dic_word2center_all = mlpre.load_allcresjson(basefolder+'/w2c',aim_pattern)
    # fns=[u'KJWH201503033',u'1018161746.NH',u'WSJJ201003003',u'KJWH201503030',u'WSJJ201003005']
    if dic_word2center_all:
        if getsubres:
            fns=fnkwsdic.keys()[:getsubres]
        else:
            fns=fnkwsdic.keys()
        res=predict_tag_byfn4code(fns,dic_word2center_all,fnkwsdic,querrytokbase=False,savecodename=False)
        # uc.json2txt(res,'predicttag_01.txt')
        if res:
            tmp_mergeres(res,kwdic=fnkwsdic,respath=basefolder+'/fn_tag_'+resprefix+'.txt')
        else:
            logger.info("fn-tag-dic is null")
    else:
        logger.info("dic_word2center_all is null")


if __name__ == '__main__':
    cres_kws_highq=path.path_datacluster+'/bigram_I_onetf1_bitf6_1_stoped2/embed_allkwsAbsTitle'
    # predict_tag_run(cres_kws_highq + '/cres01', CUR_FNKWS_DIC_HIGHQ, resprefix='kws_highq_cres01', getsubres=0)
    predict_tag_run(cres_kws_highq + '/cres02', CUR_FNKWSRAW_BIGRAM, resprefix='1811_I_bigram', getsubres=0)
    # predict_tag_run(cres_kws_highq + '/cres03', CUR_FNKWS_DIC_HIGHQ, resprefix='kws_highq_cres03', getsubres=0)

    # cres_kwsraw_highq = path.path_datacluster + '/w2vkwraw1811_sgns_code_highq'
    # predict_tag_run(cres_kwsraw_highq + '/cres01', CUR_FNKWSRAW_DIC_HIGHQ, resprefix='kwsraw_highq_cres01', getsubres=0)
    # predict_tag_run(cres_kwsraw_highq + '/cres02', CUR_FNKWSRAW_DIC_HIGHQ, resprefix='kwsraw_highq_cres02', getsubres=0)
    # predict_tag_run(cres_kwsraw_highq + '/cres03', CUR_FNKWSRAW_DIC_HIGHQ, resprefix='kwsraw_highq_cres03', getsubres=0)