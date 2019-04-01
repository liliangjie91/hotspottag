#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2019/3/11 

import util_path as path
import util_common as uc
import numpy as np
import ml_cluster as mlclu
import ml_prepare as mlpre
import data_preprocess as dp
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%d %b %y %H:%M:%S',
                    filename='./logs/all.log',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("./logs/cluster.log")
ch = logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
# logger.addHandler(ch)

def run01():
    '''
    1,生成初级停用词 自选+一元词+二元词
    2,根据初级停用词生成 code-kws 表
    3,计算各个code下词的tfidf，卡阈值之后，剩余的词用于本code下聚类
    :return: 
    :rtype: 
    '''
    basefolder = path.path_dataraw+'/fn_kwsraw1811bycode'

    #词向量模型的路径
        #仅使用中文关键词序列训练词向量
    vecpath0 = (path.path_model +
              '/w2v_kwsraw1811bycodeI_d300w8minc2iter5_sgns/w2v_kwsraw1811bycodeI_d300w8minc2iter5_sgns.normwv')

        #使用中文关键词-机标关键词-标题-摘要内容训练的词向量。训练参数：d300w5minc3iter5_sgns
    vecpath1 = (path.path_model +
               '/w2v_kws1811_I_allKwsAbsTitle_d300w5minc3iter5_sgns/w2v_kws1811_I_allKwsAbsTitle_d300w5minc3iter5_sgns.normwv')

        #使用中文关键词-机标关键词-标题-摘要内容训练的词向量。训练参数：d300w8minc2iter5_sgns
    vecpath2 = (path.path_model +
               '/w2v_kws1811_I_allKwsAbsTitle_d300w8minc2iter5_sgns/w2v_kws1811_I_allKwsAbsTitle_d300w8minc2iter5_sgns.normwv')

    #聚类数据的路径
    clusterpath = path.path_dataroot + '/cluster/bigram_I_onetf1_bitf6_1_stoped2/embed_allkwsAbsTitle/data_wv'

    #停用词相关的路径
    stopwords_allready = uc.load2list(basefolder+'/stopword/stopws_allByCode_o1b6_tf_step1.txt')
    stopwords_humanselect = uc.load2list('./otherfile/tfidf_stopwords_I.txt')


    #文件名-关键词-专题子栏目代码等文件的路径
    fn_kwords_path = path.path_dataraw + '/fn_kwsraw1811bycode/fnKws_I_all.txt'
    fn_fcode_path = path.path_dataraw + '/fn_code_1811_seg'
    codekws_path = path.path_dataraw + '/fn_kwsraw1811bycode/codeKwsDict/bigram_I_codekws_stoped1.json'

    #tfidf文件的路径
    idf_allwords = path.path_dataraw + '/fn_kwsraw1811bycode/tfidf/tfidf_I_allByCode_idf.json'
    tfidf_bycode_folder = path.path_dataraw + '/fn_kwsraw1811bycode/tfidf/bycode'

    topk = 0
    codesubfix = 'I'
    logger.info("getting code-[kes] dict topk = %d code subfix = %s" % (topk, codesubfix))
    # tfidfStopwords=[]
    # stopwords2use = set(stopwords_allready + stopwords_humanselect)

    # 加载fn-code文件，并转换成code-[fn]字典
    # dic_code_fn = uc.load2dic_02(fn_fcode_path)
    # 加载fn-keywords文件，并转换成fn-[kw]字典
    # dic_fn_kws = uc.load2dic(fn_kwords_path, value2list=True)
    # 获取code-kws文件
    # dic_code_kws = mlpre.get_kwsfromfn_bycode(dic_code_fn, dic_fn_kws,
    #                                           respath=codekws_path, topkw=topk,
    #                                           stopwords=stopwords2use, codesubfix=codesubfix)

    # 根据上一步获取的文件，计算各code下词的tfidf
    # dp.caculate_tfidf_code(dic_code_kws, idf_allwords, respath=tfidf_bycode_folder)

    # 对每个code下的tfidf，做掐头去尾，获取剩余的词，并确定这些词的词向量，送入聚类
    if not os.path.exists(clusterpath):
        os.makedirs(clusterpath)
    print("\n\n**************cf=%s*****************\n\n" % clusterpath)
    dic_code_kws = uc.loadjson(codekws_path)
    mlpre.get_w_v_bycode(vecpath2, dic_code_kws, clusterpath,ifstopword=True,tfidffolder=tfidf_bycode_folder)

    # 聚类
    mlclu.run03(ktype=1, basefolder=clusterpath)
    mlclu.run03(ktype=2, basefolder=clusterpath)
    mlclu.run03(ktype=3, basefolder=clusterpath)

if __name__ == '__main__':
    run01()