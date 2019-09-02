#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/7 
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors
import data_preprocess as dp
import util_common as uc
import util_path as path
import numpy as np
import os,time,json
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("./logs/ml_prepare.log")
ch = logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
# logger.addHandler(ch)

'''
本段主要涉及到向量数据的准备工作，主要指词向量模型训练完成之后，根据样本获取特定的向量以送入模型
'''

def get_samplevec_gensimmodel(vec_u, vec_fn, samplefile):
    #根据样本文件(uid+fid)获取用户向量vec_u和物品向量vec_fn。并储存
    data=[]
    label=[]
    print('loading vecfile : %s' % vec_u)
    # muser=Doc2Vec.load(usermodel)
    v_user = load_vec(vec_u)
    print('loading vecfile : %s' % vec_fn)
    v_file = load_vec(vec_fn)
    sample00=uc.load2list(samplefile)
    cnt=0
    for l in sample00:
        if cnt%10000==0:
            print(cnt)
        cnt+=1
        if cnt==1000:
            break
        l=l.strip().split()
        label0=l[1]
        uid='*dt_' + l[0].split("+")[0]
        fn='*dt_' + l[0].split("+")[1]
        if uid in v_user and fn in v_file:
            uvec=list(v_user[uid])
            fvec=list(v_file[fn])
            sampvec=uvec+fvec
            data.append(sampvec)
            label.append(label0)
    del v_file
    del v_user
    np.savetxt('./test.txt',np.array(data))
    # util.list2txt(label,'./lable.txt')

def load_vec(vecfilepath,norm=True,replace=True):
    #加载词向量文件（非词向量模型）做标准化，且替换原非标准化的vector
    vect=KeyedVectors.load_word2vec_format(vecfilepath)
    if norm:
        vect.init_sims(replace=replace) #做向量归一化即生成vectors_norm
    return vect

def get_code_kws_dict(dic_code_fns, fn_kws, stopwords=None, respath=None, topkw=0, codesubfix=''):
    '''
    根据子栏目代吗-fn文件和fn-关键词文件，获取子栏目代码-关键词文件 
    '''
    res={}
    logger.info("getting code-[kws] dict.")
    if stopwords:
        print('using stopwords.')
    for k in dic_code_fns.keys():
        if codesubfix in k[:4]:
            tmpkws=[]
            tmpfns=dic_code_fns[k]
            logger.info("for code : %s" %k)
            for fn in tmpfns:
                if fn in fn_kws:
                    tmpl = []
                    if stopwords:
                       for w in fn_kws[fn]:
                           if not w in stopwords:
                               tmpl.append(w)
                    else:
                        tmpl=fn_kws[fn]

                    if topkw:
                        tmpkws.extend(tmpl[:topkw])
                    else:
                        tmpkws.extend(tmpl)
            res[k]=tmpkws
    if respath:
        uc.savejson(respath, res)
    return res

def get_w_v_all(vecfilepath):
    vect=load_vec(vecfilepath)
    words=vect.index2word
    vecs=vect.vectors_norm
    # words 和 vecs 顺序是一样的，即index相同
    return words,vecs

def get_w_v_bycode(vecfilepath,dic_code_kws,respath,ifstopword=False,tfidffolder=None,thre=0.1):
    '''
    根据词向量和专题子栏目代码-关键词字典获取每个子栏目代码下的关键词及其向量
    :param vecfilepath: 词向量模型对应的词向量
    :type vecfilepath: str
    :param dic_code_kws: key：专题子栏目代码 value：该代码下所有的关键词
    :type dic_code_kws: dict
    :return: 
    :rtype: 
    '''
    vect = load_vec(vecfilepath)
    if not os.path.exists(respath):
        os.mkdir(respath)

    for k in dic_code_kws.keys():
        print("for code %s" %k)
        if '_' in k:
            basepath = respath + '/' + k.split('_')[0]
        else:
            basepath = respath + '/others'
        if not os.path.exists(basepath):
            os.mkdir(basepath)
        resfileword=basepath+'/words_'+k+'.txt'
        resfilevec=basepath+'/vecs_'+k+'.txt'
        if os.path.exists(resfileword) and os.path.exists(resfilevec):
            print("file %s already exists,skip code %s..." %(resfileword,k))
            continue
        curkws=dic_code_kws[k]
        if len(curkws)>50:
            words=[]
            vecs=[]
            curkws_uniq=list(set(curkws))   #去重
            stopwords=set()
            if ifstopword:
                #引入各个code下的tfidf，并计算出一定比例下的停用词
                tfidfdpath=tfidffolder+'/%s.json' %k
                tfidfd=uc.loadjson(tfidfdpath) if os.path.exists(tfidfdpath) else {}
                stopwords = dp.genTFIDF_stopwords_step2(tfidfd,thre)
                logger.info("get %d stopwords at thre=%f" %(len(stopwords),thre))
            for w in curkws_uniq:
                if w in vect and w not in stopwords:
                    words.append(w)
                    vec_norm=vect.vectors_norm[vect.vocab[w].index]
                    vecs.append(vec_norm)
            if words:
                print("saving data for code %s get res %d" %(k,len(words)))
                uc.list2txt(words, resfileword)
                np.savetxt(resfilevec,np.array(vecs))
    print("get words & vecs by code done!")

def load_allcresjson(basefolder,aim_pattern):
    #聚类bycode模式完成后，分别获取这些聚类结果，然后用户predict
    #结果文件是一个二级字典。第一级key为code，value为一个字典，第二级字典key为该code下的词，value为中心词
    # basefolder = path.path_dataroot + '/cluster/w2vkw1811_sgns_code/data_wv'
    # aim_pattern = r'data_wv.*dic_word2center_.*json'
    logger.info('loading cluster res json file in : %s' %basefolder)
    cresw2cdicfile = uc.getfileinfolder(basefolder, prefix=aim_pattern, recurse=2)
    allcresdic={}
    cnt=0
    for fj in cresw2cdicfile:
        if cnt%(len(cresw2cdicfile)/20)==0:
            print('loading cluster res file : %d/%d' %(cnt,len(cresw2cdicfile)))
        cnt+=1
        filesplit = os.path.split(fj)
        jname=filesplit[1]
        code = jname[16:jname.rfind('_')]
        allcresdic[code]=json.load(open(fj))
    logger.info("loading all cres dic ok.")
    return allcresdic

def get_goodfn_byfn(rawfns,respath):
    #选择优质杂志的文献
    goodj = uc.load2dic(path.good_journal)
    if isinstance(rawfns,str):
        fns=uc.load2list(rawfns)
    else:
        fns=rawfns
    res=[]
    cnt=-1
    for fn in fns:
        cnt+=1
        if cnt%500000==0:
            print(cnt)
        if fn[:4] in goodj or fn[-3:] == '.NH':
            res.append(fn)
    uc.list2txt(res, respath)

def get_highqsample(goodfns,input1,respath):
    gfns=uc.load2list(goodfns)
    resdic={}
    indic=uc.load2dic(input1, interactor=' ')
    cnt=-1
    for fn in gfns:
        cnt+=1
        if cnt%500000==0:
            print(cnt)
        if fn in indic:
            resdic[fn]=indic[fn]
    if resdic:
        uc.json2txt(resdic, respath, interactor=' ')





if __name__ == '__main__':
    # get_samplevec_gensimmodel(path.path_model + '/d2v_udownhighq5wposi_d300w5minc3iter30_dmns/d2v_udownhighq5wposi_d300w5minc3iter30_dmns.dv',
    #                           path.path_model + '/d2v_highq5w_l1t1_d300w5minc3iter30_dmns/d2v_highq5w_l1t1_d300w5minc3iter30_dmns.dv',
    #                           path.path_datahighq5w + '/sample_highq5w_neg.txt')
    # get_goodfn_byfn(path.path_dataraw+'/fns1811',path.path_dataraw+'/fns1811ingoodj')
    # get_highqsample(path.path_dataraw+'/fns1811ingoodj',path.path_dataraw+'/fn_code_1811_seg',path.path_dataraw+'/highqpaper/fn_code_1811_seg')
    # caculate_idf(path.path_dataraw+'/fn_kwsraw1811bycode/I/I135_52.txt',path.path_dataraw+'/fn_kwsraw1811bycode/test')
    pass
