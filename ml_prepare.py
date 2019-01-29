#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/7 
from gensim.models.doc2vec import Doc2Vec
from gensim.models import keyedvectors
import util_common as util
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


def get_samplevec_gensimmodel(vecpath1, vecpath2, samplefile):
    data=[]
    label=[]
    print('loading vecfile : %s' % vecpath1)
    # muser=Doc2Vec.load(usermodel)
    v_user = load_vec(vecpath1)
    print('loading vecfile : %s' % vecpath2)
    v_file = load_vec(vecpath2)
    sample00=util.load2list(samplefile)
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

def load_vec(vecfilepath,norm=True):
    vect=keyedvectors.Word2VecKeyedVectors.load_word2vec_format(vecfilepath)
    if norm:
        vect.init_sims() #做向量归一化即生成vectors_norm
    return vect

def get_kwsfromfn_bycode(dic_code_fns,fn_kws,respath=None,topkw=0):
    '''
    根据子栏目代吗-fn文件和fn-关键词文件，获取子栏目代码-关键词文件
    :param dic_code_fns: 
    :type dic_code_fns:dict 
    :param fn_kws: 
    :type fn_kws: dict
    :return: 
    :rtype: 
    '''
    res={}
    for k in dic_code_fns.keys():
        tmpkws=[]
        tmpfns=dic_code_fns[k]
        for fn in tmpfns:
            if fn_kws.has_key(fn):
                if topkw:
                    tmpkws.extend(fn_kws[fn][:topkw])
                else:
                    tmpkws.extend(fn_kws[fn])
        res[k]=tmpkws
    if respath:
        util.savejson(respath,res)
    return res

def get_w_v_all(vecfilepath):
    vect=load_vec(vecfilepath)
    vect.init_sims()
    words=vect.index2word
    vecs=vect.vectors_norm
    # words 和 vecs 顺序是一样的，即index相同
    return words,vecs

def get_w_v_bycode(vecfilepath,dic_code_kws,respath):
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
    vect.init_sims()
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
            for w in curkws_uniq:
                if w in vect:
                    words.append(w)
                    vec_norm=vect.vectors_norm[vect.vocab[w].index]
                    vecs.append(vec_norm)
            if words:
                print("saving data for code %s get res %d" %(k,len(words)))
                util.list2txt(words,resfileword)
                np.savetxt(resfilevec,np.array(vecs))
    print("get words & vecs by code done!")

def load_allcresjson(basefolder,aim_pattern):
    #聚类bycode模式完成后，分别获取这些聚类结果，然后用户predict
    # basefolder = path.path_dataroot + '/cluster/w2vkw1811_sgns_code/data_wv'
    # aim_pattern = r'data_wv.*dic_word2center_.*json'
    logger.info('loading cluster res json file in : %s' %basefolder)
    cresw2cdicfile = util.getfileinfolder(basefolder, prefix=aim_pattern, recurse=True, maxdepth=2)
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
    goodj = util.load2dic(path.good_journal)
    if isinstance(rawfns,(str,unicode)):
        fns=util.load2list(rawfns)
    else:
        fns=rawfns
    res=[]
    cnt=-1
    for fn in fns:
        cnt+=1
        if cnt%500000==0:
            print cnt
        if fn[:4] in goodj or fn[-3:] == '.NH':
            res.append(fn)
    util.list2txt(res,respath)

def get_highqsample(goodfns,input1,respath):
    gfns=util.load2list(goodfns)
    resdic={}
    indic=util.load2dic(input1,interactor=' ')
    cnt=-1
    for fn in gfns:
        cnt+=1
        if cnt%500000==0:
            print cnt
        if fn in indic:
            resdic[fn]=indic[fn]
    if resdic:
        util.json2txt(resdic,respath,interactor=' ')


if __name__ == '__main__':
    # get_samplevec_gensimmodel(path.path_model + '/d2v_udownhighq5wposi_d300w5minc3iter30_dmns/d2v_udownhighq5wposi_d300w5minc3iter30_dmns.dv',
    #                           path.path_model + '/d2v_highq5w_l1t1_d300w5minc3iter30_dmns/d2v_highq5w_l1t1_d300w5minc3iter30_dmns.dv',
    #                           path.path_datahighq5w + '/sample_highq5w_neg.txt')
    # get_goodfn_byfn(path.path_dataraw+'/fns1811',path.path_dataraw+'/fns1811ingoodj')
    get_highqsample(path.path_dataraw+'/fns1811ingoodj',path.path_dataraw+'/fn_code_1811_seg',path.path_dataraw+'/highqpaper/fn_code_1811_seg')
