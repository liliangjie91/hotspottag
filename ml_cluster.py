#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/20
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import time,os,re,json
# import kbase
import numpy as np
import util_common as util
import util_path as path
import ml_prepare as mlpre
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

def vec_cluster(vecs,respath,true_k=None,subfix=''):
    # 对vecs数据做聚类kmeans，返回聚类结果和ch评分
    if not true_k:
        # logger.info("using auto k ...")
        num_samp=len(vecs)
        if num_samp>10000000:
            true_k=num_samp/2000 #1000
        elif num_samp>1000000:
            true_k=num_samp/2000 #500
        elif num_samp>100000:
            true_k=num_samp/1000  #200
        elif num_samp>10000:
            true_k=num_samp/500  #100
        elif num_samp>1000:
            true_k=num_samp/200   #50
        else:
            true_k=num_samp/50   #40
    cresname="/cres_%s_k%d.txt" %(subfix,true_k) if subfix else "/cres_k%d.txt" %true_k
    respath=respath+cresname
    minibatch=0 if len(vecs)<10000 else 1
    if minibatch:
        print("doing minibatchkmeans...")
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=true_k*3, batch_size=1000, verbose=False)
    else:
        print("doing basic kmeans...")
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                    verbose=False)
    print("-------Clustering begin...")
    t1 = time.time()
    cresraw=km.fit_predict(vecs)
    # np.savetxt(respath+'.np',cresraw)
    result = cresraw.tolist()
    util.list2txt(result, respath)
    print("-------Clustering train cost %s s" % (time.time() - t1))
    # kmscore = metrics.calinski_harabaz_score(vecs, result)
    return true_k, result

def get_cluter_center(cres,vecs,words,respath,prefix):
    # 获取聚类结果的类中心(其实是离类中心最近的那个样本)
    assert len(cres) == len(vecs)
    dic_cluser = {} #{类号：[样本id,...]}
    dic_vecs={} #{类号：[[样本vec],[...]]}
    dic_words={} #{类号：[[样本词],[...]]}
    dic_center={} #{类号：中心词}
    dic_res={} #{词：中心词}
    dic_res2={} #{中心词：[词1，词2，...]} 与dic_res相对应
    res_ccenter_fns = []  # 输出list，即每个类的类中心文件
    logger.info("getting dict of cluster_number and userlist...")
    for i,v in enumerate(cres):  # 对所有的样本，转字典：{类号：[样本id]}
        if not dic_cluser.has_key(v):
            dic_cluser[v] = [i]
            dic_vecs[v] = [vecs[i]]
            dic_words[v] = [words[i]]
        else:
            tmpli = dic_cluser[v]
            tmpli.append(i)
            dic_cluser[v] = tmpli
            tmplv = dic_vecs[v]
            tmplv.append(vecs[i])
            dic_vecs[v] = tmplv
            tmplw = dic_words[v]
            tmplw.append(words[i])
            dic_words[v] = tmplw
    logger.info("getting min dist id...")
    for j in dic_cluser.keys():  # 对所有类号，计算类中心和类中心最近id
        # logger.info("cluster num: %d" %j)
        cluster_index_list = dic_cluser[j]
        cluster_vec_mat_percenter = dic_vecs[j]
        tmpmat=np.array(cluster_vec_mat_percenter)
        centervec=np.mean(tmpmat,axis=0)
        dists=np.linalg.norm(tmpmat-centervec,axis=1)
        minindex=np.argmin(dists)
        minword=words[cluster_index_list[minindex]]
        dic_center[j]=minword
        dic_res2[minword]=dic_words[j]
    util.savejson('%s/dic_center2words_%s.json' %(respath,prefix), dic_res2)
    del dic_res2
    for i,w in enumerate(words):
        dic_res[w]=dic_center[cres[i]]
    util.savejson('%s/dic_word2center_%s.json' %(respath,prefix),dic_res)
    return dic_res



def run01():
    #普通聚类流程：获取vec，words(从gensim vector文件)--按不同k聚类--保存结果
    vecpath=path.path_model+'/w2v_kws1811_d300w8minc1iter5_sgns/w2v_kws1811_d300w8minc1iter5_sgns.vector'
    #vecpath=path.path_model+'/w2v_kws1811_d300w5minc3iter5_cbowns/w2v_kws1811_d300w5minc3iter5_cbowns.vector'
    fn_kwords_path=path.path_dataroot + '/other/fn_kws_1811'
    clusterpath=path.path_dataroot+'/cluster/w2vkw1811_sgns'
    words,vecs=mlpre.get_w_v_all(vecpath)
    vecl = vecs.tolist()
    del vecs
    for k in [50000,10000,5000,1000]:
        kmscore,cres=vec_cluster(vecl,clusterpath,true_k=k)
        dic_word2center=get_cluter_center(cres,vecl,words,clusterpath,'k%05d' %k)
    del vecl,words

def run02():
    #按照专题子栏目代码分别获取vec，words
    vecpath = path.path_model + '/w2v_kws1811_d300w8minc1iter5_sgns/w2v_kws1811_d300w8minc1iter5_sgns.vector'
    fn_kwords_path = path.path_dataroot + '/others/fn_kws_1811'
    fn_fcode_path = path.path_dataroot + '/others/ulog1811_fncode'
    codekws_path = path.path_dataroot + '/others/ulog1811_code_kws.json'
    clusterpath = path.path_dataroot + '/cluster/w2vkw1811_sgns_code/data_wv'
    # dic_code_fn=util.load2dic_02(fn_fcode_path,';')
    # dic_fn_kws=util.load2dic(fn_kwords_path)
    # dic_code_kws=mlpre.get_kwsfromfn_bycode(dic_code_fn,dic_fn_kws,codekws_path)
    dic_code_kws=json.load(open(codekws_path))
    mlpre.get_w_v_bycode(vecpath,dic_code_kws,clusterpath)

def run03():
    #根据run02中获取的按专题子栏目代码分类的words，vecs，再聚类(此处不再设k值，而是自动生成k)
    basefolder=path.path_dataroot + '/cluster/w2vkw1811_sgns_code/data_wv'
    kwordsfile_bycode = util.getfileinfolder(basefolder,prefix='data_wv.*words_.*txt',recurse=True,maxdepth=2)
    cnt = 0
    for kwf in kwordsfile_bycode:
        filesplit=os.path.split(kwf)
        tmpfolder=filesplit[0]
        code=filesplit[1][6:-4]
        if len(util.getfileinfolder(tmpfolder,prefix='cres.*%s.*txt' %code))==0:
            cnt+=1
            logger.info("clustering for %s" %code)
            vecpath=os.path.join(tmpfolder,'vecs_%s.txt' %code)
            tmpwords=util.load2list(kwf)
            tmpvecnp=np.loadtxt(vecpath)
            assert len(tmpvecnp)==len(tmpwords)
            vecl = tmpvecnp.tolist()
            true_k, cres = vec_cluster(vecl, tmpfolder, subfix=code)
            dic_word2center = get_cluter_center(cres, vecl, tmpwords, tmpfolder, "%s_%05d" %(code,true_k))
            del tmpwords,tmpvecnp,vecl,dic_word2center
        else:
            logger.info("cluster res : cres_%s_kxxx.txt allready exist!" %code)
    logger.info("clustering times : %d/%d" %(cnt,len(kwordsfile_bycode)))



if __name__ == '__main__':
    # run01()
    # run02()
    run03()