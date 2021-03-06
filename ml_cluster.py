#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/20
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import time,os,re,json
# import kbase
import numpy as np
import util_common as uc
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

'''
聚类分析
'''


def vec_cluster(vecs,respath,ktype,true_k=None,subfix=''):
    # 对vecs数据做聚类kmeans，返回聚类结果和ch评分
    num_per_clsses=[[2000,500,200,100,50,40],
                    [2000,2000,1000,500,200,50],
                    [2000,200,100,50,30,20],
                    [2000,100,50,30,20,10]] #根据样本量大小来确定聚类k值的大小
    if not true_k:
        # logger.info("using auto k ...")
        num_samp=len(vecs)
        if num_samp>10000000:
            true_k=num_samp//num_per_clsses[ktype][0] #1000,2000,2000
        elif num_samp>1000000:
            true_k=num_samp//num_per_clsses[ktype][1] #500,2000,200,100
        elif num_samp>100000:
            true_k=num_samp//num_per_clsses[ktype][2]  #200,1000,100,50
        elif num_samp>10000:
            true_k=num_samp//num_per_clsses[ktype][3]  #100,500,50,30
        elif num_samp>1000:
            true_k=num_samp//num_per_clsses[ktype][4]   #50,200,30,20
        else:
            true_k=num_samp//num_per_clsses[ktype][5]   #40,50,20,10

    cresname="/cres_%s_k%d.txt" %(subfix,true_k) if subfix else "/cres_k%d.txt" %true_k
    respath=respath+'/cres'+cresname
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
    uc.list2txt(result, respath)
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
        if v not in dic_cluser:
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
    uc.savejson('%s/c2w/dic_center2words_%s.json' % (respath, prefix), dic_res2)
    uc.json2txt(dic_res2,'%s/c2w/txt_dic_center2words_%s.txt' % (respath, prefix))
    del dic_res2
    for i,w in enumerate(words):
        dic_res[w]=dic_center[cres[i]]
    uc.savejson('%s/w2c/dic_word2center_%s.json' % (respath, prefix), dic_res)
    return dic_res

def get_cluter_center_tfidf(cres,vecs,words,respath,prefix,tfidf=None,tf=None,idf=None):
    # 获取聚类结果的类中心--选取类内tfidf前两大值。若多个，取离类中心最近的两个
    if os.path.exists('%s/c2w/dic_center2words_%s_toptfidf.json' % (respath, prefix)):
        print("res allready exists %s/c2w/dic_center2words_%s_toptfidf.json" % (respath, prefix))
        return
    assert len(cres) == len(vecs)
    dic_cluser = {} #{类号：[样本id,...]}
    dic_vecs={} #{类号：[[样本vec],[...]]}
    dic_words={} #{类号：[[样本词],[...]]}
    dic_tf={} #{类号：[[tf1],[...]]}
    dic_idf={} #{类号：[[idf1],[...]]}
    dic_tfidf = {}  # {类号：[[tfidf1],[...]]}
    dic_center={} #{类号：中心词}
    dic_res_w2c={} #{词：中心词}
    dic_res_c2w={} #{中心词：[词1，词2，...]} 与dic_res_w2c相对应
    mean_idf = np.mean(list(tfidf.values())) if tfidf else np.mean(list(idf.values()))
    logger.info("getting dict of cluster_number and userlist...")
    for i,v in enumerate(cres):  # 对所有的样本，转字典：{类号：[样本id]}
        if v not in dic_cluser:
            dic_cluser[v] = [i]
            dic_vecs[v] = [vecs[i]]
            dic_words[v] = [words[i]]
            if tfidf:
                dic_tfidf[v] = [tfidf[words[i]] if words[i] in tfidf else mean_idf]
            else:
                dic_tf[v] = [tf[words[i]] if words[i] in tf else 1]
                dic_idf[v] = [idf[words[i]] if words[i] in idf else mean_idf]
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

            if tfidf:
                tmptfidf = dic_tfidf[v]
                tmptfidf.append(tfidf[words[i]] if words[i] in tfidf else mean_idf)
                dic_tfidf[v] = tmptfidf
            else:
                tmptf = dic_tf[v]
                tmptf.append(tf[words[i]] if words[i] in tf else 1)
                dic_tf[v] = tmptf
                tmpidf = dic_idf[v]
                tmpidf.append(idf[words[i]] if words[i] in idf else mean_idf)
                dic_idf[v]=tmpidf

    logger.info("getting top tfidf...")
    for clust_id in dic_cluser.keys():  # 对所有类号，计算类中心和类中心最近id
        # logger.info("cluster num: %d" %j)
        tmp_words = dic_words[clust_id]
        if tfidf:
            tmp_tfidf_l = dic_tfidf[clust_id]
        else:
            tmp_tf_l = dic_tf[clust_id]
        # tmp_tf_all = np.sum(tmp_tf_l)
            tmp_idf_l = dic_idf[clust_id]
            tmp_tfidf_l = np.multiply(tmp_tf_l,tmp_idf_l)
        tmp_w_tfidf = list(zip(tmp_words,tmp_tfidf_l))
        tmp_w_tfidf_sorted = sorted(tmp_w_tfidf,key=lambda x:x[1],reverse=True)[:2]
        tag_l=[tmp_w_tfidf_sorted[0][0]]
        if len(tmp_w_tfidf_sorted)>1:
            tag_l.append(tmp_w_tfidf_sorted[1][0])
        tags=tag_l[0] #top1 tfidf 词做tag
        # tags=';'.join(tag_l) #top2 tfidf 词做tag
        dic_center[clust_id]=tags
        dic_res_c2w[tags]=tmp_words
    uc.savejson('%s/c2w/dic_center2words_%s_toptfidf.json' % (respath, prefix), dic_res_c2w)
    uc.json2txt(dic_res_c2w,'%s/c2w/txt_dic_center2words_%s_toptfidf.txt' % (respath, prefix))
    del dic_res_c2w
    for i,w in enumerate(words):
        dic_res_w2c[w]=dic_center[cres[i]]
    uc.savejson('%s/w2c/dic_word2center_%s_toptfidf.json' % (respath, prefix), dic_res_w2c)
    return dic_res_w2c

def run00():
    # 普通聚类流程：获取vec，words(从gensim vector文件)--按不同k聚类--保存结果
    vecpath = path.path_model + '/w2v_kws1811_d300w8minc1iter5_sgns/w2v_kws1811_d300w8minc1iter5_sgns.vector'
    # vecpath=path.path_model+'/w2v_kws1811_d300w5minc3iter5_cbowns/w2v_kws1811_d300w5minc3iter5_cbowns.vector'
    fn_kwords_path = path.path_dataroot + '/other/fn_kws_1811'
    clusterpath = path.path_dataroot + '/cluster/w2vkw1811_sgns'
    words, vecs = mlpre.get_w_v_all(vecpath)
    vecl = vecs.tolist()
    del vecs
    for k in [50000, 10000, 5000, 1000]:
        kmscore, cres = vec_cluster(vecl, clusterpath, 0, true_k=k)
        dic_word2center = get_cluter_center(cres, vecl, words, clusterpath, 'k%05d' % k)
    del vecl, words

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
        kmscore,cres=vec_cluster(vecl,clusterpath,0,true_k=k)
        dic_word2center=get_cluter_center(cres,vecl,words,clusterpath,'k%05d' %k)
    del vecl,words

def run02(subfix=None):
    #按照专题子栏目代码分别获取vec，words

    # vecpath = path.path_model + '/w2v_kws1811_d300w8minc1iter5_sgns/w2v_kws1811_d300w8minc1iter5_sgns.vector'
    # vecpath = path.path_model + '/w2v_kws1811_raw_d300w8minc2iter5_sgns/w2v_kws1811_raw_d300w8minc2iter5_sgns.normwv'
    vecpath = path.path_model + '/w2v_kwsraw1811bycodeI_d300w8minc2iter5_sgns/w2v_kwsraw1811bycodeI_d300w8minc2iter5_sgns.normwv'
    # fn_kwords_path = path.path_dataroot + '/data_raw/highqpaper/fn_kws_1811'
    fn_kwords_path = path.path_dataraw + '/fn_kwsraw1811bycode/fnKws_I_all.txt'
    fn_fcode_path = path.path_dataroot + '/data_raw/fn_code_1811_seg'
    # TFIDF_stopwords = path.path_dataraw + '/fn_kwsraw1811bycode/bigram_I_kwsraw_stopwords_onetf5_bitf5.txt'
    TFIDF_stopwords = (path.path_dataraw +
                       '/fn_kwsraw1811bycode/stopword/stopws_allByCode_o1b6_tf_step1.txt') # 初级停用词人工+onegram1+bigram6
    print(fn_fcode_path)
    # codekws_path = path.path_dataroot + '/data_raw/highqpaper/ulog1811_code_kws_top3.json'
    # codekws_path = path.path_dataraw + '/fn_kwsraw1811bycode/bigram_I_codekws_%s.json' %subfix
    codekws_path = path.path_dataraw + '/fn_kwsraw1811bycode/codeKwsDict/bigram_I_codekws_stoped1.json'
    # clusterpath = path.path_dataroot + '/cluster/w2vkw1811_sgns_code_highq/data_wv_top3'
    clusterpath = path.path_dataroot + '/cluster/bigram_I_%s/data_wv' %subfix
    if not os.path.exists(clusterpath):
        os.makedirs(clusterpath)
    print("\n\n**************cf=%s*****************\n\n" %clusterpath)
    #加载fn-code文件，并转换成code-[fn]字典
    # dic_code_fn=uc.load2dic_02(fn_fcode_path)

    #加载fn-keywords文件，并转换成fn-[kw]字典
    # dic_fn_kws=uc.load2dic(fn_kwords_path, value2list=True)

    #根据上述两文件获取code-[kws...]字典
    topk=0
    codesubfix='I'
    logger.info("getting code-[kes] dict topk = %d code subfix = %s" %(topk,codesubfix))
    # tfidfStopwords=uc.load2list(TFIDF_stopwords)
    # tfidfStopwords=[]
    # dic_code_kws=mlpre.get_kwsfromfn_bycode(dic_code_fn,dic_fn_kws,
    #                                         respath=codekws_path,topkw=topk,
    #                                         stopwords=set(tfidfStopwords),codesubfix=codesubfix)
    dic_code_kws=json.load(open(codekws_path))
    #根据上述code-[kws...]字典，获取具体的words和vecs文件
    mlpre.get_w_v_bycode(vecpath,dic_code_kws,clusterpath)
    return clusterpath

def run03(ktype,basefolder=None):
    #根据run02中获取的按专题子栏目代码分类的words，vecs，再聚类(此处不再设k值，而是自动生成k,而自动生成k有几种模式，通过ktype来选择)

    kwordsfile_bycode = uc.getfileinfolder(basefolder, prefix='data_wv.*words_.*txt', recurse=2)
    cnt = 0
    # 路径设置
    resfoldername = "cres%02d" %ktype
    resfolder = os.path.join(basefolder, os.path.pardir) + '/' + resfoldername
    if os.path.exists(resfolder):
        logger.info("resfolder %s allready exists!!" % resfolder)
        return
    else:
        os.mkdir(resfolder)
        os.mkdir(resfolder + '/cres')
        os.mkdir(resfolder + '/c2w')
        os.mkdir(resfolder + '/w2c')

    #聚类
    for kwf in kwordsfile_bycode:
        # 路径设置
        filesplit=os.path.split(kwf)
        tmpfolder=filesplit[0]
        # 聚类
        code=filesplit[1][6:-4]
        if len(uc.getfileinfolder(tmpfolder, prefix='cres.*%s.*txt' %code))==0:
            cnt+=1
            logger.info("clustering for %s" %code)
            vecpath=os.path.join(tmpfolder,'vecs_%s.txt' %code)
            tmpwords=uc.load2list(kwf)
            tmpvecnp=np.loadtxt(vecpath)
            assert len(tmpvecnp)==len(tmpwords)
            vecl = tmpvecnp.tolist()
            if len(vecl)<50: #100
                continue
            true_k, cres = vec_cluster(vecl, resfolder, ktype=ktype,subfix=code)
            dic_word2center = get_cluter_center(cres, vecl, tmpwords, resfolder, "%s_%05d" %(code,true_k))
            del tmpwords,tmpvecnp,vecl,dic_word2center
        else:
            logger.info("cluster res : cres_%s_kxxx.txt allready exist!" %code)
    logger.info("clustering times : %d/%d" %(cnt,len(kwordsfile_bycode)))

def run04(base_folder,crestype='cres01'):
    #单独获取类中心。在聚类结果cres获取之后，根据不同算法获取类标签
    wvpath=base_folder+'/data_wv/'
    crespath=base_folder+ '/' + crestype +'/cres/'
    cresfiles=uc.getfileinfolder(crespath,'cres')
    tfidf_bycode_folder = './data/data_raw/fn_kwsraw1811bycode/tfidf/bycode/'
    for cresf in cresfiles:
        logger.info("for file : %s" %cresf)
        filename = os.path.split(cresf)[1]
        code = filename[5:filename.rfind('_')]
        code01=code[:code.find('_')] if len(code)>4 else code
        cres = uc.load2list(cresf)
        words = uc.load2list(wvpath+'/%s/words_%s.txt' %(code01,code))
        vecs = np.loadtxt(wvpath+'/%s/vecs_%s.txt' %(code01,code)).tolist()
        tfidfpath=tfidf_bycode_folder+'/%s.json' %code
        if os.path.exists(tfidfpath):
            tfidf = uc.loadjson(tfidfpath)
            get_cluter_center_tfidf(cres,vecs,words,respath='%s/%s/' %(base_folder,crestype),prefix=filename[5:-4],tfidf=tfidf)


def runall(subfix=None):
    version = "20190306"
    print("\n\n***************version=%s***********\n\n" % version)
    # cf=path.path_dataroot + '/cluster/w2vkwraw1811_sgns_code_highq/data_wv_top3'
    cf = run02(subfix)
    # cf = path.path_dataroot + '/cluster/bigram_I/data_wv'
    print("\n\n**************cf=%s*****************\n\n" % cf)
    run03(ktype=1, basefolder=cf)
    run03(ktype=2, basefolder=cf)
    run03(ktype=3, basefolder=cf)


if __name__ == '__main__':
    # run01()
    # runall(subfix='onetf1_bitf6_0')
    # runall(subfix='onetf10_bitf10')
    # run02()
    run04(base_folder='./data/cluster/bigram_I_onetf1_bitf6_1_stoped2/embed_allkwsAbsTitle',crestype='cres01')
    run04(base_folder='./data/cluster/bigram_I_onetf1_bitf6_1_stoped2/embed_allkwsAbsTitle', crestype='cres02')
    run04(base_folder='./data/cluster/bigram_I_onetf1_bitf6_1_stoped2/embed_allkwsAbsTitle', crestype='cres03')