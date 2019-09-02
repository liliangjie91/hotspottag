#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/11/23
import os,json,time,sys,math
import util_path as path
import util_common as uc
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

#主要用于数据的预处理。
#1，针对用户的浏览(b),下载(d)日志原始数据结合从数据库提取的某些字段，做分组，归纳，去重，统计，提取字段等预处理
#2，计算关键词数据的tfidf
#3，根据tf，idf等信息计算停用词表。为后续词聚类做词过滤工作。

datapath=path.path_dataroot
goodpath=path.path_datagood
highqpath=path.path_datahighq
alllog_b09=datapath+r'/log_b_09.txt'
alllog_b18=datapath+r'/log_b_18.txt'
alllog_d09=datapath+r'/log_d_09.json'
alllog_d18=datapath+r'/log_d_18.json'
user_typeinter_18 = datapath+r'/userdb_intersec_18.txt'
user_typeinter_09 = datapath+r'/userdb_intersec_09.txt'
ulog_typeinter09_d = datapath + r'/ulog_typeinter09_d.json'
ulog_typeinter09_b = datapath + r'/ulog_typeinter09_b.json'
ulog_typeinter09_dbdiff = datapath + r'/ulog_typeinter09_dbdiff.json'
ulog_typeinter18_d = datapath + r'/ulog_typeinter18_d.json'
ulog_typeinter18_b = datapath + r'/ulog_typeinter18_b.json'
ulog_typeinter18_dbdiff = datapath + r'/ulog_typeinter18_dbdiff.json'

user_timeinter_b = datapath+r'/userb_intersec_0918.txt'
user_timeinter_d = datapath+r'/userd_intersec_0918.txt'

ulog_sample_18_highq_posi=highqpath+'/log18_highq_posi.txt'
ulog_sample_18_highq_neg=highqpath+'/log18_highq_neg.txt'



def get_sub_dic(dicin, keys):
    '''
    获取子字典，根据keys中的key值
    :param dicin: 
    :type dicin: dict
    :param keys: 
    :type keys: list
    :return: 
    :rtype: dict
    '''
    logger.info("getting sub dicts from input keys...")
    res = {}
    for u in keys:
        res[u] = dicin[u]
    return res

def get_dic_diff(logb, logd):
    #计算差集。用户浏览了但并没有下载 logb-logd。同时，这种情况(负样本)应对的是用户浏览同时也下载(正样本)了。
    logger.info("getting two dicts` difference...")
    logdb={}
    for i in logb.keys():
        diff = list(set(logb[i]).difference(set(logd[i])))
        if diff:
            logdb[i]=diff
    logger.info("length of diff file %d " %len(logdb))
    #util.savejson(datapath+'/ufn_typeinter_09_bddiff.json',logdb)
    return logdb

def get_intersec_log(user_interseclist, alllog_b, alllog_d,prefix,rootpath=datapath):
    '''
    获取用户d,b日志的交集用户，并获取这群用户的d，b以及b-d日志分别储存
    :param user_interseclist: 
    :type user_interseclist: 
    :param alllog_b: 
    :type alllog_b: 
    :param alllog_d: 
    :type alllog_d: 
    :param prefix: 
    :type prefix: 
    :return: 
    :rtype: 
    '''
    blog=uc.load2dic(alllog_b)
    # dlog=util.loadjson(alllog_d)
    dlog = uc.load2dic(alllog_d)
    userb=blog.keys()
    userd=dlog.keys()
    if not os.path.exists(user_interseclist):
        logger.info("caculating two logs` intersection user...")
        uintersec = list(set(userb).intersection(set(userd)))
        uc.list2txt(uintersec, user_interseclist)
    else:
        logger.info("loading two logs` intersection user file : %s" %user_interseclist)
        uintersec = uc.load2list(user_interseclist)
    interseced_d = get_sub_dic(dlog, uintersec)
    interseced_b = get_sub_dic(blog, uintersec)
    del dlog
    del blog
    # interseced_dbdiff = get_dic_diff(interseced_b, interseced_d)
    logger.info("saving ress...")
    uc.savejson("%s/%s_posi.json" % (rootpath, prefix), interseced_d)
    uc.savejson("%s/%s_neg.json" % (rootpath, prefix), interseced_b)
    # util.savejson("%s/%s_dbdiff.json" %(rootpath,prefix), interseced_dbdiff)
    logger.info("done!")

def del_once_action():
    #删除只有一次操作的用户？
    pass

def get_highquality_ulog(inpath,outpath,actmin=2,actmax=300):
    #优质用户历史，操作数>2 <300(操作太多可能是爬虫)
    oldulog = uc.load2list(inpath)
    newulog = []
    for l in oldulog:
        ws=l.strip().split()[1:] #每一行第一个是id
        if actmax>len(ws)>actmin:
            newulog.append(l)
    uc.list2txt(newulog, outpath)

def get_userlist(path,logpath=None):
    #获取用户id列表，返回list
    if os.path.exists(path):
        return uc.load2list(path)
    else:
        ul = uc.load2list(logpath, get1column=0)
        uc.list2txt(ul, path)
        return ul

def get_fnlist(path,logpath):
    #获取文件名列表，返回list
    if os.path.exists(path):
        return uc.load2list(path)
    else:
        ul = uc.load2list(logpath, to1column=True, start=1)
        res=list(set(ul))
        uc.list2txt(res, path)
        return res

def fns_preprocess(inpath, outpath):
    #对文件名做去重以及小写操作
    inli=get_fnlist(inpath,'')
    res=[]
    for fn in inli:
        res.append(fn.lower())
    a= list(set(res))
    uc.list2txt(a, outpath)
    print(len(a))
    return a

def fns_merge(path1, path2, respath):
    #合并两组fn
    la=uc.load2list(path1)
    lb=uc.load2list(path2)
    res=list(set(la).union(set(lb)))
    uc.list2txt(res, respath)

def gen_samples(ulog_d, ulog_diff, prefix, outpath):
    #根据logd和get_dic_diff函数生成的差集，生成正负样本对uid+fn+0 or 1
    logger.info("generate posi & neg samples for myrec...")
    if '.json' in ulog_d:
        dlog=uc.loadjson(ulog_d)
        difflog = uc.loadjson(ulog_diff)
    else:
        dlog=uc.load2dic(ulog_d)
        difflog = uc.load2dic(ulog_diff)
    posisam=[]
    negsam=[]
    logger.info("gen posi samples...")
    for k in dlog.keys():
        fns=dlog[k]
        if fns:
            for fn in fns:
                posisam.append("%s+%s\t%d"%(k,fn.lower(),1))
    print(len(posisam))
    uc.list2txt(posisam, outpath + '/' + prefix + '_posi.txt')
    del dlog
    del posisam
    logger.info("gen neg samples...")
    for k in difflog.keys():
        fns=difflog[k]
        if fns:
            for fn in fns:
                negsam.append("%s+%s\t%d"%(k,fn.lower(),0))
    print(len(negsam))
    uc.list2txt(negsam, outpath + '/' + prefix + '_neg.txt')

def get_fnkws_by_code(fn_code_file,fn_kws_file,respath):
    '''
    根据文件名-子栏目代码文件和文件名-关键词文件，获取 子栏目代码-关键词文件
    :param fn_code_file: 文件名-子栏目代码文件
    :type fn_code_file: 
    :param fn_kws_file: 文件名-关键词文件
    :type fn_kws_file: 
    :param respath: 结果文件夹
    :type respath: 
    :return: 结果文件夹内文件名为aimedcode内的各个元素，二级文件夹内的文件是具体到子栏目代码的关键词文件，文件名是具体的子栏目代码
    :rtype: 
    '''
    dic_codefn = uc.load2dic_02(fn_code_file)
    dic_fnkws = uc.load2dic(fn_kws_file,interactor=' ')
    aimedcode=["D","I"]
    codes = dic_codefn.keys()
    for c in codes:
        if c[:1] in aimedcode:
            logger.info("now precessing code : %s" % c)
            tmpres=[]
            tmpfns=dic_codefn[c]
            if len(tmpfns)>10:
                for fn in tmpfns:
                    if fn in dic_fnkws:
                        kws=dic_fnkws[fn]
                        tmpres.append("%s\t%s" %(fn,kws))
                if tmpres:
                    folder="%s/%s" %(respath,c[:1])
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    uc.list2txt(tmpres,"%s/%s.txt" %(folder,c))

def filter_data_by_fnsset(inputdata, fns, respath):
    #获取inputdata中含有集合fns元素的条目.相当于过滤
    indata=uc.load2list(inputdata)
    fnsset = set(uc.load2list(fns))
    res=[]
    cnt=-1
    for i in indata:
        cnt+=1
        if cnt%100000==0: print(cnt)
        fn=i.strip().split()[0]
        fn=fn.strip().upper()
        if fn in fnsset:
            res.append(i)
    uc.list2txt(res,respath+'/'+os.path.split(inputdata)[1])

def get_bigram_words(infolder, resfile, justbigram=False, centerword=False, justonegram=False):
    '''
    获取二元词并存储
    :param infolder:一元词所在文件夹 
    :type infolder: 
    :param resfile: 
    :type resfile: 
    :return: 
    :rtype: 
    '''

    files = uc.getfileinfolder(infolder) if os.path.isdir(infolder) else [infolder]
    raw_curpus=[]
    bigram_curpus = []
    linecnt, wordcnt = -1, 0
    for f in files:
        logger.info("loading file : %s" %f)
        filename=os.path.splitext(os.path.split(f)[1])[0] if centerword else None
        tmpwlist=uc.load2list(f,row2list=True) #加载文件，返回二维list(每一行一个子list)
        raw_curpus.extend(tmpwlist)
        for l in tmpwlist:
            wordcnt += len(l)
            linecnt += 1
            if linecnt % 100000 == 0:
                logger.info("cat bigram words %d" % wordcnt)
            if justonegram:
                bigram_curpus.append(' '.join(l))
            else:
                bigram_curpus.append(' '.join(uc.extend2bigram(l, start=1,
                                                               justbigram=justbigram, centerword=filename,
                                                               addstartelem=True)))  # start=1，因为第一个元素是fn,不用
    logger.info("writing res to file : %s" %resfile)
    uc.list2txt(bigram_curpus,resfile)

def caculate_tf(infile,resfile=None,start=1):
    '''
    计算词频,此处为infile的总词频
    '''
    corpus = uc.load2list(infile,row2list=True,start=start) if isinstance(infile,str) else infile
    docscnt,wordscnt=len(corpus),0
    invertedindex_dic = {}
    for i, l in enumerate(corpus):
        wordscnt+=len(l)
        if i % 100000 == 0:
            logger.info("caculate wordfreq in line: %d/%d" % (i, docscnt))
        for w in l:
            invertedindex_dic[w] = invertedindex_dic[w] + 1 if w in invertedindex_dic else 1
    for k,v in invertedindex_dic.items():
        invertedindex_dic[k]=v/wordscnt
    if resfile:
        uc.savejson(resfile + '_tf_%07d.json' %wordscnt, invertedindex_dic)
    return invertedindex_dic

def caculate_idf(infile,resfile=None,start=0):
    '''
    计算idf
    '''
    corpus = uc.load2list(infile, row2list=True, start=start) if isinstance(infile, str) else infile
    docscnt, wordscnt = len(corpus), 0
    invertedindex_dic={}
    for i,l in enumerate(corpus):
        if i%100000==0:
            logger.info("caculate idf_freq in line: %d/%d" %(i,docscnt))
        for w in set(l):
            invertedindex_dic[w]=invertedindex_dic[w]+1 if w in invertedindex_dic else 1
    logger.info("starting caculate idf for each word...")
    for k,v in invertedindex_dic.items():
        invertedindex_dic[k]=math.log(docscnt/v)
    if resfile:
        uc.savejson(resfile + '_idf.json', invertedindex_dic)
    return invertedindex_dic

def caculate_tfidf(tf,idf,resfile=None):
    '''
    根据tf和idf文件计算tfidf
    '''
    res={}
    meanidf=sum(idf.values())/len(idf)
    for w in tf.keys():
        res[w]=tf[w]*idf[w] if w in idf else tf[w]*meanidf
    if resfile:
        uc.savejson(resfile + '_tfidf.json', res)
    return res

def caculate_tfidf_code(code_kws_dict, idftable, respath):
    #为每个code下的词计算tfidf
    from collections import Counter
    if not os.path.exists(respath): os.makedirs(respath)
    idf=uc.loadjson(idftable) if isinstance(idftable,str) else idftable
    meanidf = np.mean(list(idf.values()))
    codekwsd = uc.loadjson(code_kws_dict) if isinstance(code_kws_dict,str) else code_kws_dict
    for code,kws in codekwsd.items():
        logger.info('for code %s' %code)
        allwtmp = len(kws)
        freqtmp=dict(Counter(kws))
        tfidftmp={}
        for w,freq in freqtmp.items():
            tfidftmp[w]=freq*idf[w]/allwtmp if w in idf else freq*meanidf/allwtmp
        if tfidftmp:
            uc.savejson(respath+'/%s.json' %code,tfidftmp)
        del tfidftmp

def get_threshold_tfidf(tfidfs,thre):
    '''
    给一组tfidf，获取最小比例的阈值，例如要卡掉10%的tfidf，应如何确定阈值 
    '''
    from collections import Counter
    c1 = Counter(tfidfs)
    alllen = len(tfidfs)
    c1sorted = sorted(c1.items(), key=lambda x: x[0])
    sum0 = 0
    for i, v in enumerate(c1sorted):
        sum0 += v[1]
        if sum0 > int(alllen*thre):
            return c1sorted[i - 1][0] if i > 0 else c1sorted[0][0]

def genTFIDF_stopwords_step2(tfidf_dict,thre,respath=None):
    #生成二级停用词表，在数据经初级停用词过滤后，再根据词tfidf表，确定一定比例的停用词，然后获取tfidf阈值，构建停用词表
    aimthre=get_threshold_tfidf(tfidf_dict.values(),thre=thre)
    res=[]
    for k,v in tfidf_dict.items():
        if v<=aimthre:
            res.append(k)
    if respath:
        uc.list2txt(res,respath)
    return set(res)

def genTFIDF_stopword_step1(tffile, appendstopword, tfthreh=1, resfile=None):
    #生成初级停用词表，主要通过简单的卡词频和手工选出的停用词(appendstop)
    stopws=[]
    tfjson=uc.loadjson(tffile)
    for k,v in tfjson.items():
        if v<=tfthreh:
            stopws.append(k)
    stopws.extend(appendstopword)
    stopws=set(stopws)
    logger.info("totally get %d stopwords" %len(stopws))
    if resfile:
        uc.list2txt(list(stopws),resfile)
    return stopws

def map_running(path,cpus=15):
    """
    多进程并将对应结果集写入共享资源，维持执行的进程总数，当一个进程执行完毕后会添加新的进程进去(非阻塞)
    """
    from multiprocessing import Pool

    # rec_path = IOTools.open_folder(path)
    rec_path = uc.getfileinfolder(path,prefix='')
    pool = Pool(cpus)
    pool.map(wrap, rec_path)
    pool.close()
    pool.join()

def wrap(path):
    filter_data_by_fnsset(path,
                       './data/data_raw/fn_kwsraw1811bycode/fns_I.txt',
                       './data/data_raw/fn_kwsraw1811bycode/tmp')

if __name__ == '__main__':
    # get_intersec_log(goodpath+'/highq/uintersec18.txt',
    #                  ulog_sample_18_highq_neg,ulog_sample_18_highq_posi,
    #                  "ulog18_highq_interseced",rootpath=goodpath+'/highq')
    # util.json2txt(goodpath+'/highq/ulog18_highq_interseced_neg.json',goodpath+'/highq/ulog18_highq_interseced_neg.txt')
    # util.json2txt(goodpath + '/highq/ulog18_highq_interseced_posi.json',
    #               goodpath + '/highq/ulog18_highq_interseced_posi.txt')

    # gen_samples(ulog_typeinter18_d,ulog_typeinter18_dbdiff,"samples_18")
    # gen_samples(ulog_typeinter09_d,ulog_typeinter09_dbdiff,"samples_09")
    # get_highquality_ulog(goodpath+'/log18_neg.txt', goodpath+'/highq/log18_highq_neg.txt')
    # a=get_fnlist('./data/data_seg/logall/fn18all_d.txt','./data/data_seg/logall/log_d_18.txt')
    # b=get_fnlist('./data/data_seg/logall/fn18all_b.txt','./data/data_seg/logall/log_b_18.txt')
    # res = list(set(a).union(set(b)))
    # util.list2txt(res, './data/data_seg/logall/fn18all_all.txt')
    # mergefns('./data/data_seg/logall/fn18all_d.txt',
    #          './data/data_seg/logall/fn18all_b.txt',
    #          './data/data_seg/logall/fn18all_all.txt')
    # a=filter_fns('./data/highq_5w/fn18_5w_all.txt','./data/highq_5w/fn18_5w_all_unic.txt')
    # gen_samples(Path.path_datahighq5w+'/log18_highq_5w_posi.txt',
    #             Path.path_datahighq5w+'/log18_highq_5w_neg.txt',
    #             'sample_highq5w',
    #             Path.path_datahighq5w)
    # get_fnkws_by_code(datapath+'/data_raw/fn_code_1811_seg', datapath+'/data_raw/fn_kws_1811',
    #                   datapath+'/data_raw/fn_kws1811bycode')
    # get_fnkws_by_code(datapath + '/data_raw/fn_code_1811_seg', datapath + '/data_raw/fn_kwsraw_1811',
    #                   datapath + '/data_raw/fn_kwsraw1811bycode')

    fnkwsfolder=path.path_dataraw + '/fn_kwsraw1811bycode/I/'
    fnkwsbigram=path.path_dataraw + '/fn_kwsraw1811bycode/bigram_I_oneGram.txt'
    # fnkwsbigram = path.path_dataraw + '/fn_kwsraw1811bycode/test.txt'
    fnkwsbg_tfidfbi=path.path_dataraw + '/fn_kwsraw1811bycode/tfidf_I_oneGramByCode'
    stopwords_resfile=path.path_dataraw + '/fn_kwsraw1811bycode/stopword/stopws_oneGramByCode'
    # fnkwsbg_tfidf=path.path_dataraw + '/fn_kwsraw1811bycode/bigram_I_kwsraw'
    idf_allwords= path.path_dataraw + '/fn_kwsraw1811bycode/tfidf/tfidf_I_allByCode_idf.json'
    appendStopw_humanFind= './otherfile/tfidf_stopwords_I.txt'
    appendStopw= path.path_dataraw + '/fn_kwsraw1811bycode/bigram_I_kwsraw_justbigram_stopwords_tf3.txt'
    stopwords_step1 = path.path_dataraw + '/fn_kwsraw1811bycode/stopword/stopws_allByCode_o1b6_tf_step1.txt'
    code_kws_stoped1 = path.path_dataraw + '/fn_kwsraw1811bycode/codeKwsDict/bigram_I_codekws_stoped1.json'
    get_bigram_words('./data/data_seg/alltype_KwsAbsTitle/kws_jb_1811.txt',
                     resfile='./data/data_seg/alltype_KwsAbsTitle/kws_jb_1811_bigram.txt',justbigram=True)
    get_bigram_words('./data/data_seg/alltype_KwsAbsTitle/kws_raw_1811.txt',
                     resfile='./data/data_seg/alltype_KwsAbsTitle/kws_raw_1811_bigram.txt',justbigram=True)

    # idf=caculate_idf(fnkwsbigram, fnkwsbg_tfidf)
    # tf=caculate_tf(fnkwsbigram, fnkwsbg_tfidf)
    # caculate_tfidf(tf,idf,fnkwsbg_tfidf)

    # genTFIDF_Stopword(fnkwsbg_tfidf+'_tf.json',uc.load2list(appendstopw),resfile=fnkwsbg_tfidf+'_stopwords_tf5.txt',tfthreh=5)
    # genTFIDF_Stopword(fnkwsbg_tfidfbi + '_tf.json', uc.load2list(appendStopw_humanFind),
    #                   resfile=stopwords_resfile + '_tf1.txt', tfthreh=1)

    # genTFIDF_Stopword(fnkwsbg_tfidf + '_tf.json', uc.load2list(appendstopw1),
    #                   resfile=fnkwsbg_tfidf + '_stopwords_onetf10_bitf10.txt', tfthreh=10)

    # jcodekws = uc.loadjson(path.path_dataraw +
    #                              '/fn_kwsraw1811bycode/codeKwsDict/bigram_I_codekws_justbigram.json')
    # kws=list(jcodekws.values())
    # for code,kws in jcodekws.items():
        # caculate_tf([kws])
    # idf = caculate_idf(kws,fnkwsbg_tfidf)
    # tf = caculate_tf(kws,fnkwsbg_tfidf)
    # caculate_tfidf(tf, idf, fnkwsbg_tfidf)
    # caculate_tfidf_code(code_kws_stoped1,idf_allwords,
    #                     respath=path.path_dataraw+'/fn_kwsraw1811bycode/tfidf/bycode')
    # map_running('./data/data_seg/title/')
    pass
