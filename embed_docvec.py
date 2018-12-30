#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/6

import psutil
import time,sys,os,codecs,logging
import util_path as path
import util_common as util
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

BIANMA= 'utf8'
datapath= r'./data'
# bianma='gb18030'
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
testwl=['zgjq200914010','1011143537.nh','DYPJ200924005','2010261634.nh','1014310786.nh','1012347178.nh',
        '1012258129.nh','ddyi201218094','fxsy201508029']

class MyDocuments(object):
    '''
    根据分好词的文件生成句子序列，用于word2vec训练
    dirname:分好词的文件路径，可以是单个文件路径也可以是文件夹地址，文件以txt结尾
    start:从一行的第几个元素开始算词。因为有的文件每行第一个元素是用户id，则start=1用于略过id，
    '''
    def __init__(self, dirname, start=0, subfix='.txt'):
        self.dirname = dirname
        self.start=start
        self.subfix=subfix #默认只读取txt文件

    def __iter__(self):
        cnt = -1
        if os.path.isfile(self.dirname):
            folder,fn=os.path.split(self.dirname)
            fns=[fn]
        else:
            folder=self.dirname
            fns=os.listdir(self.dirname)
        for fname in fns:
            if self.subfix in fname:
                # logger.info("now precessing file %s" %fname)
                with codecs.open(os.path.join(folder, fname), 'rU', 'utf8', errors='ignore') as f:
                    for line in f:
                        l = line.strip().split()
                        if self.start:
                            if len(l) > self.start+1:
                                yield TaggedDocument(l[self.start:],l[:self.start])
                        else:
                            if len(l) > 1:
                                cnt += 1
                                yield TaggedDocument(l, [cnt])

def update_gensim(oldmodelname, indatapath, prefix='',
                  size=200, window=5, minc=2, iter=5, sg=0, hs=0, neg=5, trainwv4dbow=1, newdoc=False):
    #setting filenames
    modelfolder= datapath + r'/model/%s' % oldmodelname
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    modelpath= "%s/%s.model" % (modelfolder, oldmodelname)
    wvecname = "%s/%s_up_%s.wv" % (modelfolder, oldmodelname,prefix)
    dvecname = "%s/%s_up_%s.dv" % (modelfolder, oldmodelname,prefix)
    if not os.path.exists(modelpath):
        logger.info("model %s not exists!!! please train it first you can using 'train_gensim' in embed_docvec.py ")
        return
    #retrain model
    word2vec_start_time = time.time()
    model = Doc2Vec.load(modelpath)
    print("retrain gensim模型:当前时间 : %s" %time.asctime(time.localtime(time.time())))
    model.sg=sg
    model.hs=hs
    model.window=window
    model.dbow_words=trainwv4dbow
    model.netative=neg
    if newdoc:
        model.min_count = minc
        model.build_vocab(MyDocuments(indatapath,start=1))
    model.train(MyDocuments(indatapath,start=1),total_examples=model.corpus_count,epochs=iter)
    print("retrain gensim 完毕 %.2f secs" % (time.time() - word2vec_start_time))
    #saving model
    modelpath = "%s/%s_up_%s.model" % (modelfolder, oldmodelname,prefix)
    model.save(modelpath)  #保存整个模型以及训练过程的数据（其实会生成3个文件model,syn0,syn1 or syn1neg）
    model.wv.save_word2vec_format(wvecname)
    model.docvecs.save_word2vec_format(dvecname)
    print model
    print os.path.split(modelpath)[1]
    # return os.path.splitext(os.path.split(modelpath)[1])[0]
    # KeyedVectors.load_word2vec_format(vecname)
    # print_mostsimi(model, testwl)

def train_gensim(modelname, indatapath,
                 size=200, window=5, minc=2, iter=5, sg=0, hs=0, neg=5, trainwv4dbow=1,annoy=False):
    modelfolder= path.path_dataroot + r'/model/%s' % modelname
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    modelpath,wvecname,dvecname="%s/%s.model" %(modelfolder,modelname),\
                                "%s/%s.normwv" % (modelfolder, modelname),\
                                "%s/%s.normdv" % (modelfolder, modelname)
    if os.path.exists(modelpath):
        logger.info("model %s has already exists!!!")
        return
    doc2vec_start_time = time.time()
    model = Doc2Vec(MyDocuments(indatapath,start=1), size=size, iter=iter, window=window, min_count=minc,
                    dm=1-sg, hs=hs, negative=neg, workers=1, dbow_words=trainwv4dbow)  # workers=multiprocessing.cpu_count()
    print("开始训练gensim模型时间 : %s" % doc2vec_start_time)
    print("gensim训练完毕 %.2f secs" % (time.time() - doc2vec_start_time))
    model.save(modelpath)  #保存整个模型以及训练过程的数据(其实会生成3个文件model,syn0,syn1 or syn1neg)
    model.init_sims(replace=True) #句向量归一化并替代原向量
    model.docvecs.save_word2vec_format(dvecname,prefix='') #单纯保存文档向量,文本文件

    model.wv.init_sims(replace=True) #词向量归一化并替代原向量
    model.wv.save_word2vec_format(wvecname)  # 单纯保存词向量,文本文件
    print model
    # KeyedVectors.load_word2vec_format(vecname)
    # print_mostsimi(model, testwl)
    # return modelname

def print_mostsimi(model, wordlist, annoyindex=None,top=10):
    '''
    获取模型的mostsimilar结果
    :param model: 
    :type model:Word2Vec 
    :param wordlist: 
    :type wordlist: list
    :param annoyindex: 
    :type annoyindex: AnnoyIndexer
    :return: 
    :rtype: 
    '''
    t = time.time()
    outpath='./modeltest.txt'
    print('writing most similar res in file : %s' %outpath)
    f=open(outpath,'w')
    f.write('%s\n'%model)
    for w in wordlist:
        f.write("------------for word : %s\n" %w)
        try:
            result = model.docvecs.most_similar(w,topn=top,indexer=annoyindex)
            for e in result:
                f.write("%s : %.3f\n" % (e[0], e[1]))
            f.write("**********************************************\n")
            result = model.most_similar([model.docvecs[w]], topn=top, indexer=annoyindex)
            for e in result:
                f.write("%s : %.3f\n" % (e[0].encode('utf8'), e[1]))
        except KeyError, e:
            f.write("word %s is not in the model!\n" %w)
    print("--------------time cost %.3f secs/word " %((time.time()-t)/float(len(wordlist))))
    f.write("\n")
    f.close()

def test_model(d2vmodel, testwl=testwl, topn=10, evalut=False):
    model = Doc2Vec.load(d2vmodel) if type(d2vmodel) is str else d2vmodel
    print_mostsimi(model, testwl, top=topn)
    if evalut:
        ina = raw_input("model score 1-9 : ")
        f = open(datapath + r'/model/gensim/modelscore.txt', 'a')
        f.write("size=%d window=%d mincount=%d iter=%d sg=%d hs=%d ns=%d score=%s\n"
                %(model.vector_size, model.window, model.min_count, model.iter, model.sg, model.hs, model.negative, ina))
        f.close()

def gen_model_name(argslist, mnameprefix='model'):
    (size, win, minc, iter, sg, hs, neg, traindbowwv) = argslist
    modetype = 'dbow' if sg else 'dm'
    opttypehs = 'hs' if hs else ''
    opttypens = 'ns' if neg else ''
    mode="%s%s%s" %(modetype, opttypehs, opttypens)
    modelname = 'd2v_%s_d%dw%dminc%diter%d_%s' % (mnameprefix, size, win, minc, iter, mode)
    return modelname,mode

def run_single_train(argslist,inputdocs,oldmodelname='',mnameprefix='model', justgetname=False):
    (size, win, minc, iter, sg, hs, neg, traindbowwv) = argslist
    modelname,mode=gen_model_name(argslist,mnameprefix)
    if not justgetname:
        if oldmodelname:
            modelname='%s_up_%s'%(oldmodelname,mode)
            logger.info("Starting retrain model : %s" % oldmodelname)
            update_gensim(oldmodelname, indatapath=inputdocs, prefix=mode,size=size, window=win,
                          minc=minc, iter=iter, sg=sg, hs=hs, neg=neg, trainwv4dbow=traindbowwv)
        else:
            logger.info("Starting train model : %s" %modelname)
            train_gensim(modelname, indatapath=inputdocs, size=size, window=win,
                     minc=minc, iter=iter, sg=sg, hs=hs, neg=neg, trainwv4dbow=traindbowwv)
    return modelname

def run_train(mnameprefix='model',oldmodelname='',justgetname=False):
    #        dim,win,min,itr,sg,hs,neg,traindbowwv
    argls = [[300, 5, 3, 30, 0, 0, 10, 1]]
    # inputdocs = datapath + r'/data_seg/sumery_highq5w/old'
    inputdocs = path.path_datahighq5w + r'/log18_highq_5w_posi.txt'
    mnames=[]
    for argl in argls:
        modelname = run_single_train(argl,inputdocs,oldmodelname=oldmodelname,mnameprefix=mnameprefix ,justgetname=justgetname)
        mnames.append(modelname)
    return  mnames

if __name__ == '__main__':
    oldmodelname = ''
    names=run_train(mnameprefix='udownhighq5wposi',oldmodelname=oldmodelname,justgetname=True)
    for mname in names:
        if oldmodelname:
            modelpath = "%s/%s/%s.model" % (path.path_model, oldmodelname, mname)
        else:
            modelpath = "%s/%s/%s.model" % (path.path_model, mname, mname)
        test_model(modelpath)