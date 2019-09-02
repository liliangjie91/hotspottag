#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/11/5 

import os
import psutil
import codecs
import logging
import sys
import time
import util_path as path
import util_common as uc
from gensim.models import Word2Vec,KeyedVectors,Doc2Vec
from gensim.similarities.index import AnnoyIndexer

ENCODE= 'utf8'
DATAPATH= r'./data'

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
testwl=[u'中国',u'一带一路',u'广东省',u'计算机',u'经济',u'微信']


'''
词向量训练，annoy索引构建
'''

class MySentences(object):
    '''
    根据分好词的文件生成句子序列，用于word2vec训练
    dirname:分好词的文件路径，可以是单个文件路径也可以是文件夹地址，文件以txt结尾
    start:从一行的第几个元素开始算词。因为有的文件每行第一个元素是用户id，则start=1用于略过id，
    '''
    def __init__(self, dirname, start=0, subfix='.txt',bigramword=False,trainfilename=False):
        '''
        初始化操作
        :param dirname:词向量训练原始数据文件夹，内部文件是分词后的数据 
        :type dirname: str
        :param start: 从每一行的第几个词开始训练？
        :type start: int
        :param subfix: 匹配dirname中的文件
        :type subfix: str
        :param bigramword: 是否生成并训练二元词
        :type bigramword: bool
        :param trainfilename: 是否把文件名也加入训练
        :type trainfilename: bool
        '''
        self.dirname = dirname
        self.start=start
        self.subfix=subfix
        self.bigram=bigramword
        self.trainfname=trainfilename

    def __iter__(self):
        if os.path.isfile(self.dirname):
            # folder, fn = os.path.split(self.dirname)
            fns = [self.dirname]
        else:
            # folder = self.dirname
            # fns = os.listdir(self.dirname)
            fns = uc.getfileinfolder(self.dirname,prefix=self.subfix,recurse=2)
        for i,fname in enumerate(fns):
            # if self.subfix in fname:
            logger.info('-------------%s(%d/%d)' %(fname,i,len(fns)))
            filename=os.path.splitext(os.path.split(fname)[1])[0] if self.trainfname else None
            with codecs.open(fname, 'rU', 'utf8', errors='ignore') as f:
                for line in f:
                    l = line.strip().split()
                    if self.start:
                        if len(l) > self.start + 1:  # words must more than 1
                            yield uc.extend2bigram(l[self.start:],filename) if self.bigram else l[self.start:]
                    else:
                        if len(l) > 1:
                            yield uc.extend2bigram(l,filename) if self.bigram else l


def train_gensim(modelname,
                 indatapath,startfrom=0, train_bigram=False, train_filename=False,
                 size=200, window=5,minc=2, iterr=5, sg=0, hs=0, neg=5,
                 annoy=False,oldmodelpath=None):
    '''
    第一行参数：模型名
    第二行参数：MySentences参数
    第三行参数：词向量模型参数
    第四行参数：是否计算annoy索引 
    '''
    #初始化模型参数
    ms = MySentences(indatapath, start=startfrom, bigramword=train_bigram, trainfilename=train_filename)
    if oldmodelpath:
        model=Word2Vec.load(oldmodelpath)
        # assert model.vector_size == size
    else:
        model = Word2Vec(size=size, iter=iterr, window=window, min_count=minc,
                     sg=sg, hs=hs, negative=neg, workers=25,max_final_vocab=7000000,max_vocab_size=50000000)  # workers=multiprocessing.cpu_count()
    #设置结果路径
    modelfolder= path.path_dataroot + r'/model/%s' % modelname
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    modelpath="%s/%s.model" %(modelfolder,modelname)
    # vecname = "%s/%s.normwv" % (modelfolder, modelname)
    annoypath="%s/%s.annoy" %(modelfolder,modelname)
    if os.path.exists(modelpath):
        logger.info("model %s has already exists!!!")
        return
    #训练
    logger.info("!!!!!!!!!!!!!!! max_final_vocab=7000000,max_vocab_size=10000000 !!!!!!!!!!!!!!")
    word2vec_start_time = time.time()
    if oldmodelpath:
        model.build_vocab(ms, progress_per=500000, dry_run=False, update=True)  # 构建词库，每处理500000行报告一次
    else:
        model.build_vocab(ms,progress_per=3000000,dry_run=False) #构建词库，每处理3000000行报告一次about 1 min
    # wordscnt=model.vocabulary.raw_vocab
    # cnts=sorted(wordscnt.keys())

    model.train(ms, total_examples=model.corpus_count, epochs=iterr) #词向量训练
    logger.info("trainning gensim cost : %s" %(time.time() - word2vec_start_time))
    model.save(modelpath)  #保存整个模型以及训练过程的数据（其实会生成3个文件model,syn0,syn1 or syn1neg）
    # model.init_sims(replace=True) #词向量标准化，replace=True 用标准化的词向量替代原始词向量
    # model.wv.save_word2vec_format(vecname, binary=False) #单纯保存词向量
    # KeyedVectors.load_word2vec_format(vecname) #单纯载入词向量
    # print_mostsimi(model, testwl)
    if annoy:
        save_annoy(annoypath, model)
        # print_mostsimi(model, testwl, aindex)

def save_annoy(annoypath, model):
    if not os.path.exists(annoypath):
        print("开始构建annoy索引:当前时间 : " + time.asctime(time.localtime(time.time())))
        starttime12 = time.time()
        aindex = AnnoyIndexer(model, 200)
        print("构建索引完毕 %.2f secs" % (time.time() - starttime12))
        # 保存annoy索引
        print("开始保存annoy索引")
        starttime13 = time.time()
        aindex.save(annoypath)
        print("保存索引完毕 %.2f secs" % (time.time() - starttime13))

def load_annoy(annoypath, model):
    '''
    加载annoy索引，如果没有，则创建
    :param annoypath: 
    :type annoypath: 
    :param model: 
    :type model: Word2Vec
    :return: 
    :rtype: AnnoyIndexer
    '''
    aindex = AnnoyIndexer()
    aindex.load(annoypath)
    return aindex

def test_model_byhuman(w2vmodel, testwl=testwl, topn=10, evalute=False):
    #根据最相似结果人工评价模型质量
    model = Word2Vec.load(w2vmodel) if isinstance(w2vmodel,str) else w2vmodel
    print_mostsimi(model, testwl, top=topn)
    if evalute:
        ina = input("model score 1-9 : ")
        f = open(DATAPATH + r'/model/gensim/modelscore.txt', 'a')
        f.write("size=%d window=%d mincount=%d iter=%d sg=%d hs=%d ns=%d score=%s\n"
                %(model.vector_size, model.window, model.min_count, model.iter, model.sg, model.hs, model.negative, ina))
        f.close()

def print_relationsimi(model,l,top=10):
    '''
    l = [a,b,c,d]
    a-b=c-d ==>  a+d-b=c
    man - woman = king - queen  ==>  queen + man - women = king
    so in 金庸 shuld be 杨过 - 小龙女 = 郭靖 - 黄蓉 ==> 黄蓉(d) + 杨过(a) - 小龙女(b) = 郭靖
    :param model: 
    :param l:
    :type l:list
    :return: 
    '''
    if len(l) != 4:
        print("input l length is not 4!")
        return
    a,b,c,d = l[0],l[1],l[2],l[3]
    print("------------for word : %s + ( %s - %s ) . the answer should be like %s" %(d,a,b,c))
    try:
        result = model.most_similar([a,d],[b],topn=top)
        for e in result:
            print("%s : %.3f" % (e[0], e[1]))
    except KeyError:
        print("some word is not in the model!")

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
    for w in wordlist:
        print("------------for word : %s" %w)
        try:
            result = model.most_similar(w,topn=top,indexer=annoyindex)
            for e in result:
                print("%s : %.3f" %(e[0],e[1]))
        except KeyError:
            print("word %s is not in the model!" %w)
    print("--------------time cost %.3f secs/word " %((time.time()-t)/float(len(wordlist))))

def run_single_train(W2V_args, MySectence_args, ifannoy=False, mnameprefix='test', justgetname=False):
    #运行单词模型训练
    #根据参数，生成文件名等，之后送入训练
    (size, win, minc, iter, sg, hs, neg) = W2V_args
    (inputpath, startfrom, train_bigram, train_filename) = MySectence_args
    modeltype = 'sg' if sg else 'cbow'
    opttypehs = 'hs' if hs else ''
    opttypens = 'ns' if neg else ''
    modelname = 'w2v_%s_d%dw%dminc%diter%d_%s%s%s'% (mnameprefix, size, win, minc, iter, modeltype, opttypehs, opttypens)
    print("inputpath=%s \nmodelname=%s" %(inputpath,modelname))

    if not justgetname:
        logger.info("Starting train model : %s" %modelname)
        train_gensim(modelname,
                     indatapath=inputpath,startfrom=startfrom,train_bigram=train_bigram,train_filename=train_filename,
                     size=size, window=win,minc=minc, iterr=iter, sg=sg, hs=hs, neg=neg,
                     annoy=ifannoy)
    return modelname

def run_train(indatapath,mnameprefix='test',ifannoy=False):
    # 词向量训练入口，输入训练数据，内部设置训练参数
    # 此处有很多参数，需要注意：1，输入数据 2，模型超参 3，
    #        dim,win,min,itr,sg,hs,neg
    # argls = [[300, 5, 2, 5, 0, 0, 5],
    #          [300, 8, 2, 5, 1, 0, 5]]
    default_args = [[200, 5, 5, 5, 0, 0, 5]]
    W2V_argls = [[200, 5, 5, 20, 1, 0, 5]] #用于外网词推荐接口：[200, 8, 9, 10, 1, 0, 5],[200, 5, 8, 3, 0, 0, 5]
    #             (input, Startfrom, bigram, train_filename)
    MySectence_args=(indatapath, 1 , False , False)
    for wvargl in W2V_argls:
        modelname = run_single_train(wvargl,MySectence_args,mnameprefix=mnameprefix, ifannoy=ifannoy)


if __name__ == '__main__':
    # indatapath=path.path_dataseg+'/kws_raw'
    # print(indatapath)
    # run_train(indatapath,mnameprefix="kws1811_raw")
    # indatapath=path.path_dataraw+'/fn_kws1811bycode/I/'
    # indatapath = './data/data_seg/alltype_KwsAbsTitle/'
    indatapath = '../userInterestDynamic/data/data_seg/'
    # !!!!!!重要!!!!!! 训练之前，一定要检查默认参数，注意train_gensim 函数中的参数设置！！！！
    # indatapath='./data/data_raw/fn_kwsraw1811bycode/tmp/'
    # a=uc.getfileinfolder(indatapath,recurse=2)
    # for i in a:
        # print(i)
    run_train(indatapath, mnameprefix="fulltextskws19m4m5",ifannoy=True)
    # !!!!!!重要!!!!!! 训练之前，一定要检查默认参数，注意train_gensim 函数中的参数设置！！！！
    # !!!!!!重要!!!!!! 训练之前，一定要检查默认参数，注意train_gensim 函数中的参数设置！！！！