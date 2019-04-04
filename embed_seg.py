#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by liliangjie on 2018/11/10 
# Email llj : laiangnaduo91@gmail.com
import util_segment
import os
import codecs
import logging
from tc_conversion.langconv import *
from tc_conversion.full_half_conversion import *
import time
import util_common as util

ENCODE= 'utf8' #'gb18030'
BASEPATH= r'./data'

SENTSEG = util_segment.SentenceSegmentation()
WORDSEG = util_segment.WordSegmentation()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


'''
本文件主要用于分词-为接下来的词嵌入训练做准备
为应对不同输入形式，主要分为：
1，对单行做分词 segword4*
2，对文件做分词 seg4file_*
常用seg4file_1line1text 针对大型文本，一行即为一篇文章的输入

!!!! 注意 !!!!
1,编码格式 默认utf-8
2,半全角转换和繁简转换会很大减慢速度 注意设置segword4oneline的参数convert
'''

def segword4oneline(line, minwc=3, minwlen=0, sseg=False, convert=False):
    '''
    对一行输入分词，分词结果是一行list
    :param line: 一行，一句
    :type line: str
    :param minwc: 分词结果最小词数，如果分词后结果小于minwc，则输出空
    :type minwc: int
    :param minwlen: 分词后单个词的最小长度如，'我' 长度=1 '我们' = 2 小于改长度则不计入分词结果
    :type minwlen: int
    :param convert: 是否做全角转化，简繁转换
    :type convert: bool
    :return: 
    :rtype: 
    '''
    line = line.strip().lower()
    if convert:
        line = str_full2half(line)  # 全角转半角
        line = Converter('zh-hans').convert(line)  # 繁体转简体
    wordseg2 = WORDSEG.segment(line)
    wordseg = [i for i in wordseg2 if len(i) > minwlen]
    if len(wordseg) > minwc:
        return wordseg
    else:
        return []

def segword4onetext(fulltext, sent_num=1):
    '''
    对一段/篇文章做分词，分词结果为列表，每个元素为：每句话的分词结果用空格拼接的字符串
    默认原文一句为分词中的一句即sent_num=1
    (因为分句会按照。；回车等符号断句，
    但对于有些文章或小说，句子比较碎，即句子普遍都很短，则可设置sent_num>1
    则可以设置多句为一句用于分词，这样，一行就能包含较丰富的信息)
    :param fulltext: 全文或段落,即若干句话拼接成的一行str
    :type fulltext: str 
    :param sent_num: 原文几句为分词时的一句输入
    :type sent_num: int
    :return: 
    :rtype: list
    '''
    wordres = []
    # 对整体全文直接做转换效率比每句做一遍高
    # fulltext = str_full2half(fulltext)  # 全角转半角
    # fulltext = Converter('zh-hans').convert(fulltext)  # 繁体转简体
    sentences = SENTSEG.segment(fulltext.lower())
    # logger.info("sentences seg done! cost %d secs , get %d sentences" % ((time.time() - t1), len(sentences)))
    if sentences:
        cnt = 0
        tmpsents=[]
        for i, sent in enumerate(sentences, 1):
            cnt += 1
            tmpsents.append(sent)
            if i%sent_num==0:
                wseg=segword4oneline(''.join(tmpsents))
                if wseg:
                    wordres.append(' '.join(wseg))
                tmpsents=[]
        if tmpsents:
            wseg = segword4oneline(''.join(tmpsents))
            if wseg:
                wordres.append(' '.join(wseg))
    return wordres

def seg4file_book(infile, respath):
    '''
    对文章或小说做分词，一行多句，一个file是一本小说或全文
    本方法原旨在处理一部系列小说，其小说格式较为奇葩,太多断句
    :param infile: 
    :param respath: 
    :return: 
    '''
    alllines=[]
    fsize = util.get_FileSize(infile)
    logger.info("doing %s  and its %.2fMB" %(infile,float(fsize)))
    with codecs.open(infile, 'rU', encoding=ENCODE, errors='replace') as f:
        for line in f:
            l = line.strip()
            if l:
                alllines.append(l)
    fulltext=''.join(alllines)
    del alllines
    if fulltext:
        t2 = time.time()
        wordres=segword4onetext(fulltext)
        timecha = time.time() - t2
        logger.info("%s done. totaly  %.2f MB cost %.1f secs" % (infile, fsize, timecha))
        logger.info(" --------   %.2f MB/sec --------    %.2fMB/min" % (
            float(fsize) / float(timecha), float(fsize) * 60 / float(timecha)))
        if wordres:
            filename = os.path.split(infile)[1]
            util.list2txt(wordres, os.path.join(respath, filename))
        else:
            raise Exception("Word Seg Res is None, Please Check Your Input File!!!")

def seg4file_1line1text(infile, resprefix='',to1line=False, hastitle=False, spliter=None):
    '''
    对单个文件分词，涉及到分句，
    默认输入的一行就是一段完整的文本，例如全文or摘要(与小说不同，小说是整个文件为全文)
    这里分词有很多种方式
    1:一行输入一行输出，即一段文本分词变成一行(to1line=True, hastitle=False)
    2(default):一行输入多行输出，即一段文本先分句变成多行，每行句子再分词(to1line=False, hastitle=False)
    3:输入文本第一列标题，后面是text，text分成一行(类似1),输出类似 title [text分词](to1line=True, hastitle=True)
    4:输入文本第一列标题，后面是text，text分成多行(类似2),而输出每行都要加上标题 (to1line=False, hastitle=True)
    结果文件存放在输入文件目录下segres文件夹内
    分词速度：单核10M/min(XEON) 
    :param infile: 
    :return: 
    '''
    alllines=[]
    fsize = util.get_FileSize(infile)
    mode01='l1' if to1line else 'l0'
    mode02='t1' if hastitle else 't0'
    mode=mode01+mode02
    logger.info("doing %s  and its %.2fMB and split mode is %s" %(infile,float(fsize),mode))
    t2 = time.time()
    cnt=-1
    with codecs.open(infile, 'rU', encoding=ENCODE, errors='replace') as f:
        for line in f:
            cnt+=1
            if cnt%50000==0:
                print("processed line %d" %cnt)
            l = line.strip()
            if not l:
                continue
            title = ''
            if hastitle:
                title = l.split(spliter)[0]
                l = ''.join(l.split(spliter)[1:])
            if to1line:
                wseg=segword4oneline(l)
                if wseg:
                    alllines.append(' '.join([title]+wseg))
            else:
                textseg=segword4onetext(l)
                for ts in textseg:
                    alllines.append("%s %s" %(title,ts))
    timecha = time.time() - t2
    logger.info("%s done. totaly  %.2f MB cost %.1f secs" % (infile, fsize, timecha))
    logger.info(" --------   %.2f MB/sec --------    %.2fMB/min" % (
        float(fsize) / float(timecha), float(fsize) * 60 / float(timecha)))
    if alllines:
        segresfolder=os.path.join(os.path.split(infile)[0],'segres')
        filename = os.path.splitext(os.path.split(infile)[1])[0]
        if not os.path.exists(segresfolder):
            os.mkdir(segresfolder)
        util.list2txt(alllines, os.path.join(segresfolder, '%s_%s_%s.txt' %(filename,resprefix,mode)))
    else:
        raise Exception("Word Seg Res is None, Please Check Your Input File!!!")

def seg4file_1line1sent(infile, respath):
    '''
    对单个文件分词，本方法不涉及到分句，默认输入的一行就是一句
    例如一行对应一段关键词，或者就简单的一行对应一个标题
    '''
    res = []
    t0 = time.time()
    cnt = 0
    fsize = util.get_FileSize(infile)
    logger.info("now doing file : %s and its size is %.2f MB" % (infile, fsize))
    with codecs.open(infile, 'rU', encoding=ENCODE, errors='replace') as f:
        for line in f:
            cnt += 1
            if cnt % 50000 == 0: #差不多1分钟
                logger.info("doing %s line: %d" % (infile, cnt))
            wseg=segword4oneline(line)
            if wseg:
                res.append(' '.join(wseg))
    timecha = time.time() - t0
    logger.info("file %s done. there are totaly %d lines %.2f MB cost %.1f secs" % (infile, cnt, fsize, timecha))
    logger.info(" --------   %.2f MB/sec --------    %.2fMB/min" % (
    float(fsize) / float(timecha), float(fsize) * 60 / float(timecha)))
    if res:
        filename = os.path.split(infile)[1]
        util.list2txt(res, os.path.join(respath, filename))

def single_running(path,respath):
    '''
    单线程跑
    :param path: 文本分割后的路径
    :type path: 
    :param respath: 结果路径文件夹
    :type respath: 
    :return: 
    :rtype: 
    '''
    infiles = util.getfileinfolder(path, prefix='2')
    for i,f in enumerate(infiles):
        print("dong %d th file" %i)
        seg4file_book(f, respath)

def parallel_running(path,cpucnt=10):
    """
    多进程并将对应结果集写入共享资源，维持执行的进程总数，当一个进程执行完毕后会添加新的进程进去(非阻塞)
    :param path: 文本分割后的路径
    :type path: unicode string
    :return: 结果集
    :rtype: list
    """
    from multiprocessing import Pool
    infiles = util.getfileinfolder(path, prefix='rec_abs')
    logger.info("input folder is %s , get %d files in this folder" %(path,len(infiles)))
    num_cpus = cpucnt   # 直接利用multiprocessing.cpu_count()
    pool = Pool(num_cpus)
    pool.map(wrap, infiles)
    pool.close()
    pool.join()

def wrap(inpath):
    """
    多进程执行的包裹函数
    :param inpath: 单文本路径
    :type inpath: unicode string
    :return: 以元组形式返回结果集
    :rtype: tuple
    """
    # seg4file(rec_path,segdatapath)
    # seg4file_book(rec_path, segdatapath)
    seg4file_1line1text(inpath, resprefix='_l1t1',to1line=True,hastitle=True)
    # return amount_and_dict



if __name__ == '__main__':
    # basepath=r'./data'
    # rawdatapath=basepath+r'/data_raw/jinyongquanji'
    # segdatapath=basepath+r'/data_seg/title'
    # seg4file(rawdatapath,segdatapath)
    # seg4file_book(tlbbpath, segdatapathtmp)
    # seg4file_1line1text(Path.path_datahighq5w+'/fn18_5w_summery.txt',
    #                     Path.path_dataseg + '/sumery_highq5w',
    #                     resprefix='_1l1t',to1line=True,hastitle=True)
    # parallel_running('../rec_files/CJFD16-19hexin_others/selected/abs')
    # parallel_running('./otherfile/test.txt',cpucnt=1)
    seg4file_1line1text('./otherfile/test.txt','seg',to1line=False,hastitle=False)
    # single_running(rawdatapath,segdatapath)
    pass