#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2019/3/15 


import codecs
import logging
import os
from tc_conversion.langconv import *
from tc_conversion.full_half_conversion import *
import util_common as uc

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


'''
rec文件中其中一篇文献的示例

<REC>
<篇名>=基于改进迭代收缩阈值算法的微观3D重建方法
<中文关键词>=微观3D重建;;离焦深度恢复;;迭代收缩阈值算法;;加速算子梯度估计;;割线线性搜索
<中文摘要>=迭代收缩阈值算法(ISTA)求解离焦深度恢复动态优化问题时,采用固定迭代步长,导致算法收敛效率不佳,使得重建的微观3D形貌精度不高。为此,提出一种基于加速算子梯度估计和割线线性搜索的方法优化ISTA——FL-ISTA。首先,在每一次迭代中,由当前点和前一个点的线性组合构成加速算子重新进行梯度估计,更新迭代点;其次,为了改变迭代步长固定的限制,引入割线线性搜索,动态确定每次最优迭代步长;最后,将改进的迭代收缩阈值算法用于求解离焦深度恢复动态优化问题,加快算法的收敛速度、提高微观3D形貌重建的精度。在对标准500 nm尺度栅格的深度信息重建实验中,与ISTA、快速ISTA(FISTA)和单调快速ISTA(MFISTA)相比,FL-ISTA收敛速度均有所提升,重建的深度信息值下降了10个百分点,更接近标准500 nm栅格尺度;与ISTA相比,FL-ISTA重建的微观3D形貌均方差(MSE)和平均误差分别下降了18个百分点和40个百分点。实验结果表明,FL-ISTA有效提升了求解离焦深度恢复动态优化问题的收敛速度,提高了微观3D形貌重建的精度。
<引文>=[1]LI C,SU S,MATSUSHITA Y,et al.Bayesian depth-from-defocus with shading constraints[J].IEEE Transactions on Image Processing,2016,25(2):589-600.
[2]TAO M W,SRINIVASAN P P,HADAP S,et al.Shape etimation from shaping,defocus,and correspondence using light-field angular coherence[J].IEEE Transactions on Pattern Analysis and Machine Intelligence,2017,39(3):546-560.
[3]PENTLAND A P.A new sense for depth of field[J].IEEE Transactions on Pattern Analysis and Machine Intelligence,1987,9(4):523-531.
[4]NAYAR S K,W ATANABE M,NOGU CH IM.Real time focus range sensor[J].IEEE Transactions on Pattern Analysis and Machine Intelligence,1996,18(12):1186-1198.
[5]SUBBARAO M,SURYA G.Depth from defocus:a spatial domain approach[J].International Journal of Computer Vision,1994,13(3):271-294.
[6]杨洁,田翠萍,钟桂生.随机光学重构显微成像技术及其应用[J].光学学报,2017,37(3):44-56.(YANG J,TIAN C P,ZHONG G S.Stochastic optical reconstruction microscopy and its application[J].Acta Optica Sinica,2017,37(3):44-56.).
[7]魏阳杰,董再励,吴成东.摄像机参数固定的全局离焦深度恢复[J].中国图象图形学报,2010,15(12):1811-1817.(WEI Y J,DONG Z L,WU C D.Global shape reconstruction with fixed camera parameters[J].Journal of Image and Graphics,2010,15(12):1811-1817).
[8]FAVARO P,SOATTO S,BURGER M,et al.Shape from defocus via diffusion[J].IEEE transactions on Pattern Analysis and Machine intelligence,2008,30(3):518-531.
[9]FAVARO P,MENNUCCI A,SOATTO S.Observing shape from defocused images[J].International Journal of Computer Vision,2003,52(1):25-43.
[10]BECK A,TEBOULLE M.A fast iterative shrinkage-thresholding algorithm for linear inverse problems[J].SIAM Journal on Imaging Sciences,2009,2(1):183-202.
[11]ZIBETTI M V W,HELOU E S,PIPA D R.Accelerating overrelaxed and monotone fast iterative shrinkage-thresholding algorithms with line search for sparse reconstructions[J].IEEE Transactions on Image Processing,2017,26(7):3569-3578.
[12]ZIBETTI M V W,PIPA D R,DE PIERRO A R.Fast and exact unidimensional L2-L1 optimization as an accelerator for iterative reconstruction algorithms[J].Digital Signal Processing,2016,48:178-187.
<文件名>=JSJY201808044
<专题子栏目代码>=I138_C11;
<出版日期>=2018-04-19 11:10
<机标关键词>=阈值算法,迭代点,线性搜索,离焦,微观3D重建,动态优化问题,深度信息,算法收敛,梯度,均方差,
<页数>=7
<文件大小>=464K
<下载频次>=61
<影响因子>=0.941
<基金代码>=0001;
<作者代码>=39828566;39828567;24165362;16371199;35049372;24165363;
<机构代码>=0041682;0192280;0111402;
<通讯作者>=张明新;
<行业分类代码>=128012543
<第一作者H指数>=0
'''


#用于处理rec文件，rec文件即kbase数据库批量导出的文件，内部有若干篇文献信息,例如篇名，关键词，作者信息，全文，摘要等
#主要函数：preprocess 预处理，半全角转换，繁简转换，转码等

recfolder = '../rec_files'

def prepeocess(path):
    '''
    预处理，主要是 全半角转换，繁简转换，转换为utf8编码
    :param path: 
    :type path: 
    :return: 
    :rtype: 
    '''
    logger.info("for file : %s" %path)
    with open(path+'.preproed.txt','wb') as fw:
        with codecs.open(path, 'rU', 'gb18030', errors='replace') as f:  # REC文件为gb18030编码
            for line in f:
                line = line.strip()
                line = str_full2half(line)        # 检测全角字符，将全角字符转换成半角
                line = Converter('zh-hans').convert(line)         # 繁体字检测，将繁体字转换成简体字
                fw.write((line+'\n').encode('utf-8'))

def get_features_fromRec(path,resfolder,iffulltext=False,allfeature=False):
    '''
    读取预处理后的rec文件，并从中获取不同字段，保存之,为了便于读取，
    将保存4个文件：文件名+标题+摘要；文件名+标题+全文；文件名+中文关键词+机标关键词；文件名+各种其他特征
    :param path: rec单个文件的路径
    :type path: str
    :param resfolder:结果保存文件夹 
    :type resfolder: str
    :return: 
    :rtype: 
    '''
    res_abstract=[]     # 存放 摘要
    res_fatures=[]      # 存放 各种数字特征
    res_fulltext=[]     # 存放 全文
    res_kws=[]          # 存放 关键词信息(中文关键词，机标关键词)

    filename=''         #文件名
    title=''            #篇名，标题
    abstract=''         #中文摘要
    fulltext=''         #全文
    kws=''              #中文关键词
    kws_jb=''           #机标关键词
    authorcode=''       #作者代码
    jigoucode=''        #机构代码
    code_zhuanti=''     #专题子栏目代码
    code_hangye=''      #行业分类代码
    date=''             #出版日期
    pages=''            #页数
    size=''             #文件大小
    downtime=''         #下载频次
    citedtime=''        #被引频次
    impact_factor=''    #影响因子-期刊
    fund_code=''        #基金代码
    cite_paper=''       #引文

    recfilename = os.path.split(path)[1]
    logger.info("---------------now processing file : %s" %path)
    cnt=-1
    with codecs.open(path, 'rU', 'utf8', errors='replace') as f:  # 预处理后的REC文件为utf8编码
        for line in f:
            cnt+=1
            if cnt%5000000==0:
                logger.info("now processing %d in file %s" %(cnt,path))
            line = line.strip()
            if line.find('<文件名>=') == 0: filename = line[6:].upper()
            elif line.find('<篇名>=')==0: title = line[5:]
            elif line.find('<中文摘要>=') == 0: abstract = line[7:]
            elif line.find('<全文>=') == 0:
                title_index = line.find(title + '@')  # 去掉全文末尾的标题、作者、摘要、引文等
                fulltext = line[5:title_index]
            elif line.find('<中文关键词>=') == 0: kws = line[8:]
            elif line.find('<机标关键词>=') == 0: kws_jb = line[8:]
            elif line.find('<作者代码>=') == 0: authorcode = line[7:]
            elif line.find('<机构代码>=') == 0: jigoucode = line[7:]
            elif line.find('<专题子栏目代码>=') == 0: code_zhuanti = line[10:]
            elif line.find('<行业分类代码>=') == 0: code_hangye = line[9:]
            elif line.find('<出版日期>=') == 0: date = line[7:17]
            elif line.find('<页数>=') == 0: pages = line[5:]
            elif line.find('<文件大小>=') == 0: size = line[7:]
            elif line.find('<下载频次>=') == 0: downtime = line[7:]
            elif line.find('<被引频次>=') == 0: citedtime= line[7:]
            elif line.find('<影响因子>=') == 0: impact_factor = line[7:]
            elif line.find('<基金代码>=') == 0: fund_code = line[7:]
            elif line.find('<引文>=') == 0: cite_paper = line[5:]
            elif line.find('<REC>') == 0:
                if filename and filename is not 'null':
                    if not iffulltext or allfeature:
                        res_fatures.append(' '.join([filename,downtime,citedtime,
                                              size,pages,date,impact_factor,
                                              fund_code,authorcode,jigoucode,code_zhuanti,
                                              code_hangye]))
                        res_abstract.append(' '.join([filename,title,abstract]))
                        res_kws.append(' '.join([filename,kws,kws_jb]))
                    if iffulltext or allfeature:
                        res_fulltext.append(' '.join([filename, title, fulltext]))
                #储存后再初始化各个元素
                filename,title,abstract,fulltext,kws,kws_jb,authorcode = 'null','null','null','null','null','null','null'
                jigoucode,code_zhuanti,code_hangye,date,pages,size = 'null','null','null','null','null','null'
                downtime,citedtime,impact_factor,fund_code,cite_paper = 'null','null','null','null','null'

    if filename and filename is not 'null':
        res_fatures.append(' '.join([filename, downtime, citedtime,size,
                                     pages, date, impact_factor,fund_code,
                                     authorcode, jigoucode, code_zhuanti,code_hangye]))
        res_abstract.append(' '.join([filename, title, abstract]))
        res_kws.append(' '.join([filename, kws, kws_jb]))
        if iffulltext or allfeature:
            res_fulltext.append(' '.join([filename, title, fulltext]))
    if not iffulltext or allfeature:
        uc.list2txt(res_abstract,resfolder+'/rec_abs_%s' %recfilename)
        uc.list2txt(res_kws, resfolder + '/rec_kws_%s' % recfilename)
        uc.list2txt(res_fatures, resfolder + '/rec_feas_%s' % recfilename)
    if iffulltext or allfeature:
        uc.list2txt(res_fulltext, resfolder + '/rec_fulls_%s' % recfilename)

def read_rec_data(path):
    """
    读取REC文本文件
    :param path: REC文件所在路径
    :type path: unicode string
    :returns:篇名，正文组成的列表以及中文摘要
    """
    doc = []
    # abstract = ''
    date = ''
    periodical_title = ''
    impact_factor = ''
    download_frequency = ''
    cite_frequency = ''
    filename = ''
    fulltext_count = 0

    with codecs.open(path, 'rU', 'gb18030', errors='replace') as f:  # REC文件为gb18030编码
        for line in f:
            line = line.strip()
            line = str_full2half(line)        # 检测全角字符，将全角字符转换成半角
            line = Converter('zh-hans').convert(line)         # 繁体字检测，将繁体字转换成简体字
            if line.find(u'<篇名>=') == 0:
                title = line[5:]
            # elif line.find(u'<中文摘要>=') == 0:
            #     abstract = line[7:]
            elif line.find(u'<出版日期>=') == 0:
                date = line[7:]
            elif line.find(u'<中英文刊名>=') == 0:
                periodical_title = line[8:]
            elif line.find(u'<影响因子>=') == 0:
                impact_factor = line[7:]
            elif line.find(u'<下载频次>=') == 0:
                download_frequency = line[7:]
            elif line.find(u'<被引频次>=') == 0:
                cite_frequency = line[7:]
            elif line.find(u'<文件名>=') == 0:
                filename = line[6:]
            elif line.find(u'<全文>=') == 0:
                title_index = line.find(title + '@')  # 去掉全文末尾的标题、作者、摘要、引文等
                if title_index != -1:
                    text = line[5:title_index]
                else:
                    text = line[5:]
                fulltext_count = len(text)
                doc.append((title, text))
    # return doc, abstract
    return doc, download_frequency, cite_frequency, date, periodical_title, impact_factor, filename, fulltext_count

def kws_split(path):
    #分割关键词序列，去除中间的标点符号，一般包括[;,.；，。]
    kws = uc.load2list(path)
    res=[]
    for line in kws:
        line=line.strip()
        linea=re.split('[;, ；，]+',line)
        res.append(' '.join(linea))
    uc.list2txt(res,path+'.splited.txt')

def parallel_running(path,cpus=10):
    """
    多进程并将对应结果集写入共享资源，维持执行的进程总数，当一个进程执行完毕后会添加新的进程进去(非阻塞)
    :param path: 文本分割后的路径
    :type path: unicode string
    :return: 结果集
    :rtype: list
    """
    from multiprocessing import Pool
    rec_path = uc.getfileinfolder(path,prefix='rec_kws_')
    pool = Pool(cpus)
    pool.map(wrap, rec_path)
    pool.close()
    pool.join()


def wrap(rec_path):
    """
    多进程执行的包裹函数
    :param rec_path: 单文本路径
    :type rec_path: unicode string
    :return: 以元组形式返回结果集
    :rtype: tuple
    """
    # prepeocess(rec_path)
    # get_features_fromRec(rec_path,os.path.split(rec_path)[0]+'/../selected/',allfeature=True)
    kws_split(rec_path)
    # pos = rec_path.find('.txt')
    # name = rec_path[pos - 11:pos]
    # save_seg_rec(rec_path, path.input_data_path, name)
    # return amount_and_dict


if __name__ == '__main__':
    # prepeocess('./otherfile/testrec.txt')
    parallel_running(recfolder+'/CJFD16-19hexin_others/selected/',14)
    # get_features_fromRec('./otherfile/testrec2.txt','./otherfile')