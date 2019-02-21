#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/11/23
import os,json,time,sys
import util_path as Path
import util_common as uc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

datapath=Path.path_dataroot
goodpath=Path.path_datagood
highqpath=Path.path_datahighq
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

def filter_fns(inpath,outpath):
    inli=get_fnlist(inpath,'')
    res=[]
    for fn in inli:
        res.append(fn.lower())
    a= list(set(res))
    uc.list2txt(a, outpath)
    print(len(a))
    return a

def mergefns(path1,path2,respath):
    la=uc.load2list(path1)
    lb=uc.load2list(path2)
    res=list(set(la).union(set(lb)))
    uc.list2txt(res, respath)

def gen_samples(ulog_d, ulog_diff, prefix, outpath):
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

def get_fnkws_by_code(fn_code_file,fn_kws_file):
    dic_codefn = uc.load2dic_02(fn_code_file)

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
    pass
