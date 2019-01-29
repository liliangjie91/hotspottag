#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/28 
import util_dbmysql as msql
# import util_dbkbase as kbase
import util_path as path
import util_common as uc
CUR_FIELDCODE_DIC = uc.load2dic(r'source/field_dic.txt')

def write2mysqlbylist(startfrom=0,count=0):
    db=msql.mysql()
    datapath=path.path_dataraw
    # datapath=path.path_dataraw+'/sample1w'
    fn_code=uc.load2dic(datapath+'/fn_code_1811')
    fn_title = uc.load2dic(datapath + '/fn_title_1811')
    fn_kws = uc.load2dic(datapath + '/fn_kws_1811')
    fn_sum = uc.load2dic(datapath + '/fn_summery_1811')

    tmpfn,tmpcode,tmptitle,tmpkws,tmpfield0,tmpfield1,tmpsum=[],[],[],[],[],[],[]
    tablename='papertag'
    cnt=0
    realyget=0
    fields = ['filename', 'title', 'kwords_jb', 'fieldcode','field0','field','abstract']
    print('writing data to mysql table %s' %tablename)
    for index,key in enumerate(fn_title.keys()):
        cnt += 1
        if cnt%10000==0:
            print cnt,realyget
        if cnt<startfrom:
            continue
        if count:
            if cnt-startfrom>count:
                break
        if index%10000==2 and tmptitle:
            values=[tmpfn,tmptitle,tmpkws,tmpcode,tmpfield0,tmpfield1,tmpsum]
            db.insertmany_bylist(tablename, fields, values)
            tmpfn, tmpcode, tmptitle, tmpkws, tmpfield0, tmpfield1 ,tmpsum= [], [], [], [], [], [],[]
        if key in fn_title and key in fn_kws and key in fn_code:
            realyget+=1
            tmpfn.append("'%s'" %key)
            tmptitle.append("'%s'" %(fn_title[key].replace("'", "''")))
            tmpkws.append("'%s'" %fn_kws[key].replace("'", "''"))
            fnsum = fn_sum[key] if key in fn_sum else "NULL"
            tmpsum.append("'%s'" %fnsum.replace("'", "''"))
            fieldcodes = fn_code[key]
            fieldcodes = fieldcodes.strip(';')
            tmpcode.append("'%s'" %fieldcodes)
            res0, res1 = uc.get_code_field(fieldcodes, dic_codefield=CUR_FIELDCODE_DIC)
            tmpfield0.append("'%s'" %res0)
            tmpfield1.append("'%s'" %res1)
    if tmpfn:
        print "rest res..."
        values = [tmpfn, tmptitle, tmpkws, tmpcode ,tmpfield0, tmpfield1,tmpsum]
        db.insertmany_bylist(tablename, fields, values)
    print('insert cnt : %d' %realyget)
    db.close()

def write2mysqlbyfor(startfrom=0,count=0):
    db=msql.mysql()
    datapath=path.path_dataraw
    # datapath=path.path_dataraw+'/sample1w'
    fn_code=uc.load2dic(datapath+'/fn_code_1811')
    fn_title = uc.load2dic(datapath + '/fn_title_1811')
    fn_kws = uc.load2dic(datapath + '/fn_kws_1811')
    fn_sum = uc.load2dic(datapath + '/fn_summery_1811')

    tmpfn,tmpcode,tmptitle,tmpkws,tmpfield0,tmpfield1,tmpsum=[],[],[],[],[],[],[]
    tablename='papertag'
    cnt=0
    realyget=0
    fields = ['filename', 'title', 'kwords_jb', 'fieldcode','field0','field','abstract']
    print('writing data to mysql table %s' %tablename)
    for index,key in enumerate(fn_title.keys()):
        cnt += 1
        if cnt%100000==0:
            print cnt,realyget
        if cnt<startfrom:
            continue
        if count:
            if cnt-startfrom>count:
                break
        if index%10000==2 and tmptitle:
            values=[tmpfn,tmptitle,tmpkws,tmpcode,tmpfield0,tmpfield1,tmpsum]
            db.insertmany_byfor(tablename,fields,values)
            tmpfn, tmpcode, tmptitle, tmpkws, tmpfield0, tmpfield1 ,tmpsum= [], [], [], [], [], [],[]
        if key in fn_title and key in fn_kws and key in fn_code:
            realyget+=1
            # tmpfn.append(key)
            # tmptitle.append(fn_title[key].replace('"', r'\"'))
            # tmpkws.append(fn_kws[key])
            # fnsum=fn_sum[key] if key in fn_sum else "NULL"
            # tmpsum.append(fnsum.replace('"', r'\"'))
            # fieldcodes = fn_code[key]
            # fieldcodes = fieldcodes.strip(';')
            # tmpcode.append(fieldcodes)
            # res0, res1 = uc.get_code_field(fieldcodes,dic_codefield=CUR_FIELDCODE_DIC)
            # tmpfield0.append(res0)
            # tmpfield1.append(res1)
            tmpfn.append("'%s'" %key)
            tmptitle.append("'%s'" %(fn_title[key].replace("'", "''")))
            tmpkws.append("'%s'" %fn_kws[key].replace("'", "''"))
            fnsum = fn_sum[key] if key in fn_sum else "NULL"
            tmpsum.append("'%s'" %fnsum.replace("'", "''"))
            fieldcodes = fn_code[key]
            fieldcodes = fieldcodes.strip(';')
            tmpcode.append("'%s'" %fieldcodes)
            res0, res1 = uc.get_code_field(fieldcodes, dic_codefield=CUR_FIELDCODE_DIC)
            tmpfield0.append("'%s'" %res0)
            tmpfield1.append("'%s'" %res1)
    if tmpfn:
        print "rest res..."
        values = [tmpfn, tmptitle, tmpkws, tmpcode ,tmpfield0, tmpfield1,tmpsum]
        db.insertmany_bylist(tablename, fields, values)
    print('insert cnt : %d' %realyget)
    db.close()

def run01():
    write2mysqlbyfor(startfrom=2200000)

if __name__ == '__main__':
    run01()
    # tmp()