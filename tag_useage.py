#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/28 
import util_dbmysql as msql
# import util_dbkbase as kbase
import util_path as path
import util_common as uc

def write2mysql():
    db=msql.mysql()
    fn_code=uc.load2dic(path.path_dataraw+'/fn_code_1811')
    fn_title = uc.load2dic(path.path_dataraw + '/fn_title_1811_1w')
    fn_kws = uc.load2dic(path.path_dataraw + '/fn_kws_1811_1w')
    tmpfn,tmpcode,tmptitle,tmpkws=[],[],[],[]
    tablename='test03'
    cnt=0
    fields = ['filename', 'title', 'kwords_jb', 'fieldcode']
    print('writing data to mysql table %s' %tablename)
    for index,key in enumerate(fn_title.keys()):
        if cnt>=6500:
            break
        if index%500==1 and tmptitle:
            values=[tmpfn,tmptitle,tmpkws,tmpcode]
            # db.insertmany_bylist(tablename, fields, values)
            db.insertmany_byfor(tablename,fields,values)
            tmpfn, tmpcode, tmptitle, tmpkws = [], [], [], []
        if key in fn_title and key in fn_kws and key in fn_code:
            cnt+=1
            tmpfn.append(key)
            tmptitle.append(fn_title[key])
            tmpcode.append(fn_code[key])
            tmpkws.append(';'.join(fn_kws[key]))
    if tmpfn:
        values = [tmpfn, tmptitle, tmpkws, tmpcode]
        db.insertmany_bylist(tablename, fields, values)
    print('insert cnt : %d' %cnt)
    db.close()

def run01():
    write2mysql()

if __name__ == '__main__':
    run01()