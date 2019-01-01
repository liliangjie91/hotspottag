#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/28 
import util_dbmysql as msql
# import util_dbkbase as kbase
import util_path as path
import util_common as uc

def get_code_field(code,dic_codefield):

    res0 = 'NULL'
    res1 = 'NULL'
    if isinstance(code, list):
        l0,l1=[],[]
        for c in code:
            tmp0,tmp1=get_code_field(c,dic_codefield)
            l0.append(tmp0)
            l1.append(tmp1)
        res0=';'.join(l0)
        res1=';'.join(l1)
    else:
        if code in dic_codefield:
            res1 = dic_codefield[code]
            if '_' in code:
                subcode=code[:code.find('_')]
                if subcode in dic_codefield:
                    res0=dic_codefield[subcode]
        if res0 is 'NULL':
            res0=res1
    return res0,res1

def write2mysql():
    db=msql.mysql()
    fn_code=uc.load2dic(path.path_dataraw+'/sample1w/fn_code_1811_1w')
    fn_title = uc.load2dic(path.path_dataraw + '/sample1w/fn_title_1811_1w')
    fn_kws = uc.load2dic(path.path_dataraw + '/sample1w/fn_kws_1811_1w')
    fn_sum = uc.load2dic(path.path_dataraw + '/sample1w/fn_summery_1811_1w')
    code_fields = uc.load2dic(r'source/field_dic.txt')

    tmpfn,tmpcode,tmptitle,tmpkws,tmpfield0,tmpfield1,tmpsum=[],[],[],[],[],[],[]
    tablename='papertag'
    cnt=0
    fields = ['filename', 'title', 'kwords_jb', 'fieldcode','field0','field','abstract']
    print('writing data to mysql table %s' %tablename)
    for index,key in enumerate(fn_title.keys()):
        if cnt>=500:
            break
        if index%100==2 and tmptitle:
            values=[tmpfn,tmptitle,tmpkws,tmpcode,tmpfield0,tmpfield1,tmpsum]
            # db.insertmany_bylist(tablename, fields, values)
            db.insertmany_byfor(tablename,fields,values)
            tmpfn, tmpcode, tmptitle, tmpkws, tmpfield0, tmpfield1 ,tmpsum= [], [], [], [], [], [],[]
        if key in fn_title and key in fn_kws and key in fn_code:
            cnt+=1
            print cnt
            # tmpfn.append('"%s"' %key)
            # tmptitle.append('"%s"' %fn_title[key].replace('"',r'\"'))
            # fieldcodes=fn_code[key]
            # tmpcode.append('"%s"' %(';'.join(fieldcodes) if isinstance(fieldcodes,list) else fieldcodes))
            # tmpkws.append('"%s"' %';'.join(fn_kws[key]))
            # tmpsum.append('"%s"' %(fn_sum[key].replace('"',r'\"') if key in fn_sum else "NULL"))
            # res0,res1=get_code_field(fn_code[key],code_fields)
            # tmpfield0.append('"%s"' %res0)
            # tmpfield1.append('"%s"' %res1)
            tmpfn.append(key)
            tmptitle.append(fn_title[key].replace('"', r'\"'))
            fieldcodes = fn_code[key]
            tmpcode.append((';'.join(fieldcodes) if isinstance(fieldcodes, list) else fieldcodes))
            tmpkws.append(';'.join(fn_kws[key]))
            tmpsum.append(((fn_sum[key].replace('"', r'\"')) if key in fn_sum else "NULL"))
            res0, res1 = get_code_field(fn_code[key], code_fields)
            tmpfield0.append(res0)
            tmpfield1.append(res1)
    if tmpfn:
        print "rest res..."
        values = [tmpfn, tmptitle, tmpkws, tmpcode ,tmpfield0, tmpfield1,tmpsum]
        db.insertmany_bylist(tablename, fields, values)
    print('insert cnt : %d' %cnt)
    db.close()

def run01():
    write2mysql()

if __name__ == '__main__':
    run01()
    # tmp()