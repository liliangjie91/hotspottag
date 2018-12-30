#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/28 
import MySQLdb
import util_path as myconf

class mysql(object):
    """docstring for mysql"""

    def __init__(self, host=None,dbname='test',charset='utf8'):
        self.host = host if host else myconf.mysqlhost
        self.user = myconf.mysqluser
        self.passwd = myconf.mysqlpasswd
        self.dbname = dbname
        self.charset = charset
        self._conn = None
        self._connect()
        self._cursor = self._conn.cursor()

    def _connect(self):
        try:
            self._conn = MySQLdb.connect(host=self.host,
                                         user=self.user,
                                         passwd=self.passwd,
                                         db=self.dbname,
                                         charset=self.charset)
        except MySQLdb.Error, e:
            print e

    def query(self, sql):
        try:
            result = self._cursor.execute(sql)
        except MySQLdb.Error, e:
            print e
            result = False
        return result

    def select(self, table, column='*', condition=''):
        condition = ' where ' + condition if condition else None
        if condition:
            sql = "SELECT %s from %s %s" % (column, table, condition)
        else:
            sql = "select %s from %s" % (column, table)
        self.query(sql)
        return self._cursor.fetchall()

    def insertmany_bydic(self,table,tdict):
        field0=[]
        values0=[]
        for key in tdict:
            field0.append(key)
            v = tdict[key] if isinstance(tdict[key],list) else [tdict[key]]
            values0.append(v)
        field= ','.join(field0)
        values = zip(*values0)
        sql="insert into %s (%s) VALUES " %(table,field)
        vs="%s,"*len(field0)
        vs="(%s)" %(vs[:-1])
        sql +=vs
        self._cursor.executemany(sql,values)
        self._conn.commit()
        return self.affected_num()

    def insertmany_bylist(self, table, fields, values):
        '''
        fields:['name','gender','age']
        values:[['zhangsan','lisi'],['male','male'],[21,22]]
        '''
        field= ','.join(fields)
        value = zip(*values)
        sql="insert into %s (%s) VALUES " %(table,field)
        vs="%s,"*len(fields)
        vs="(%s)" %(vs[:-1])
        sql +=vs
        self._cursor.executemany(sql,value)
        self._conn.commit()
        return self.affected_num()

    def insertmany_byfor(self, table, fields, values):
        '''
        fields:['name','gender','age']
        values:[['zhangsan','lisi'],['male','male'],[21,22]]
        '''
        field= ','.join(fields)
        value = zip(*values)
        for v in value:
            vv=','.join(v)
            try:
                sql = "insert into %s (%s) VALUES (%s)" % (table, field,vv)
                self._cursor.execute(sql)
            except MySQLdb.Error, e:
                print 'insert fail: %s' %vv
        self._conn.commit()
        return self.affected_num()

    def insert(self, table, tdict):
        column = ''
        value = ''
        for key in tdict:
            column += ',' + key
            value += "','" + tdict[key]
        column = column[1:]
        value = value[2:] + "'"
        sql = "insert into %s(%s) values(%s)" % (table, column, value)
        self._cursor.execute(sql)
        self._conn.commit()
        return self._cursor.lastrowid  # 返回最后的id

    def update(self, table, tdict, condition=''):
        if not condition:
            print "must have id"
            exit()
        else:
            condition = 'where ' + condition
        value = ''
        for key in tdict:
            value += ",%s='%s'" % (key, tdict[key])
        value = value[1:]
        sql = "update %s set %s %s" % (table, value, condition)
        self._cursor.execute(sql)
        return self.affected_num()  # 返回受影响行数

    def delete(self, table, condition=''):
        condition = 'where ' + condition if condition else None
        sql = "delete from %s %s" % (table, condition)
        # print sql
        self._cursor.execute(sql)
        self._conn.commit()
        return self.affected_num()  # 返回受影响行数

    def rollback(self):
        self._conn.rollback()

    def affected_num(self):
        return self._cursor.rowcount

    def __del__(self):
        try:
            self._cursor.close()
            self._conn.close()
        except:
            pass

    def close(self):
        self.__del__()


if __name__ == '__main__':
    db = mysql()
    createsql='''create table test03(
              filename VARCHAR(20) NOT NULL PRIMARY KEY,
              title VARCHAR(150) NOT NULL,
              fieldcode VARCHAR(30),
              field VARCHAR(50),
              tag VARCHAR(150),
              kwords_jb VARCHAR(150),
              abstract VARCHAR(3000),
              abstract_seg VARCHAR(3000)
              )ENGINE=InnoDB DEFAULT CHARSET=utf8'''

    db.query(createsql)
    # print db.select('paper')
    # print db.select('msg','id,ip,domain','id>2')
    # print db.affected_num()

    # tdict = {
    #     'filename':['llj1234555.hn','ll001234.hn'],
    #     'downtime':['10','5'],
    #     'kw_jb':[u'你好;计算机;刀;咖啡',u'计算机;机器学习;推荐系统;太古'],
    #     'title':[u'深度应用',u'学习123']
    # }
    # tdict = {
    #     'filename': 'test1234555.hn',
    #     'downtime': '1',
    #     'kw_jb': u'刀;咖啡',
    #     'title': u'深度应用123'
    # }
    # print db.insertmany('paper', tdict)
    # print db.select('paper')

    # tdict = {
    #     'ip':'111.13.100.91',
    #     'domain':'aaaaa.com'
    # }
    # print db.update('msg', tdict, 'id=5')

    # print db.delete('msg', 'id>3')

    db.close()