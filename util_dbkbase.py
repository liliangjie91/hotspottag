#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wx on 2016/7/13
import codecs
import sys
#reload(sys)
sys.setdefaultencoding('utf-8')
try:
    import KBase
except ImportError:
    import libkbaseclientu_python as KBase

import util_path

__version__ = '2018-04-18'

# try:
#     callable
# except NameError:
#     def callable(obj):
#         return any('__call__' in cls.__dict__ for cls in type(obj).__mro__)


class NoIndexFiledFoundError(Exception):
    """没有索引值报错"""


class PooledKBaseError(Exception):
    """综合KBase数据库连接错误"""


class NotConnectionError(PooledKBaseError):
    """非法host或错误psd或非法usr或错误端口导致数据库模型无法连接报错"""


class NotClosedError(PooledKBaseError):
    """数据库连接使用后未关闭错误"""


class PoolExhaustedError(PooledKBaseError):
    """ 链接池中的链接已经用尽"""


class ConnectionNotFoundError(PooledKBaseError):
    """ 链接池中的链接没有找到"""


class NoRelativeMethodError(PooledKBaseError):
    """没有相关kbase获取数据集接口"""


class DataBase(object):
    """
    通过KBase数据库取数据信息
    Parameters
    ----------
    host : default=192.168.106.101 需要连接的服务器ip地址，缺省值是101服务器

    Examples
    --------
    ->>>from kbase import DataBase
    ->>>kb = DataBase()
    ->>>sql = u"select 篇名,作者,被引频次 from CJFDTOTAL where SMARTS = '糖尿病' order by FFD"
    ->>>n = 5
    ->>>test = kb.database_handle(sql=sql, mode='batch', top=n)
    ->>>print(test)
    ->>>[[u'人生健康一定要知道“四多”“四少”', u'', u'0'], [u'保加利亚的藥品生產', u'格魯波夫;大謙;', u'']

    --------
    ->>>from novelty import Interface4Novelty
    ->>>nv = Interface4Novelty()
    ->>>nv.add_external_character4data(title=u'机构')
    """

    def __init__(self, host=u'192.168.100.113'):
        """
        初始化类，参数数据库ip地址
        """
        self.host = host
        # 大文本字段类型和二进制字段类型列表
        self.large_list = ['HSD_LTEXT', 'HSD_SOB', 'HSD_LOB', 'HSD_DOB']
        # 枚举字典，key-value分别对应字段类型和字段类型值
        self.dict = self.enumerate(HSD_INTEGER=0x400, HSD_DATE=0x401, HSD_TIME=0x402, HSD_NUM=0x403,
                                   HSD_INT64=0x404, HSD_AUTO=0x405, HSD_CHAR=0x406, HSD_VCHAR=0x407,
                                   HSD_MVCHAR=0x408, HSD_LMVCHAR=0x409, HSD_VECTOR=0x40a, HSD_MTEXT=0x40b,
                                   HSD_LTEXT=0x40c, HSD_VIRTUAL=0x40d, HSD_SOB=0x40e, HSD_LOB=0x40f,
                                   HSD_DOB=0x410, HSD_DOCUMENT=0x411, HSD_DOCPATH=0x412, HSD_TNAME=0x413,
                                   HSD_TANAME=0x414, HSD_RECORDID=0x415, HSD_RELEVANT=0x416, HSD_SNAPSHOT=0x417,
                                   HSD_XML=0x418, HSD_PTEXT=0x419, HSD_NULL=0x41a)

    def enumerate(self, **kwargs):
        """
        构造枚举字典，为检查索引字段在数据库中存储类型做准备
        :param kwargs: 可变参数
        :type kwargs: dict
        :return: 构造的枚举字典
        :rtype: dict
        """
        _dict = {}
        for key, value in kwargs.items():
            _dict[value] = key
        return _dict

    def kbase_dict(self, path):
        """
        读取保存KBase中CJFD表的所有字段信息的文本成字典
        :param path: 保存文本的路径
        :type path: unicode string
        :return: 若读取成功，返回字典格式的字段信息
        :rtype: dict
        """
        try:
            _dict = {}
            with codecs.open(path, 'rU', 'gb18030', errors='replace') as f:
                for line in f.readlines():
                    (key, value) = line.strip().split(':')
                    _dict[key] = int(value)
            return _dict
        except IOError as error:
            print('except:', error)
            exit(-1)

    def __database_retrieval_results(self, tpiclient, retrieval_list, mode, *col_num):
        """
        获取检索字段的列长度，将对应字段结果集编码格式转换后保存到列表，游标自动跳转到下一个记录集上
        :param tpiclient: 传入游标后的对象
        :type tpiclient: KBase.TPIClient
        :param retrieval_list: 保存检索结果集的列表
        :type retrieval_list: list
        :param *col_num: 非关键字可变参数
        :type *col_num: tuple
        """
        save_list = list()
        for i in range(len(col_num[0])):
            # 通过字段序号获取字段类型对应的字段类型定义值，例如tpi_getFieldType(26) 返回（1036，）即错误码或字段类型值
            field_type = tpiclient.getFieldType(col_num[0][i])
            # 判断取数据时采用何种接口，当数据>32kb时（即字段类型为二进制类型或大文本类型）
            # 采用二进制取数据，并对所取数据做编码转换
            if self.dict[field_type[0]] in self.large_list:
                col_num_len = tpiclient.getColDataLen(col_num[0][i])
                # getBlob方法需要设置偏移位，辅助多次截断取数据，当前服务针对一次性取，偏移位为结果集真实长度
                _, field_name_swap = tpiclient.getBLob(col_num[0][i], 0, col_num_len[0])
                if field_name_swap:
                    field_name_swap = field_name_swap.decode('gb18030', 'replace')
            # 当数据<=32kb时，采用取小数据接口，返回数据类型为unicode
            else:
                # col_num_len = tpiclient.getColDataLen(col_num[0][i])
                # _, field_name_swap = tpiclient.getBLob(col_num[0][i], 0, col_num_len[0])
                _, field_name_swap = tpiclient.getFieldValue(col_num[0][i])
                # field_name = unicode(field_name_swap.decode('gb18030', 'replace'))
            if field_name_swap:
                save_list.append(field_name_swap)
            # 存在检索的字段存在，但是字段结果集是空的情况
            else:
                save_list.append(None)
        retrieval_list.append(save_list)
        # 如果结果集中有多行结果，根据top值的设定，启动moveNext方法获取多行结果集
        if mode == 'batch':
            tpiclient.moveNext()

    def __column_num_index(self, field):
        """
        通过检索的字段信息去字典中查找对应KBase表中的字段序号
        :param field: 检索的字段名称
        :type field: unicode string
        :return: 若检索字段部位空并且检索字段在KBase表中存在，返回字段的序号
        :rtype: int
        """
        try:
            self._dict = self.kbase_dict(util_path.kbase_dict_path)
            assert field
            if self._dict.has_key(field):
                return self._dict[field]
            else:
                print('FATAL ERROR: THE TABLE HAS NO %s FIELD!' % field)
                exit(-1)
        except AssertionError as e:
            print("%s:%s" % (e.__class__.__name__, e))

    def get_field_position(self, table_name, tpiclient, str_):
        """
        构造数据库字段字典（原生接口调用）
        :param table_name: 需要查询的数据表名
        :type table_name:  string or sequence of string
        :param tpiclient: 传入游标后的对象
        :type tpiclient: KBase.TPIClient
        :return: 字段号列表
        :rtype: list
        """
        check_list = []
        try:
            for each_str in str_:
                # 获取当前字段信息对应表结构中的字段序号
                hField = tpiclient.getFieldIndex(table_name, each_str)[0]
                if hField >= 0:
                    check_list.append(hField)
                else:
                    # 无检索的字段，直接抛出异常
                    raise NoIndexFiledFoundError
            return check_list
        except:
            print("FATAL ERROR: THERE IS NO INDEX FILED!")
            raise NoIndexFiledFoundError

    def __check_sql_old(self, sql):
        """
        检查sql语句的格式，并根据检查后的sql语句返回相应的字段信息
        :param sql: 检索KBase数据库的sql语句
        :type sql: unicode string
        :return: 以元组的形式返回多个值，包括检查后的sql语句和对应的存放字段序号的列表
        :rtype: tuple（unicode string， list）
        """
        if sql:
            col_list = []
            start = sql.find(u'select')
            end = sql.find(u'from')
            search_field = sql[start+7:end]
            if search_field != '* ':
                while search_field:
                    pos = search_field.find(u',')
                    if pos != -1:
                        swap = search_field[:pos]
                        col_num = self.__column_num_index(swap)
                        col_list.append(col_num)
                        search_field = search_field[pos+1:]
                    else:
                        swap = search_field[:-1]
                        col_num = self.__column_num_index(swap)
                        col_list.append(col_num)
                        search_field = ''
                new_sql = u'SELECT * ' + sql[end:]
                return new_sql, col_list
            else:
                ret = list()
                for key in self._dict:
                    ret.append(key)
                for i in range(len(ret)):
                    col_num = self.__column_num_index(ret[i])
                    col_list.append(col_num)
                return sql, col_list
        else:
            print('FATAL ERROR: sql is None!')

    def __check_sql(self, sql):
        """
        重置sql语句，内部处理
        :param sql: 数据库查询sql语句
        :type sql: unicode string
        :return: 处理后的sql语句
        :rtype: unicode string
        """
        if sql:
            end = sql.find(u'from')
            new_sql = u'select * ' + sql[end:]
            return new_sql
        else:
            print('FATAL ERROR: sql is None!')
            exit(-1)

    def __check_field(self, sql, tpiclient):
        """
        检查索引的字段名称
        :param sql: 检索数据库的sql语句
        :type sql: string or sequence of string
        :param tpiclient: 传入游标后的对象
        :type tpiclient: KBase.TPIClient
        :return: 通过sql语句检查出来的字段名称
        :rtype: list
        """
        if sql:
            start = sql.find(u'select')
            end = sql.find(u'from')
            table_name = sql[end + 5:] if sql.find(u'where') == -1 else sql[end+5:sql.find(u'where')-1]
            search_field = sql[start+7:end-1].replace(u' ', u'')
            if search_field != u'*':
                str_ = search_field.split(u',')
                col_list = self.get_field_position(table_name, tpiclient, str_)
                return col_list
            else:
                field_list = tpiclient.getRecordSetFieldName()
                str_ = field_list[1].split(u',')[:-1]
                col_list = self.get_field_position(table_name, tpiclient, str_)
                return col_list
        else:
            print('FATAL ERROR: sql is None!')

    def database_handle(self, conn=None, sql=u"select * from CJFDTOTAL where SMARTS = '糖尿病' order by FFD", mode='once', top=None):
        """
        # mode=once or batch
        检索KBase数据库，通过KBase接口创建游标等操作，执行ksql语句，并将字段结果保存到对应list中,当产生异常时，程序退出运算
        :param conn: 连接kbase的连接对象
        :type conn: KBase.Object
        :param sql: 执行的sql语句，这里默认是SELECT * FROM CJFDTOTAL where SMARTS = '机器学习' order by FFD
        :type sql: unicode string
        :param mode: 当前检索的模式，检索结果集的命中单条记录或是多条记录，default=once
        :type mode: string or sequence of string
        :param top: 取检索集中的前top个结果，default=None
        :type top: int
        :return: 保存索引字段信息的结果集列表
        :rtype: list
        注：默认服务器地址为192.168.106.101（可修改）
        特殊说明：
        例如：SELECT * FROM CJFDTOTAL where 专题子栏目代码 = "B026_62" order by 出版日期
             SELECT 全文,篇名,作者,是否高被引 FROM CJFDTOTAL where 专题子栏目代码 = "B026_62" order by 出版日期
        调用示例：
        1.首先确定KBASEServer服务是启动下并正常运行的状态
        2.调用方法：
        ->>>result = IOTools.database_handle()----默认状态
        ->>>result = IOTools.database_handle(host=u'', sql=u'', top=int)----非默认状态下，需手动赋值参数信息
        3.返回值说明：type(result)=list,返回值包含检索的字段信息以list形式保存
        4.抛出异常机制：当键入的host地址为非法地址时，系统报错，程序退出运算
        """
        retrieval_list = []
        switch = False
        try:
            # 将sql语句统一格式，全部转换成小写表示在整个代码逻辑里实现
            sql = sql.lower()
            swap_sql = self.__check_sql(sql)
            if conn:
                tpiclient, cur = conn[1], conn[0]
            else:
                conn = KBase.connect(host=self.host, port=4567, user=u'DBOWN', passwd=u'')
                tpiclient = KBase.TPIClient(conn)
                cur = conn.cursor()
                switch = True
            cur.execute(swap_sql)
            tpiclient.setCursor(cur)
            # 获取检索记录集的命中数目，对抽取个数和记录集个数添加判断
            # 如果记录集个数为0，直接返回空即可，非0再继续判断是多数据集还是单数据集
            cnt = tpiclient.getRecordSetCount()
            if sql.find(u'count(*)') != -1:
                return [cnt]
            if cnt[0] != 0:
                col_list = self.__check_field(sql, tpiclient)
                if mode == 'once':
                    self.__database_retrieval_results(tpiclient, retrieval_list, 'once',
                                                      [col_list[i] for i in range(len(col_list))])
                    if switch:
                        conn.close()
                        cur.close()
                    return retrieval_list
                else:
                    if top and top <= cnt[0]:
                        for _ in range(top):
                            self.__database_retrieval_results(tpiclient, retrieval_list, 'batch',
                                                              [col_list[i] for i in range(len(col_list))])
                    else:
                        for _ in range(cnt[0]):
                            self.__database_retrieval_results(tpiclient, retrieval_list, 'batch',
                                                              [col_list[i] for i in range(len(col_list))])
                    if switch:
                        conn.close()
                        cur.close()
                    return retrieval_list
            # 记录集无检索结果，直接返回None
            else:
                if switch:
                    cur.close()
                    conn.close()
                return []
        except SystemError as e:
            print('except:', e)
            print('FATAL ERROR: KBASEServer is not connected, please check host!!')
            exit(-1)

    def datbase_cite_count(self, sql=u"select count(*) from REFTOTAL where 参考文献 = 'QCSY201124297'"):
        """
        通过KBase查询参考文献的个数
        :param sql: sql语句
        :type sql: string or sequence of string
        :return: 参考文献个数
        :rtype: int
        """
        if sql:
            from_pos = sql.find(u'from')
            where_pos = sql.find(u'where')
            view_name = sql[from_pos+5:where_pos-1]

            conn = KBase.connect(host=self.host, port=4567, user=u'DBOWN', passwd=u'')
            tpiclient = KBase.TPIClient(conn)
            cur = conn.cursor()
            cur.execute(sql)
            tpiclient.setCursor(cur)
            try:
                ret = tpiclient.viewNameExists(view_name)
                if ret[0] == 1:
                    _, record_count = tpiclient.getBLob(0, 0, 65535)
                    conn.close()
                    return record_count
            except SystemError:
                print('view is not exit, please check ~~')

    def choice_new(self, data):
        """
        日期排序（通用）
        :param data: 相关排序日期序列
        :type data: list
        :return: 排序需求后的日期
        :rtype: string or sequence of string
        """
        result = []
        d_map = {}
        for d in data:
            d_map.setdefault(d[:], []).append(d)
        d_list = d_map.keys()
        d_list.sort()
        for k in d_list:
            result.append(min(d_map[k]))
        return result[0]


class PooledKBase(object):
    """
    KBase数据库连接池操作
    当建立了连接池后，连接池中准备好maxconnections个链接的游标，可通过任一可用游标进行数据库检索查询
    使用连接池的示例：
    import kbase
    pk = kbase.PooledKBase()   # 抛出kbase.py文件，然后实例化一个PooledKBase类的对象
    thread_cur = pk.get_connection()   # 从已经准备好的连接池中获取一个可用的连接
    ret = pk.execute_ksql(ksql,thread_cur,mode='all')   # 执行kbase数据库检索，并返回相应结果集，结果集返回[(res, errorcode)]
    pk.close_connection()    # 当所有查询任务执行完成之后，统一关闭连接，并将连接池清空，待下次使用
    """
    version = __version__

    def __init__(self, max_connections_cnt=500, host=u'192.168.100.113', mode=None):
        """
        创建数据库连接已经产生对应max_connections_cnt个连接数的连接池
        :param max_connections_cnt: kbase连接池接受的最大连接数，default=500
        :type max_connections_cnt: int<=500
        :param host: kbase服务器ip地址
        :type host: string or sequence of string
        """
        from queue import Queue
        if max_connections_cnt > 10000:  # KBase注册后允许链接数最大值=10000，超过后重新赋值
            max_connections_cnt = 9999
        self._maxsize = max_connections_cnt
        self._pool = Queue(maxsize=max_connections_cnt)  # 利用队列解决，方便日后添加缓存过期机制
        self._using = list()  # 将每次使用过的连接放入_using中记录
        self.host = host
        try:
            self.conn = self.create_connection_pool(host)
            if mode == 'cur':
                for _ in range(max_connections_cnt):
                    self.fill_connection(self.conn, mode='cur')
            elif mode == 'tpi':
                for _ in range(max_connections_cnt):
                    self.fill_connection(self.conn)
            else:
                raise NoRelativeMethodError()
        except Exception as e:
            raise e

    def fill_connection(self, connection, mode='tpi'):
        """
        将数据库连接池填充已经建立好的待使用的数据库游标连接
        :param connection: 数据库连接
        :type connection: KBase.Connection
        :param mode: 创建数据库连接的方式
        :type mode: string or sequence of string
        :return: 数据库连接
        :rtype: KBase.Connection.Object or KBase.Cursor.Object
        """
        try:
            cur = connection.cursor()
            if mode == 'cur':
                self._pool.put(cur)
            elif mode == 'tpi':
                tpiclient = KBase.TPIClient(connection)
                combine = (cur, tpiclient)
                self._pool.put(combine)
            else:
                exit(-1)
        except Exception as e:
            raise "fillConnection error:" + str(e)

    def cursor_execute_ksql(self, sql, thread_cur, mode='all'):
        """
        利用已创建好的连接池执行ksql语句
        :param sql: ksql执行语句
        :type sql: string or sequence of string
        :param thread_cur: 已创建好的连接池中获取的一个数据库连接
        :type thread_cur: Kbase.object
        :param mode: 结果集获取类型，default=‘all’
        :type mode: string or sequence of string
        :return: 数据库检索后的结果集
        :rtype: list
        """
        try:
            try:
                thread_cur.execute(sql)
            except SystemError:
                pass
            if mode == 'all':
                ret = thread_cur.fetchall()
            else:
                ret = thread_cur.fetchmany(size=1)
            return ret
        except StopIteration:
            pass
        finally:
            self.release_connection(thread_cur)
        # finally:
        #     self._pool.put(thread_cur)

    def tpiclient_execute_sql(self, sql, thread_cur, mode='once', top=None):
        """
        利用原生接口调用先前的使用kbase的代码，获取记录集数据
        :param sql: ksql执行语句
        :type sql: string or sequence of string
        :param thread_cur: 已创建好的连接池中获取的一个数据库连接
        :type thread_cur: Kbase.object
        :param mode: 结果集获取类型，default=‘once’
        :type mode: string or sequence of string
        :param top: 取数据集中前top个数据，默认=None即全部结果集全都取出
        :type top: int
        :return: 数据库检索后的结果集
        :rtype: list
        """
        try:
            db = DataBase()
            ret = db.database_handle(conn=thread_cur, sql=sql, mode=mode, top=top)
            return ret
        except StopIteration:
            pass
        finally:
            self.release_connection(thread_cur)
        # finally:
        #     self._pool.put(thread_cur)

    def getConnection(self):
        """
        获取一个数据库的游标连接
        :return: 获取数据库连接池中的一个完好可用的数据库连接游标
        :rtype: KBase.Cursor
        """
        try:
            if not self._pool.empty():
                return self._pool.get()
        except Exception as e:
            raise "getConnection error:" + str(e)

    def close_connection(self):
        """
        数据库执行完所有检索等任务后，关闭数据库的连接
        """
        try:
            self.conn.close()
            self._pool = None
        except Exception:
            raise NotClosedError()

    def create_connection_pool(self, host):
        """
        创建数据库连接池，创建一个数据库链接，通过该链接生成max_connections_cnt个数据库游标，然后统一放入链接池中待使用
        :param host: default=192.168.106.101 需要连接的服务器ip地址，缺省值是101服务器
        :type host: string or sequence of string
        :return: 数据库连接conn
        :rtype: KBase_Conn
        """
        try:
            kbase_conn = KBase.connect(host=host, port=4567, user=u'DBOWN', passwd=u'')
            return kbase_conn
        except Exception:
            raise NotConnectionError("Database module is not connection, please check host and try again!!")

    def acquire_connection(self, apply=5, applied=0):
        """
        获取一个KBase链接池中的游标连接
        :param apply: 最大尝试申请连接的个数
        :type apply: int
        :param applied: 已经尝试申请的个数
        :type applied: int
        :return: 获取一个KBase连接池中可用的游标连接
        :rtype: KBase.Cursor
        """
        if not self._pool.empty():
            conn = self._pool.get()
            self._using.append(conn)
            return conn
        else:
            import gevent
            # 如果两个池子里的链接个数小于允许链接最大限额，继续创建链接
            if self._pool.qsize() + len(self._using) < self._maxsize:
                self.fill_connection(self.create_connection_pool(self.host), mode='cur')   # 重新申请一个链接并put进入_pool池中
                return self.acquire_connection()
            else:
                if applied >= apply:  # 如果待申请链接数超过允许最多待申请数，直接返回连接池耗尽的错误提示
                    raise PoolExhaustedError()
                applied += 1
                gevent.sleep(0.1)  # 没有超出最大允许个数，每次没申请到便停0.1s后在提交申请
                return self.acquire_connection(apply=apply, applied=applied)

    def release_connection(self, conn):
        """
        释放数据库游标连接操作
        :param conn: 数据库游标连接
        :type conn: KBase.Cursor
        """
        if conn in self._using:
            self._using.remove(conn)
            self._pool.put(conn)
        else:
            raise ConnectionNotFoundError()

    def get_connection(self):
        """
        获取一个数据库游标连接，使用完后释放连接，即丢回连接池中
        :return: 数据库连接游标
        :rtype: KBase.Cursor
        """
        conn = self.acquire_connection()
        # try:
        return conn
        # finally:
        #     self.release_connection(conn)


if __name__ == "__main__":

    """for kbase test"""
    # import uniout, time

    # """测试ksql语句"""
    # ksql = u'select * from SYS_AUTHOR_INSTITUTION_NAME where CODE=00000014 09540005'
    # ksql1 = u'select 2000_1_被引均值, 2000_1_高被引量, 2000_2_被引均值, 2000_2_高被引量, 2000_被引文章数 from SYS_AUTHOR_PAPER_CITED where 作者代码=00000014'
    # ksql2 = u'select * from SYS_AUTHOR_PAPER_CITED where 作者代码=00000267'
    # ksql3 = u'select 篇名,作者 from CJFDTOTAL where 文件名=ZSHB200609018'
    # ksql3_ = u'SELECT 篇名, 作者 FROM CJFDTOTAL WHERE 文件名=ZSHB200609018'
    # ksql4 = u'select 被引来源代码 from REFSTATTOTAL where 被引作者代码=00000014'
    # ksql5 = u'select 年,作者代码 from CJFDTOTAL where 作者代码=22644365'
    # ksql6 = u'select 出版日期,机构 from CJFDTOTAL where 文件名 =XTYD201105044'
    ksql7 = u'select 篇名,中文关键词,年 from CJFDTOTAL WHERE 文件名=SCXH200806014'
    ksql8 = u'select 篇名,作者 from CJFDTOTAL where 主题 like "多模多馈" order by 出版日期'
    # ksql9 = u'select count(*) from CJFDTOTAL where smarts="信息化"'
    # ksql9_ = u'select COUNT(*) from CJFDTOTAL where smarts="安防"'
    # ksqlll = u"SELECT TOP 3000 篇名,关键词,文件名 FROM CJFD2006,CJFD2005 WHERE (主题 %='虚拟实现技术 虚拟软件' OR 题名 %='虚拟实现技术 虚拟软件') AND 出版日期>='2002-09-25' AND 出版日期<'2006-09-25' ORDER BY (ffd,'rank') DESC with set=locked"
    ksl = u"SELECT count(*) FROM CJFDTOTAL WHERE (主题 %='虚拟现实' OR 题名 %='虚拟现实') AND 出版日期 between('2000-08-15', '2013-08-14') ORDER BY (ffd, 'rank') DESC with set=locked"
    ksql = u'select 2012_6_FN, 2012_6_HDN from SYS_AUTHOR_FN_HDN where 作者代码=17753280'
    kkk = u'SELECT 全文,中文摘要,文件名,篇名 FROM CJFDTOTAL WHERE (主题 %="矿物——光谱" OR 题名 %="矿物——光谱") AND 出版日期 BETWEEN ("1980-08-15", "2013-08-14") ORDER BY (ffd,"rank") DESC with set=locked'
    sss = u'select 篇名,中文关键词,出版日期,全文,中文摘要 from CJFDTOTAL WHERE 文件名=BJSY200904003'
    """利用连接池操作的cur模式检索数据库"""
    pk = PooledKBase(max_connections_cnt=50, mode='cur')
    thread_cur = pk.get_connection()
    ret = pk.cursor_execute_ksql(sss, thread_cur, mode='all')
    print(ret)
    pk.close_connection()

    """利用连接池操作的tpi模式检索数据库"""
    pk = PooledKBase(max_connections_cnt=50, mode='tpi')
    thread_cur = pk.get_connection()
    ret = pk.tpiclient_execute_sql(sss, thread_cur, mode='batch', top=3000)
    print(ret)
    pk.close_connection()

    # """不利用连接池tpi模式检索数据库"""
    # db = DataBase()
    # ret = db.database_handle(sql=ksqlll, mode='batch')
    # print ret

    # ksql = u'select 2012_6_FN, 2012_6_HDN from SYS_AUTHOR_FN_HDN where 作者代码=17753280'
    # ksql_ = u'select 作者代码, 2012_6_FN, 2012_6_HDN from SYS_AUTHOR_FN_HDN where 作者代码=17753280'
    # _ksql = u'select 2012_6_FN, 作者代码, 作者代码, 2012_6_HDN from SYS_AUTHOR_FN_HDN where 作者代码=17753280'
    # conn = KBase.connect(host=u'192.168.100.113', port=4567, user=u'DBOWN', passwd=u'')
    # cur = conn.cursor()
    # cur.execute(ksl)
    # ret = cur.fetchall()
    # print ret
    # cur.close()
    # conn.close()
