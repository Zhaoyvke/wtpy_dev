import json
import pymysql
from wtpy import BaseIndexWriter

class MySQLIdxWriter(BaseIndexWriter):
    '''
    Mysql指标数据写入器
    '''

    def __init__(self, host, port, user, pwd, dbname, sqlfmt):
        self.__db_conn__ = pymysql.connect(host=host, user=user, passwd=pwd, db=dbname, port=port)
        self.__sql_fmt__ = sqlfmt

    def write_indicator(self, id, tag, time, data):
        sql = self.__sql_fmt__.replace("$ID", id).replace("$TAG", tag).replace("$TIME", str(time)).replace("$DATA", json.dumps(data))
        curConn = self.__db_conn__
        curBase = curConn.cursor()
        curBase.execute(sql)
        curConn.commit()

class MySQLTradeWriter(BaseIndexWriter):
    '''
    Mysql写入，模拟的交易记录
    '''
    def __init__(self, host, port, user, pwd, dbname, sqlfmt):
        self.host = host
        self.port = port
        self.user = user
        self.pwd = pwd
        self.dbname = dbname
        self.__sql_fmt__ = sqlfmt

    def write_trade(self, strategy, time, data):
        # 需要确认一下用什么形式。
        __db_conn__ = pymysql.connect(host=self.host, user=self.user, passwd=self.pwd, db=self.dbname, port=self.port)
        if len(data) > 0:
            for k, v in data.items():
                sql = self.__sql_fmt__.replace("$STRATEGY", strategy).replace("$TIME", time).replace("$CODE", k).replace("$QTY", str(v))
                curConn = __db_conn__
                curBase = curConn.cursor()
                curBase.execute(sql)
                curConn.commit()

            curConn.close()

        return

class MySQLPositionWriter(BaseIndexWriter):
    '''
    Mysql写入，模拟的交易记录
    '''
    def __init__(self, host, port, user, pwd, dbname, sqlfmt):
        self.host = host
        self.port = port
        self.user = user
        self.pwd = pwd
        self.dbname = dbname
        self.__sql_fmt__ = sqlfmt

    def write_pos(self, strategy, date, data):
        # 需要确认一下用什么形式。
        __db_conn__ = pymysql.connect(host=self.host, user=self.user, passwd=self.pwd, db=self.dbname, port=self.port)
        if len(data) > 0:
            for k, v in data.items():
                sql = self.__sql_fmt__.replace("$STRATEGY", strategy).replace("$DATE", str(date)).replace("$CODE", k).replace("$QTY", str(v))
                curConn = __db_conn__
                curBase = curConn.cursor()
                curBase.execute(sql)
                curConn.commit()
            curConn.close()

        return

