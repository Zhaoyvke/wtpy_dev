from wtpy.wrapper import WtDataHelper
from wtpy.WtCoreDefs import WTSBarStruct
from ctypes import POINTER
import pandas as pd
import numpy as np
import os

dtHelper = WtDataHelper()

def parquet_to_dsb(parfile:str, path:str, period:str = "m1"):
    '''
    将parquet文件转换为dsb文件
    parfile: parquet文件路径
    path: dsb文件存储路径，这个一般传入datakit目录结构的子目录，如period为m1，则path为storage/min1/
    period: 周期
    '''
    df = pd.read_parquet(parfile)
    df = df.rename(columns={
        'total_turnover':'money',
        'volume':'vol'
        })
    df.reset_index(inplace=True)
    tickers = df['order_book_id'].unique()
    tickers.sort()
    for ticker in tickers:
        df_ticker = df[df['order_book_id']==ticker]
        df_ticker['datetime'] = df_ticker['datetime'].apply(lambda x: int(x.strftime('%Y%m%d%H%M')))
        df_ticker['date'] = np.floor(df_ticker['datetime']/10000).astype('int64')
        df_ticker['time'] = (df_ticker['date']-19900000)*10000 + df_ticker['datetime']%10000
        df_ticker.drop(columns=['order_book_id','datetime'], inplace=True)

        BUFFER = WTSBarStruct*len(df_ticker)
        buffer = BUFFER()

        def assign(procession, buffer):
            tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))

        df_ticker.apply(assign, buffer=buffer)

        items = ticker.split(".")
        exchg = 'SSE' if items[1]=='XSHE' else 'SZSE'
        code = items[0]
        filepath = os.path.join(path, exchg)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, f"{code}.dsb")
        dtHelper.store_bars(barFile=filepath, firstBar=buffer, count=len(df_ticker), period=period)

parquet_to_dsb("D:/Conemu/wtpy-master/demos/cta_bond_bt/000832XSHG.parq", "./", "m1")