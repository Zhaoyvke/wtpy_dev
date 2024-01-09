from wtpy.wrapper import WtDataHelper
from wtpy.WtCoreDefs import WTSBarStruct, WTSTickStruct
from ctypes import POINTER
from wtpy.SessionMgr import SessionMgr
import pandas as pd
import pyarrow.parquet as pq

dtHelper = WtDataHelper()

def test_store_bar():
    parquet_file = pq.ParquetFile("D:/Conemu/wtpy-master/demos/cta_bond_bt/convertible_1min.parquet")
    # 获取parquet文件的总行数
    total_rows = parquet_file.metadata.num_rows
    batch_size = 10000
        # 分批读取数据
    for i in range(0, total_rows, batch_size):
        start_row = i
        end_row = min(i + batch_size, total_rows)
        table = parquet_file.read_row_group(i)
        df = table.to_pandas().iloc[start_row:end_row]
        #table = parquet_file.read_row_group(0, columns=columns[start_row:end_row])
        #df = table.to_pandas()
        BUFFER = WTSBarStruct*len(df)
        buffer = BUFFER()
        loc = 0
        for idx in df.index:
            curBar = buffer[loc]
            #curBar.code = idx[0]
            curBar.date = int(idx[1].strftime('%Y%m%d'))
            curBar.time= (curBar.date-19900000)*10000+int(idx[1].strftime('%H%M'))
            curBar.open = df.loc[idx, 'open']
            curBar.low = df.loc[idx,'low']
            curBar.money = df.loc[idx,'total_turnover']
            curBar.volume = df.loc[idx,'close']
            curBar.high = df.loc[idx,'high']
            loc += 1
            print(loc)
            dtHelper.store_bars(barFile="../storage/his/min5/SSE/convertible_1min.dsb", firstBar=buffer, count=len(df), period="m5")


def test_store_bars():
    parquet_file = pq.ParquetFile("D:/Conemu/wtpy-master/demos/cta_bond_bt/convertible_1min.parquet")
    num_row_groups = parquet_file.num_row_groups
    for i in range(num_row_groups):
        table = parquet_file.read_row_group(i)
        df = table.to_pandas()
        BUFFER = WTSBarStruct*len(df)
        buffer = BUFFER()
        loc = 0

        for idx in df.index:
            curBar = buffer[loc]
            curBar.date = int(idx[1].strftime('%Y%m%d'))
            curBar.time= (curBar.date-19900000)*10000+int(idx[1].strftime('%H%M'))
            curBar.open = df.loc[idx, 'open']
            curBar.low = df.loc[idx,'low']
            curBar.money = df.loc[idx,'total_turnover']
            #print(curBar.money)
            curBar.volume = df.loc[idx,'volume']
            curBar.close = df.loc[idx,'close']
            curBar.high = df.loc[idx,'high']
            #print(curBar.high)
            loc += 1    
              
        dtHelper.store_bars(barFile="./convertible_1min.dsb", firstBar=buffer, count=len(df), period="m1")

def test_store_ticks():

    df = pd.read_csv('./storage/csv/rb主力连续_20201030.csv')
    BUFFER = WTSTickStruct*len(df)
    buffer = BUFFER()

    tags = ["一","二","三","四","五"]

    for i in range(len(df)):
        curTick = buffer[i]

        curTick.exchg = b"SHFE"
        curTick.code = b"SHFE.rb.HOT"

        curTick.price = float(df[i]["最新价"])
        curTick.open = float(df[i]["今开盘"])
        curTick.high = float(df[i]["最高价"])
        curTick.low = float(df[i]["最低价"])
        curTick.settle = float(df[i]["本次结算价"])
        
        curTick.total_volume = float(df[i]["数量"])
        curTick.total_turnover = float(df[i]["成交额"])
        curTick.open_interest = float(df[i]["持仓量"])

        curTick.trading_date = int(df[i]["交易日"])
        curTick.action_date = int(df[i]["业务日期"])
        curTick.action_time = int(df[i]["最后修改时间"].replace(":",""))*1000 + int(df[i]["最后修改毫秒"])

        curTick.pre_close = float(df[i]["昨收盘"])
        curTick.pre_settle = float(df[i]["上次结算价"])
        curTick.pre_interest = float(df[i]["昨持仓量"])

        for x in range(5):
            setattr(curTick, f"bid_price_{x}", float(df[i]["申买价"+tags[x]]))
            setattr(curTick, f"bid_qty_{x}", float(df[i]["申买量"+tags[x]]))
            setattr(curTick, f"ask_price_{x}", float(df[i]["申卖价"+tags[x]]))
            setattr(curTick, f"ask_qty_{x}", float(df[i]["申卖量"+tags[x]]))

    dtHelper.store_ticks(tickFile="./SHFE.rb.HOT_ticks.dsb", firstTick=buffer, count=len(df))

def test_resample():
    # 测试重采样
    sessMgr = SessionMgr()
    sessMgr.load("sessions.json")
    sInfo = sessMgr.getSession("SD0930")
    ret = dtHelper.resample_bars("IC2009.dsb",'m1',5,202001010931,202009181500,sInfo)
    print(ret)

#df = dtHelper.read_dsb_bars(r"E:\\wtpy_master\\wtpy-master\\159845.dsb").to_df()
#df = dtHelper.read_dmb_bars(r"E:\wtpy_master\wtpy-master\demos\datakit_stk\STK_Data\rt\min1\SSE\600000.dmb").to_df()

#df=df.loc[df['date']=='20231109']
#df=df.query('date == "20231109"')
