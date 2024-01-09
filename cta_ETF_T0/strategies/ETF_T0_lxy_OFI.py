# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:52:40 2023

@author: lxy
"""

from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import pandas as pd
import polars as pl
import scipy
import pickle
import lightgbm
import joblib
import datetime
import os

# 一些函数
def next_minute(minute):
    # minute -> 1122
    minute = int(minute)
    h, m = minute//100, minute%100
    nm = (m+1)%60+ 100*((m+1)//60+h)
    if nm < 930:
        nm = 930
    elif nm > 1130 and nm <= 1300:
        nm = 1301
    elif nm > 1500:
        nm = 1500
    return nm

def minutes_list():
    hours = [9,10,11,13,14]
    minutes = range(60)
    min_list = []
    for hour in hours:
        for minute in minutes:
            if hour == 9 and minute <= 30:
                continue
            if hour == 11 and minute > 30:
                continue
            if hour ==13 and minute == 0:
                continue
            if hour == 14 and minute > 60:
                continue
            min_list.append(hour*100+minute)
    min_list.append(1500)
    return min_list

def minute_to_index():
    dic_index = dict(zip(minutes_list(),range(len(minutes_list()))))
    return dic_index

def index_to_minute():
    dic_index = dict(zip(range(len(minutes_list())),minutes_list()))
    return dic_index

def calc_wap1(df):
    wap = (df['BidPx1'] * df['OfferSize1'] + df['OfferPx1'] * df['BidSize1']) / (df['BidSize1'] + df['OfferSize1'])
    return wap

def log_return(series):
    return np.log(series).diff().sum()

# Calculate the realized volatility
def realized_volatility(series,l):
    return np.sqrt((series**2).sum()*4800/l*250)

def find_minute_Y(df,begin_minute,end_minute):
    part = df.filter((df['minute']>=begin_minute) & (df['minute']<=end_minute))
    Y = part.filter(~part['log_return'].is_nan())['log_return']
    if len(Y)>0:
        return Y
    else:
        return []
    
def tendency(price, vol, minute):    
    if len(price)<=1:
        return 0
    else:
        df_diff = np.diff(price)        
        val = (df_diff/price[1:])*100
        power = (val*vol[1:]).sum()
    return(power)

def sigma(ss,aa,l,dt=3/14400):
    return scipy.linalg.toeplitz([ss*dt+2*aa,-aa]+[0]*(l-2))

def lmbd(l):
    return scipy.linalg.toeplitz([2,-1]+[0]*(l-2))

def W(ss,aa,l,dt):
    S = sigma(ss,aa,l,dt)
    L = lmbd(l)
    L2 = L.dot(L)
    Sinv = scipy.linalg.inv(S)
    Sinv2 = Sinv.dot(Sinv)
    top1 = l*np.trace(Sinv2.dot(L))*Sinv.dot(L.dot(Sinv)) - l*np.trace(Sinv2.dot(L2))*Sinv2
    top2 = np.trace(Sinv2.dot(L))*Sinv2 - np.trace(Sinv2)*Sinv.dot(L.dot(Sinv))
    bot = np.power(np.trace(Sinv2.dot(L)),2) - np.trace(Sinv2)*np.trace(Sinv2.dot(L2))
    w1 = top1/bot
    w2 = top2/bot
    return w1,w2

def get_snapshot_sz(df):
    t0 = datetime.datetime.now()
    # level1 对应可用字段
    # 代码（六位） 交易日 日期 时间（93020223）
    # 价格 开盘价 最高价 最低价 昨收价 总成交量 成交量 总成交额 成交额 总持 增仓
    # 买一-五价 买一-五量 卖一-五价 卖一-五量 计数
    # print(f'\r {fn}',end='')
    # fn: filename, (full path)
    # load parquet, single ticker
    # level = pl.read_parquet(level_path + year + '/' + date + '/' + fn)
    # snap = pl.read_parquet(snap_path + year + '/' + date + '/' + fn)

    # level_columns = level.columns
    # snap_columns = snap.columns
    # common_columns = np.intersect1d(level_columns, snap_columns)

    # # 合并
    # df = level.join(snap[['OrigTime']+list(np.setdiff1d(snap_columns, common_columns))],on='OrigTime')
    df = df.sort(by='OrigTime')
    
    ticker = df['ticker'][0][-6:]
    df = df.with_columns(pl.lit(ticker).alias('ticker'))
    
    df = df.with_columns(df['OrigTime'].cast(pl.Utf8).str.strptime(pl.Datetime,format='%Y%m%d%H%M%S%f').alias('datetime'))
    df = df.with_columns((df['OrigTime']//100000%10000).alias('minute'))
    df = df.with_columns(df['minute'].apply(next_minute))
    # df = df.filter((df['minute']>930) & (df['minute']<1458))
    
    # 如果不成交的话 快照的ohlc就是0 不会更新
    ohlc_cols = ['OpenPx','HighPx','LowPx','LastPx']    
    for name in ohlc_cols:
        # exec(f'df = df.with_columns(pl.when(pl.col("NumTrades")==0).then({None}).otherwise(pl.col("{name}")).alias("{name}"))')
        df = df.with_columns(pl.when(pl.col("TotalVolumeTrade")==0).then(None).otherwise(pl.col(f"{name}")).alias(f"{name}"))
    df = df.fill_null(strategy='forward')
    df = df.fill_null(strategy='backward')
    
    # level1最高价和最低价的数值不准 重新算 但是如果按照lastpx去计算的话 有可能会丢失掉这个3s内的最高价之类的 但是鉴于level1那个实在错的离谱 还是按照这种方式处理
    # df = df.with_columns(pl.col('LastPx').cummax().alias('HighPx'))
    # df = df.with_columns(pl.col('LastPx').cummin().alias('LowPx'))
    
    # 涨跌停处理
    df = df.with_columns(pl.when(pl.col('BidPx1')==0).then(pl.col('OfferPx1')).otherwise(pl.col('BidPx1')).alias('BidPx1'))
    df = df.with_columns(pl.when(pl.col('OfferPx1')==0).then(pl.col('BidPx1')).otherwise(pl.col('OfferPx1')).alias('OfferPx1'))

    bid_px_cols = ['BidPx%s'%(s) for s in range(1,6)]
    offer_px_cols = ['OfferPx%s'%(s) for s in range(1,6)]
    bid_size_cols = ['BidSize%s'%(s) for s in range(1,6)]
    offer_size_cols = ['OfferSize%s'%(s) for s in range(1,6)]
    
    # Calculate Wap
    df = df.with_columns(calc_wap1(df).alias('wap1')) #只选一个

    # Calculate log returns
    df = df.with_columns(np.log(pl.col('wap1')).diff().alias('log_return')) 

    # df = df.fill_nan(0)
    df = df.fill_null(0)

    # Calculate spread
    df = df.with_columns(((df['OfferPx1'] - df['BidPx1']) / ((df['OfferPx1'] + df['BidPx1']) / 2)).alias('price_spread'))
    df = df.with_columns(((df['OfferPx2'] - df['BidPx2']) / ((df['OfferPx2'] + df['BidPx2']) / 2)).alias('price_spread2'))
    df = df.with_columns((df['BidPx1'] - df['BidPx2']).alias('bid_spread'))
    df = df.with_columns((df['OfferPx1'] - df['OfferPx2']).alias('ask_spread'))
    df = df.with_columns((abs(df['bid_spread'] - df['ask_spread'])).alias("bid_ask_spread"))

    # 添加其他基本特征
    # total bid/ask quantity, 需要补零吗(0一般是涨跌停的情况)
    total_bid_qty = np.array(df[bid_size_cols].sum(axis=1)).astype(float)
    total_ask_qty = np.array(df[offer_size_cols].sum(axis=1)).astype(float)
    # total_bid_qty[total_bid_qty==0] = np.nan
    # total_ask_qty[total_ask_qty==0] = np.nan
    df = df.with_columns(pl.Series(total_bid_qty).alias('total_bid_qty'))
    df = df.with_columns(pl.Series(total_ask_qty).alias('total_ask_qty'))
    df = df.with_columns((pl.col('total_ask_qty')-pl.col('total_bid_qty')).alias('total_qty_diff'))
    # weighted quote price
    df = df.with_columns(((df[bid_size_cols]*df[bid_px_cols]).sum(axis=1)/df['total_bid_qty']).alias('weighted_bid_prc'))
    df = df.with_columns(((df[offer_size_cols]*df[offer_px_cols]).sum(axis=1)/df['total_ask_qty']).alias('weighted_ask_prc'))
    
    df = df.with_columns((df['total_ask_qty'] + df['total_bid_qty']).alias('total_volume'))   #考虑改10档
    df = df.with_columns((abs(df['total_ask_qty'] - df['total_bid_qty'])).alias('volume_imbalance'))   #考虑改10档
    
    
    # 时间是精确到秒的，直接去groupby就行了
    df_min = df.groupby('minute').last().sort('minute')
    
    df_min_return = df.groupby('minute').agg(
        pl.col('log_return').sum()
        ).sort('minute')
    df_min = pl.concat([df_min.drop(['log_return']),df_min_return.drop('minute')],how='horizontal')
    
    df_min = df_min.with_columns((df_min['LastPx']-df_min['PreClosePx']).alias('PxChange1'))
    df_min = df_min.with_columns(pl.when(pl.col('LastPx').diff().is_null()).then(0).otherwise(pl.col('LastPx').diff()).alias('PxChange2'))
    # df_min = df_min.with_columns(((pl.col('LastPx')-pl.col('RealTimeNAV'))/pl.col('RealTimeNAV')*100).alias('ETF_discount'))
    
    df_min = df_min.with_columns(((pl.col('BidPx1')+pl.col('OfferPx1'))/2).alias('mid'))
    df_feature = df.groupby('minute').agg(
                    pl.col('LastPx').first().alias('open'),
                    pl.col('LastPx').max().alias('high'),
                    pl.col('LastPx').min().alias('low'),
                    pl.col('LastPx').last().alias('close'),
                    ).sort('minute')
    df_min = pl.concat([df_min,df_feature.drop('minute')],how='horizontal')
    # volume 直接取最后一行的话 其实拿的是最后一个tick的volume 而不是这一分钟的 所以还是用总的进行计算比较对
    df_min = df_min.with_columns(pl.col('TotalVolumeTrade').diff().alias('volume')) 
    df_min = df_min.with_columns(pl.when(pl.col('volume').is_null()).then(pl.col('TotalVolumeTrade')).otherwise(pl.col('volume')).alias('volume'))
    df_min = df_min.with_columns(pl.col('TotalValueTrade').diff().alias('amount'))
    df_min = df_min.with_columns(pl.when(pl.col('amount').is_null()).then(pl.col('TotalValueTrade')).otherwise(pl.col('amount')).alias('amount'))
    
    df_min = df_min.with_columns((pl.col('amount')/pl.col('volume')).alias('vwap'))
    df_min = df_min.with_columns(pl.when(pl.col('vwap').is_nan()).then(pl.col('LastPx')).otherwise(pl.col('vwap')).alias('vwap'))
    
    df_min = df_min.fill_null(0)
    df_min = df_min.fill_nan(0)

    lis = []
    # time.sleep(0.2)
    for minute in df_min['minute']:
        # print(minute)
        df_id = df.filter(pl.col('minute') == minute)  
        if len(df_id)>0:
            tendencyV = tendency(df_id['LastPx'], df_id['TotalVolumeTrade'], minute)
            # print(tendencyV)
            f_max = (df_id['LastPx'] > df_id['LastPx'].mean()).sum()
            f_min = (df_id['LastPx'] < df_id['LastPx'].mean()).sum()
            df_max_ = (df_id['LastPx'].diff() > 0).sum()
            df_min_ = (df_id['LastPx'].diff() < 0).sum()
            # new
            abs_diff = (df_id['LastPx'] - df_id['LastPx'].mean()).abs().median()       
            energy = (df_id['LastPx']**2).mean()
            iqr_p = np.percentile(df_id['LastPx'],75) - np.percentile(df_id['LastPx'],25)
            
            # vol vars
            abs_diff_v = (df_id['TotalVolumeTrade'] - df_id['TotalVolumeTrade'].mean()).abs().median()    
            energy_v = (df_id['TotalVolumeTrade']**2).sum()
            iqr_p_v = np.percentile(df_id['TotalVolumeTrade'],75) - np.percentile(df_id['TotalVolumeTrade'],25)
        else:
            tendencyV = 0.0
            f_max = 0
            f_min = 0
            df_max_ = 0
            df_min_ = 0
            # new
            abs_diff = 0.0     
            energy = 0.0
            iqr_p = 0.0
            
            # vol vars
            abs_diff_v = 0.0   
            energy_v = 0.0
            iqr_p_v = 0.0
        
        lis.append({'minute':minute,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max_,'df_min':df_min_,
                    'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
        # time.sleep(0.2)
    df_lr = pl.DataFrame(lis)
    df_min = df_min.join(df_lr,on='minute')
    
    
    for i in range(1,6):
        df_min = df_min.with_columns(pl.when(pl.col(f'BidPx{i}').diff()>0).then(pl.col(f'BidSize{i}')).when(pl.col(f'BidPx{i}').diff()==0).then(pl.col(f'BidSize{i}').diff()).otherwise(-pl.col(f'BidSize{i}')).alias(f'bOF{i}'))
        df_min = df_min.with_columns(pl.when(pl.col(f'OfferPx{i}').diff()>0).then(-pl.col(f'OfferSize{i}')).when(pl.col(f'OfferPx{i}').diff()==0).then(pl.col(f'OfferSize{i}').diff()).otherwise(pl.col(f'OfferSize{i}')).alias(f'aOF{i}'))
        df_min = df_min.with_columns((pl.col(f'bOF{i}')-pl.col(f'aOF{i}')).alias(f'OFI{i}'))
    
    
    # minutes = df_min['minute']
    # vol_QMLE=[]
    # vol = []
    # KK = 100
    # # vol要改成年化的 目前是5min的 *4800/l可以改成日频
    # for i in range(len(minutes)):
    #     # print(i)
    #     if i<4:
    #         Y = find_minute_Y(df,minutes[0],minutes[i])
    #     else:
    #         Y = find_minute_Y(df,minutes[i-4],minutes[i])
    #     l = len(Y)
    #     if l > 1:
    #         if i<4:
    #             dt = 1/(14400/(60/(l/(i+1))))
    #         else:
    #             dt = 1/(14400/(60/(l/5)))
    #         sss = [0]*KK
    #         aaa = [0]*KK
    #         y = -np.imag(np.fft.fft(np.array([0]+list(Y)+[0]*(l+1))))[1:l+1]
    #         s = np.array([2*np.cos(j*np.pi/(l+1),dtype=np.longdouble) for j in range(1,l+1)],dtype=np.longdouble)
    #         ss,aa = Y.std(),1e-3
            
    #         for j in range(1,KK):
    #             a = np.sum((2-s)/np.power(ss*dt+2*aa-aa*s,2))
    #             b = np.sum(np.power(2-s,2)/np.power(ss*dt+2*aa-aa*s,2))
    #             c = np.sum(np.power(ss*dt+2*aa-aa*s,-2))  # 是-2 所以没错
    #             ss = np.power(y,2).dot((l*a*(2-s)-l*b) / np.power(ss*dt+2*aa-aa*s,2)) / (a*a-b*c) *4800/l/((l+1)/2)
    #             aa = np.power(y,2).dot((a-c*(2-s)) / np.power(ss*dt+2*aa-aa*s,2)) / (a*a-b*c) /((l+1)/2)
    #             if np.abs(sss[j-1]-ss)<=(1e-15):
    #                 break
    #             sss[j] = ss
    #             aaa[j] = aa
    #         if (j == KK-1) | (ss < 0):
    #             sss = [0]*KK
    #             aaa = [0]*KK
    #             ss,aa = Y.std(),1e-3
    #             for j in range(1,KK):
    #                 # print(ss)
    #                 w1,w2 = W(ss,aa,l,dt)
    #                 ss = Y.dot(w1.dot(Y))*4800/l #真的很诡异 采用不同的单位 天或者min 收敛后的结果会相差很大
    #                 aa = Y.dot(w2.dot(Y))
    #                 if np.abs(sss[j-1]-ss)<=(1e-11):
    #                     break
    #                 sss[j] = ss
    #                 aaa[j] = aa
    #         if (j == KK-1) | (ss < 0):
    #             # vol_QMLE.append(np.sqrt(sum(Y**2))*np.sqrt(4800/l)*np.sqrt(250))  #年化的波动率
    #             vol_QMLE.append(realized_volatility(Y,l)) 
    #         else:
    #             vol_QMLE.append(np.sqrt(ss)*np.sqrt(250))  #年化的波动率
    #         vol.append(realized_volatility(Y,l))
    #     else:
    #         vol_QMLE.append(0)
    #         vol.append(0)
    minute_len = len(df['minute'].unique())
    vol_QMLE=[0]*(minute_len-1)
    vol = [0]*(minute_len-1)
    KK = 100
    # vol要改成年化的 目前是5min的 *4800/l可以改成日频
    Y = df.filter(~df['log_return'].is_nan())['log_return']
    l = len(Y)
    if l > 1:
        dt = 1/(14400/(60/(l/5)))
        sss = [0]*KK
        aaa = [0]*KK
        y = -np.imag(np.fft.fft(np.array([0]+list(Y)+[0]*(l+1))))[1:l+1]
        s = np.array([2*np.cos(j*np.pi/(l+1),dtype=np.longdouble) for j in range(1,l+1)],dtype=np.longdouble)
        ss,aa = Y.std(),1e-3
        
        for j in range(1,KK):
            a = np.sum((2-s)/np.power(ss*dt+2*aa-aa*s,2))
            b = np.sum(np.power(2-s,2)/np.power(ss*dt+2*aa-aa*s,2))
            c = np.sum(np.power(ss*dt+2*aa-aa*s,-2))  # 是-2 所以没错
            ss = np.power(y,2).dot((l*a*(2-s)-l*b) / np.power(ss*dt+2*aa-aa*s,2)) / (a*a-b*c) *4800/l/((l+1)/2)
            aa = np.power(y,2).dot((a-c*(2-s)) / np.power(ss*dt+2*aa-aa*s,2)) / (a*a-b*c) /((l+1)/2)
            if np.abs(sss[j-1]-ss)<=(1e-15):
                break
            sss[j] = ss
            aaa[j] = aa
        if (j == KK-1) | (ss < 0):
            sss = [0]*KK
            aaa = [0]*KK
            ss,aa = Y.std(),1e-3
            for j in range(1,KK):
                # print(ss)
                w1,w2 = W(ss,aa,l,dt)
                ss = Y.dot(w1.dot(Y))*4800/l #真的很诡异 采用不同的单位 天或者min 收敛后的结果会相差很大
                aa = Y.dot(w2.dot(Y))
                if np.abs(sss[j-1]-ss)<=(1e-11):
                    break
                sss[j] = ss
                aaa[j] = aa
        if (j == KK-1) | (ss < 0):
            # vol_QMLE.append(np.sqrt(sum(Y**2))*np.sqrt(4800/l)*np.sqrt(250))  #年化的波动率
            vol_QMLE.append(realized_volatility(Y,l)) 
        else:
            vol_QMLE.append(np.sqrt(ss)*np.sqrt(250))  #年化的波动率
        vol.append(realized_volatility(Y,l))
    else:
        vol_QMLE.append(0)
        vol.append(0)
    df_min = df_min.with_columns(pl.Series(np.array(vol_QMLE).astype(np.float64)).alias('volatility_QMLE'))
    df_min = df_min.with_columns(pl.Series(np.array(vol).astype(np.float64)).alias('volatility'))
    
    # df_min = df_min.with_columns((pl.col('PreClosePx')*1.1).alias('UpLimitPx'))
    # df_min = df_min.with_columns((pl.col('PreClosePx')*0.9).alias('DownLimitPx'))
    print(f"Computation costs: {datetime.datetime.now()-t0} secs!")
    return df_min
    

class ETF_T0_lxy_OFI(BaseCtaStrategy):
    
    def __init__(self, name:str, exchg:str, code:str, period:str):
        BaseCtaStrategy.__init__(self, name)
        self.__code__ = f"{exchg}.ETF.{code}"
        self.__c__ = code
        self.__period__ = period    # m1
        self.__trdUnit__ = 10000
        self.__path__ = 'F:/deploy/ETF_T0/model/'
        self.__time_period__ = 231*7
        self.load_models()

    def load_models(self):
        # 需要load models
        self.__rolling_model__ = joblib.load(f'{self.__path__}T0_level1_volume_rolling_pred_model.pkl')
        self.__fix_model__ = joblib.load(f'{self.__path__}T0_level1_volume_rolling_pred_model.pkl')
        self.__rolling_scaler__ = joblib.load(f'{self.__path__}T0_level1_volume_rolling_pred_scaler.pkl')
        self.__fix_scaler__ = joblib.load(f'{self.__path__}T0_level1_volume_fix_pred_scaler.pkl')
        tmp = pd.read_parquet(f'{self.__path__}predict_level1_volume_ETF.parquet')
        tmp = tmp[tmp.ticker==self.__c__].reset_index(drop=True)
        self.__data__ = tmp.iloc[-self.__time_period__:-1,:]
    def on_session_begin(self, context: CtaContext, curTDate: int):
        if os.path.exists(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_min1_OFI.csv'):
            self.df=pd.read_csv(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_min1_OFI.csv')
        else:
            self.df = pd.DataFrame(np.zeros((240,len(self.__data__.columns[0:-1]))),columns=self.__data__.columns[0:-1]) # 存储1min的数据
        
    
    def on_init(self, context:CtaContext):
        print(f'subscribing ETF Bar data {self.__code__}')
        context.stra_get_bars(self.__code__, self.__period__, 1, isMain = True)
        print(f'subscribing ETF snapshot data {self.__code__}')
        context.stra_sub_ticks(self.__code__)
        if not os.path.exists(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}'):
            os.makedirs(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}')
        if os.path.exists(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_OFI.csv'):
            tick_csv = pl.read_csv(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_OFI.csv')
            if len(tick_csv)>0:
                self.tick = tick_csv
            else:
                self.tick=pl.DataFrame() 
        else:
            self.tick=pl.DataFrame() 
        # print(self.tick)
    
    def on_calculate(self, context:CtaContext):
        print(f'{context.stra_get_time()} on_calc triggered')
        code = self.__code__    #品种代码
        
        time = context.stra_get_time() # 拿到当前分钟 1231
        date = context.stra_get_date() # 拿到当前日期 20180521
        new_data = self.tick
        # new_data = pd.read_csv(f'F:/deploy/ETF_T0/daily_tick/{self.__code__}_OFI.csv')  # 获取存储的tick数据
        columns_mapping ={
            'time': 'OrigTime',
            'code': 'ticker',
            'price': 'LastPx',
            'open': 'OpenPx',
            'high': 'HighPx',
            'low': 'LowPx',
            'upper_limit': 'UpLimitPx',
            'lower_limit': 'DownLimitPx',
            'total_volume': 'TotalVolumeTrade',
            # 'volume':'volume',
            'total_turnover': 'TotalValueTrade',
            'turnover': 'amount',
            'trading_date': 'TradeDate',
            'pre_close': 'PreClosePx',
            'bid_price_0': 'BidPx1',
            'bid_price_1': 'BidPx2',
            'bid_price_2': 'BidPx3',
            'bid_price_3': 'BidPx4',
            'bid_price_4': 'BidPx5',
            'ask_price_0': 'OfferPx1',
            'ask_price_1': 'OfferPx2',
            'ask_price_2': 'OfferPx3',
            'ask_price_3': 'OfferPx4',
            'ask_price_4': 'OfferPx5',
            'bid_qty_0': 'BidSize1',
            'bid_qty_1': 'BidSize2',
            'bid_qty_2': 'BidSize3',
            'bid_qty_3': 'BidSize4',
            'bid_qty_4': 'BidSize5',
            'ask_qty_0': 'OfferSize1',
            'ask_qty_1': 'OfferSize2',
            'ask_qty_2': 'OfferSize3',
            'ask_qty_3': 'OfferSize4',
            'ask_qty_4': 'OfferSize5',

        }
        new_data = new_data.rename(columns_mapping)
        time_index = minutes_list().index(time)
        if time_index<=5:
            min5_before = 930
        else:
            min5_before = minutes_list()[time_index-6]
        min1_before = minutes_list()[time_index-1]
        # print(new_data)
        # print(new_data.filter(pl.col('OrigTime')>=(date*1000000000+min5_before*100000)))
        df_min = get_snapshot_sz(new_data.filter((pl.col('OrigTime')>=(date*1000000000+min5_before*100000))&(pl.col('OrigTime')<(date*1000000000+min1_before*100000)))).to_pandas()
        # for i in df_min.columns.to_list():
        #    print(i)
        # print(df_min.columns.to_list())
        self.current_time = df_min['minute'].values[-1]
        # print(self.df)
        # print(df_min)
        df_min = df_min[self.__data__.columns[0:-1]]
        self.df.loc[minute_to_index()[self.current_time],:] = df_min.iloc[-1,:].values
        self.df.to_csv(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_min1_OFI.csv',index=False)
        # print(data.columns)
        df = self.df
        df=df[df.TradeDate!=0].reset_index(drop=True)
        # print(df)

        N = 3
        names = df.columns
        drop_names = ['TradeDate','OrigTime']  
        # drop_names += ['datetime','volatility_QMLE','real_volume','date','factor5','factor6']
        drop_names += ['datetime','real_volume','real_return','factor5','factor6','factor4','factor10']
        drop_names += ['wap1_sum','wap1_std','price_spread_sum','price_spread_max','price_spread2_sum','price_spread2_max']
        drop_names += ['bid_spread_sum','bid_spread_max','ask_spread_sum','ask_spread_max','total_volume_sum']
        drop_names += ['total_volume_max','total_bid_qty_sum','total_bid_qty_max','total_ask_qty_sum','total_ask_qty_max']
        drop_names += ['volume_imbalance_sum','volume_imbalance_max','bid_ask_spread_sum','bid_ask_spread_max']
        drop_names += ['volatility_3','volatility_1']  
        need_names = list(set(names)-set(drop_names))        
        need_names.sort()
        # print(need_names)
        # 做了点什么
        X = df[need_names]
        XX = pd.DataFrame()
        cols = X.columns
        for n in range(N):
            icols = [col+f'_{n}' for col in cols]
            tmp = X.shift(n)
            XX = pd.concat([XX,tmp.rename(columns=dict(zip(cols,icols)))], axis=1)
        X = XX.fillna(0)
        # apply transform
        X_rolling = self.__rolling_scaler__.transform(X.values)
        y_pred_rolling = self.__rolling_model__.predict(X_rolling, num_iteration=self.__rolling_model__.best_iteration)        
        X_fix = self.__fix_scaler__.transform(X.values)
        y_pred_fix = self.__fix_model__.predict(X_fix,num_iteration=self.__fix_model__.best_iteration)
        
        y_pred = (y_pred_rolling + y_pred_fix)/2      
        # print(y_pred)
        # print(df)
        df['pred_volume'] = y_pred
        # print(df['pred_volume'])
        
        # print(self.__data__)
        df_all = pd.concat([self.__data__,df],axis=0,ignore_index=True)
        # print(df_all)
        df_all['pred_volume_ma'] = df_all.pred_volume.rolling(231*3,min_periods=1).mean()
        df_all['pred_volume_ewma1'] = df_all.pred_volume.ewm(halflife=1).mean()
        df_all['pred_volume_ewma2'] = df_all.pred_volume.ewm(halflife=2).mean()
        df_all['pred_volume_ewma3'] = df_all.pred_volume.ewm(halflife=3).mean()
        df_all['pred_volume_avg']= (df_all.pred_volume+df_all.pred_volume_ewma1+df_all.pred_volume_ewma2+df_all.pred_volume_ewma3)/4
        df_all['pred_volume_stda'] = df_all.pred_volume.rolling(231*3,min_periods=1).std()
        print(f'time{df_all["minute"].values[-1]} volume_avg{df_all["pred_volume_avg"].values[-1]} volume_ma{df_all["pred_volume_ma"].values[-1]} volume_stda{df_all["pred_volume_stda"].values[-1]}')
        # df_all['position'] = np.where((df_all.pred_volume>(df_all.pred_volume_ma+1*df_all.pred_volume_stda)),1,0)
        df_all['position'] = np.where((df_all.pred_volume_avg>(df_all.pred_volume_ma+1*df_all.pred_volume_stda)),1,0)*np.sign(df_all.OFI1)
        df_all['signal'] = df_all['position'].diff()

        df_all.to_csv(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_pred_vol_OFI.csv',index=False)
        
        # print(df_all)
        # position = df_all.loc[df_all.minute==self.current_time,'position'].values[-1]   # 1 or 0
        # signal = df_all.loc[df_all.minute==self.current_time,'signal'].values[-1]  
        if time>1457:
            position = 0
        else:
            position = df_all['position'].values[-1]     
        signal = df_all['signal'].values[-1]
        
        #读取当前仓位
        curPos = context.stra_get_position(code)/self.__trdUnit__
        # 需要重写
        context.set_position(code, self.__trdUnit__*position)
        print(f'{context.stra_get_time()} set {self.__code__} position {self.__trdUnit__*position}')
        context.stra_log_text(f'{context.stra_get_time()} set {self.__code__} position {self.__trdUnit__*position}')       


    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        cols  = ['time','exchg','code']
        cols += ['price','open','high','low','settle_price']
        cols += ['upper_limit','lower_limit']
        cols += ['total_volume','volume','total_turnover','turnover','open_interest','diff_interest']
        cols += ['trading_date','action_date','action_time']
        cols += ['pre_close','pre_settle','pre_interest']
        cols += [f'bid_price_{i}' for i in range(10)]
        cols += [f'ask_price_{i}' for i in range(10)]
        cols += [f'bid_qty_{i}' for i in range(10)]
        cols += [f'ask_qty_{i}' for i in range(10)]        
        # print(len(newTick.tolist()), len(cols))
        # print(newTick.tolist())

        tmp = pl.DataFrame([newTick.tolist()], schema=cols)

        # tick_minute = tmp["time"][0]//100000%10000
        # current_time = context.stra_get_time()
        # if current_time == 931:
        #     current_time = 931
        # else:
        #     current_time = index_to_minute()[minute_to_index()[current_time]-1]

        if len(self.tick)==0:
            last_tick_time = tmp["time"][0]
        else:
            last_tick_time = self.tick['time'][-1]
        current_tick_time = tmp["time"][0]
        if current_tick_time >= last_tick_time:
            print(f'code:{self.__code__} tick_time:{tmp["time"][0]} ETF_T0_lxy_OFI ')
            self.tick = pl.concat([self.tick,tmp])
            self.tick.write_csv(f'F:/deploy/ETF_T0/daily_tick/{context.stra_get_date()}/{self.__code__}_OFI.csv')
        # print(self.tick)
        
        return
    
    def on_session_end(self, context: CtaContext, curTDate: int):
        print('on session end')
        return 
    
    
    
    