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
from autogluon.tabular import TabularDataset, TabularPredictor

# 一些函数
def next_minute(minute):
    # minute -> 1122
    minute = int(minute)
    h, m = minute//100, minute%100
    nm = (m+1)%60+ 100*((m+1)//60+h)
    if nm < 930:
        nm = 930
    # elif nm > 1130 and nm <= 1300:
    #     nm = 1301
    elif nm == 1501:
        nm = 1500
    elif nm > 1501:
        nm = 1501
    return nm

def minutes_list():
    hours = [9,10,11,13,14]
    minutes = range(60)
    min_list = []
    for hour in hours:
        for minute in minutes:
            if hour == 9 and minute < 30:
                continue
            if hour == 11 and minute > 30:
                continue
            if hour ==13 and minute == 0:
                continue
            # if hour == 14 and minute > 60:
            #     continue
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

# def log_return(series):
#     return np.log(series).diff().sum()

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

    df = df.sort(by='OrigTime')
    
    ticker = df['ticker'][0][-6:]
    df = df.with_columns(pl.lit(ticker).alias('ticker'))
    
    df = df.with_columns(df['OrigTime'].cast(pl.Utf8).str.strptime(pl.Datetime,format='%Y%m%d%H%M%S%f').alias('datetime'))
    df = df.with_columns((df['OrigTime']//100000%10000).alias('minute'))
    df = df.with_columns(df['minute'].apply(next_minute))
    df = df.filter(((df['minute']>=930) & (df['minute']<=1130)) | ((df['minute']>=1301) & (df['minute']<=1500)))
    if df['TotalVolumeTrade'].sum() == 0:
        return pl.DataFrame()
    
    df = df.filter((pl.col('BidPx1')+pl.col('OfferPx1'))>0)

    # 涨跌停处理
    df = df.with_columns(pl.when(pl.col('BidPx1')==0).then(pl.col('OfferPx1')).otherwise(pl.col('BidPx1')).alias('BidPx1'))
    df = df.with_columns(pl.when(pl.col('OfferPx1')==0).then(pl.col('BidPx1')).otherwise(pl.col('OfferPx1')).alias('OfferPx1'))

    # Calculate Wap
    df = df.with_columns(calc_wap1(df).alias('wap1')) #只选一个

    # Calculate log returns
    df = df.with_columns(np.log(pl.col('wap1')).diff().alias('log_return')) 

    # df = df.fill_nan(0)
    df = df.fill_null(0)
    
    # 如果不成交的话 快照的ohlc就是0 不会更新
    ohlc_cols = ['OpenPx','HighPx','LowPx','LastPx']    
    for name in ohlc_cols:
        # exec(f'df = df.with_columns(pl.when(pl.col("NumTrades")==0).then({None}).otherwise(pl.col("{name}")).alias("{name}"))')
        df = df.with_columns(pl.when(pl.col("TotalVolumeTrade")==0).then(None).otherwise(pl.col(f"{name}")).alias(f"{name}"))
    df = df.fill_null(strategy='forward')
    df = df.fill_null(strategy='backward')

    # 时间是精确到秒的，直接去groupby就行了
    df_min = df.groupby('minute').last().sort('minute')
    
    # level1最高价和最低价的数值不准 重新算 但是如果按照lastpx去计算的话 有可能会丢失掉这个3s内的最高价之类的 但是鉴于level1那个实在错的离谱 还是按照这种方式处理
    # df_min = df_min.with_columns(pl.col('LastPx').cummax().alias('HighPx'))
    # df_min = df_min.with_columns(pl.col('LastPx').cummin().alias('LowPx'))
    
    bid_px_cols = ['BidPx%s'%(s) for s in range(1,6)]
    offer_px_cols = ['OfferPx%s'%(s) for s in range(1,6)]
    bid_size_cols = ['BidSize%s'%(s) for s in range(1,6)]
    offer_size_cols = ['OfferSize%s'%(s) for s in range(1,6)]
    

    # Calculate spread
    df_min = df_min.with_columns(((df_min['OfferPx1'] - df_min['BidPx1']) / ((df_min['OfferPx1'] + df_min['BidPx1']) / 2)).alias('price_spread'))
    df_min = df_min.with_columns(((df_min['OfferPx2'] - df_min['BidPx2']) / ((df_min['OfferPx2'] + df_min['BidPx2']) / 2)).alias('price_spread2'))
    df_min = df_min.with_columns((df_min['BidPx1'] - df_min['BidPx2']).alias('bid_spread'))
    df_min = df_min.with_columns((df_min['OfferPx1'] - df_min['OfferPx2']).alias('ask_spread'))
    df_min = df_min.with_columns((abs(df_min['bid_spread'] - df_min['ask_spread'])).alias("bid_ask_spread"))


    # 添加其他基本特征
    # total bid/ask quantity, 需要补零吗(0一般是涨跌停的情况)
    total_bid_qty = np.array(df_min[bid_size_cols].sum(axis=1)).astype(float)
    total_ask_qty = np.array(df_min[offer_size_cols].sum(axis=1)).astype(float)
    # total_bid_qty[total_bid_qty==0] = np.nan
    # total_ask_qty[total_ask_qty==0] = np.nan
    df_min = df_min.with_columns(pl.Series(total_bid_qty).alias('total_bid_qty'))
    df_min = df_min.with_columns(pl.Series(total_ask_qty).alias('total_ask_qty'))
    df_min = df_min.with_columns((pl.col('total_ask_qty')-pl.col('total_bid_qty')).alias('total_qty_diff'))
    # weighted quote price
    df_min = df_min.with_columns(((df_min[bid_size_cols]*df_min[bid_px_cols]).sum(axis=1)/df_min['total_bid_qty']).alias('weighted_bid_prc'))
    df_min = df_min.with_columns(((df_min[offer_size_cols]*df_min[offer_px_cols]).sum(axis=1)/df_min['total_ask_qty']).alias('weighted_ask_prc'))
    
    df_min = df_min.with_columns((df_min['total_ask_qty'] + df_min['total_bid_qty']).alias('total_volume'))   #考虑改10档
    df_min = df_min.with_columns((abs(df_min['total_ask_qty'] - df_min['total_bid_qty'])).alias('volume_imbalance'))   #考虑改10档
    

    
    df_min_return = df.groupby('minute').agg(
        pl.col('log_return').sum()
        ).sort('minute')
    df_min = pl.concat([df_min.drop(['log_return']),df_min_return],how='align')
    
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
    df_min = pl.concat([df_min,df_feature],how='align')
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
            abs_diff_v = (df_id['volume_tick'] - df_id['volume_tick'].mean()).abs().median()    
            energy_v = (df_id['volume_tick']**2).sum()
            iqr_p_v = np.percentile(df_id['volume_tick'],75) - np.percentile(df_id['volume_tick'],25)
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
    
    df_min = df_min.drop('volume_tick')
    
    for i in range(1,6):
        df_min = df_min.with_columns(pl.when(pl.col(f'BidPx{i}').diff()>0).then(pl.col(f'BidSize{i}')).when(pl.col(f'BidPx{i}').diff()==0).then(pl.col(f'BidSize{i}').diff()).otherwise(-pl.col(f'BidSize{i}')).alias(f'bOF{i}'))
        df_min = df_min.with_columns(pl.when(pl.col(f'OfferPx{i}').diff()>0).then(-pl.col(f'OfferSize{i}')).when(pl.col(f'OfferPx{i}').diff()==0).then(pl.col(f'OfferSize{i}').diff()).otherwise(pl.col(f'OfferSize{i}')).alias(f'aOF{i}'))
        df_min = df_min.with_columns((pl.col(f'bOF{i}')-pl.col(f'aOF{i}')).alias(f'OFI{i}'))
    
    minute_len = len(df_min['minute'])
    vol_QMLE=[0]*(minute_len-1)
    vol = [0]*(minute_len-1)
    KK = 100
    # vol要改成年化的 目前是5min的 *4800/l可以改成日频
    Y = df.filter(~df['log_return'].is_nan())['log_return']
    l = len(Y)
    if l > 1:
        if minute_len<5:
            dt = 1/(14400/(60/(l/(minute_len+1))))
        else:
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
            if (a*a-b*c) == 0:
                    break
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

def log_return(series):
    return np.log(series).diff().fillna(0)

def mean_second_derivative_centra(x):
    sum_value=0
    for i in range(len(x)-5):
        sum_value+=(x[i+5]-2*x[i+3]+x[i])/2
    return sum_value/(2*(len(x)-5))

def first_location_of_maximum(x):
    max_value=max(x)
    for loc in range(len(x)):
        if x[loc]==max_value:
            return loc+1
    

class ETF_T0_lxy(BaseCtaStrategy):
    
    def __init__(self, name:str, exchg:str, code:str, period:str):
        BaseCtaStrategy.__init__(self, name)
        self.__code__ = f"{exchg}.ETF.{code}"
        self.__c__ = code
        self.__period__ = period    # m1
        self.__trdUnit__ = 100000
        self.__path__ = 'F:/deploy/ETF_T0 - insight/'
        self.__time_period__ = 240*10
        self.load_models()
        self.position={}

    def load_models(self):
        # 需要load models
        self.__rolling_model__ = TabularPredictor.load(f'{self.__path__}model/ag-20231031_021430/',require_py_version_match=False)
        # self.__fix_model__ = TabularPredictor.load(f'{self.__path__}model/ag-20231024_022921/')
        tmp = pd.read_parquet(f'{self.__path__}model/autogluon_pred_return_rolling_noadj.parquet')
        tmp = tmp[tmp.ticker==self.__c__].reset_index(drop=True)
        self.__data__ = tmp.iloc[-self.__time_period__:-1,:]
        self.df_ori_min1 = pd.read_parquet(f'{self.__path__}model/ETF_min_level1.parquet')
        # print(self.__data__)
    def on_session_begin(self, context: CtaContext, curTDate: int):
        if os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}_min1.csv'):
            self.df=pd.read_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}_min1.csv')
        else:
            self.df = pd.DataFrame(np.zeros((240,len(self.df_ori_min1.columns))),columns=self.df_ori_min1.columns) # 存储1min的数据

    
    def on_init(self, context:CtaContext):
        print(f'subscribing ETF Bar data {self.__code__}')
        context.stra_get_bars(self.__code__, self.__period__, 1, isMain = True)
        print(f'subscribing ETF snapshot data {self.__code__}')
        context.stra_sub_ticks(self.__code__)
        if not os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}'):
            os.makedirs(f'{self.__path__}daily_tick/{context.stra_get_date()}')
        if os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}.csv'):
            tick_csv = pl.read_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}.csv')
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
        time = context.stra_get_time() # 拿到当前分钟 1231  其实是当前分钟的下一分钟
        date = context.stra_get_date() # 拿到当前日期 20180521
        new_data_ori = self.tick
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
            'volume':'volume_tick',
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
        new_data = new_data_ori.rename(columns_mapping)
        time_index = minutes_list().index(time)
        if time_index<=5:
            min5_before = 930
        else:
            min5_before = minutes_list()[time_index-6]
        min1_before = minutes_list()[time_index-1]
        # print(min5_before)
        # print(min1_before)
        # print(new_data.filter(pl.col('OrigTime')>=(date*1000000000+min5_before*100000)))
        # print(new_data.filter((pl.col('OrigTime')>=(date*1000000000+min5_before*100000))&(pl.col('OrigTime')<(date*1000000000+min1_before*100000))))
        part = new_data.filter((pl.col('OrigTime')>=(date*1000000000+min5_before*100000))&(pl.col('OrigTime')<(date*1000000000+min1_before*100000)))
        if len(part)>0:
            df_min = get_snapshot_sz(part).to_pandas()
            self.current_time = df_min['minute'].values[-1]
            df_min = df_min[self.df.columns]
        else:
            df_min = pd.DataFrame(np.zeros((1,len(self.df.columns))),columns=self.df.columns)
            self.current_time = minutes_list()[time_index-1]

        self.df.loc[minute_to_index()[self.current_time],:] = df_min.iloc[-1,:].values
        self.df.to_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}_min1.csv',index=False)
        # print(data.columns)
        df_min_today = self.df[self.df.TradeDate!=0].reset_index(drop=True)
        # print(df_min_today.ticker)

        df = pd.concat([self.__data__[df_min_today.columns],df_min_today],axis=0,ignore_index=True)
        df['ticker'] = df['ticker'].astype('int64')
        df['minute5'] = df.minute//5
        # BidPrice1 在滚动窗口中的相对排名，其值在-1到1之间。
        df['bp_rank'] = df[['ticker','BidPx1']].groupby('ticker').rolling(100,min_periods=1).rank().reset_index().set_index('level_1')['BidPx1']/100*2-1
        # print(df[['ticker','BidPx1']].groupby('ticker').rolling(100,min_periods=1).rank())
        # 计算中间价的滚动100回报率
        df['mid_log_return'] = df[['ticker','mid']].groupby('ticker',group_keys=False).apply(log_return)['mid']
        df['mid_rolling_return'] = df[['ticker','mid_log_return']].groupby('ticker').rolling(100,min_periods=1).sum().reset_index().set_index('level_1')['mid_log_return']
        # 计算OfferPrice1的平均二阶中心差分
        df['center_second_derivative_centra'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(20).apply(mean_second_derivative_centra,raw=True).fillna(0).reset_index().set_index('level_1')['OfferPx1']
        # 近20个卖一数据里最大的那个的位置
        df['price_idxmax'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(20,min_periods=1).apply(first_location_of_maximum,raw=True).reset_index().set_index('level_1')['OfferPx1']
        # 计算快照数据中特定滚动窗口内的价格波动范围
        df['depth_price_range'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).max().reset_index().set_index('level_1')['OfferPx1']/df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).min().reset_index().set_index('level_1')['OfferPx1']-1
        # 100分钟内买1价的偏度
        df['depth_price_skew'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).skew().reset_index().set_index('level_1')['OfferPx1'].fillna(0)
        # 10min平均成交量/100min平均成交量
        # df['volume_min_factor'] = (df[['ticker','volume']].groupby('ticker').volume.rolling(10,min_periods=1).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(100,min_periods=1).mean().reset_index().set_index('level_1')['volume']).fillna(0)

        # df['abs_diff_v_ewma'] = df[['ticker','abs_diff_v']].groupby('ticker').abs_diff_v.ewm(halflife=5).mean().reset_index().set_index('level_1')['abs_diff_v']
        df['volume_ma_1'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(10).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_2'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(20).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_3'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(60).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_4'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(90).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_5'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_6'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(250).mean().reset_index().set_index('level_1')['volume']).fillna(0)

        df['price_ewma_5'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=5).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_10'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=10).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_20'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=20).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_60'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=60).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_120'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=120).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_250'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=250).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)

        df['return_std_5'] = df[['ticker','log_return']].groupby('ticker').rolling(5).std().reset_index().set_index('level_1')['log_return'].fillna(0)
        df['return_std_10'] = df[['ticker','log_return']].groupby('ticker').rolling(10).std().reset_index().set_index('level_1')['log_return'].fillna(0)
        df['return_std_20'] = df[['ticker','log_return']].groupby('ticker').rolling(20).std().reset_index().set_index('level_1')['log_return'].fillna(0)
        df['return_std_60'] = df[['ticker','log_return']].groupby('ticker').rolling(60).std().reset_index().set_index('level_1')['log_return'].fillna(0)
        df['return_std_120'] = df[['ticker','log_return']].groupby('ticker').rolling(120).std().reset_index().set_index('level_1')['log_return'].fillna(0)
        df['return_std_250'] = df[['ticker','log_return']].groupby('ticker').rolling(250).std().reset_index().set_index('level_1')['log_return'].fillna(0)

        part = df[['ticker','log_return','volume']].groupby('ticker').rolling(120).corr().reset_index()
        df['corr_120'] = part[part.level_2=='log_return'].set_index('level_1')['volume'].fillna(0)
        part = df[['ticker','log_return','volume']].groupby('ticker').rolling(250).corr().reset_index()
        df['corr_250'] = part[part.level_2=='log_return'].set_index('level_1')['volume'].fillna(0)


        df['corr_120_mean'] = df[['ticker','corr_120']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['corr_120'].fillna(0)
        df['corr_120_std'] = df[['ticker','corr_120']].groupby('ticker').rolling(120).std().reset_index().set_index('level_1')['corr_120'].fillna(0)
        df['corr_250_mean'] = df[['ticker','corr_250']].groupby('ticker').rolling(250).mean().reset_index().set_index('level_1')['corr_250'].fillna(0)
        df['corr_250_std'] = df[['ticker','corr_250']].groupby('ticker').rolling(250).std().reset_index().set_index('level_1')['corr_250'].fillna(0)


        # apply transform
        df_today = df[df.TradeDate==int(date)].reset_index(drop=True)
        # print(df_today.bp_rank)
        y_pred_rolling = self.__rolling_model__.predict(df_today)        
        y_pred = y_pred_rolling    

        df_today['pred_return'] = y_pred
        
        df_all = pd.concat([self.__data__,df_today],axis=0,ignore_index=True)

        df_all['pred_return_ewma_1'] = df_all.pred_return.ewm(halflife=1).mean()
    
        print(f'time:{df_all["minute"].values[-1]} pred_return:{round(df_all["pred_return"].values[-1]*10000,3)} pred_return_ewma_1:{round(df_all["pred_return_ewma_1"].values[-1]*10000,3)}')

        df_all['upthreshold'] = 0.00024
        df_all['downthreshold'] = -0.00024

        conditions = [
            (df_all['pred_return_ewma_1'] > df_all['upthreshold']),             # 条件1：做多
            (df_all['pred_return_ewma_1'] < df_all['downthreshold']),         # 条件2：做空
            (df_all['pred_return_ewma_1'] <= df_all['upthreshold']) & 
            (df_all['pred_return_ewma_1'] >= df_all['downthreshold']),           
        ]

        choices = [1, -1, 0]  # 分别对应条件1、条件2、条件3的标记

        df_all['position'] = np.select(conditions, choices, default=0)

        df_all['signal'] = df_all['position'].diff().fillna(0)

        df_all.to_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}_pred_return.csv',index=False)

        # 计算仓位 
        if time>1456:
            position = 0
        else:
            position = df_all['position'].values[-1]*int(self.__trdUnit__/df_all['LastPx'].values[-1])
        signal = df_all['signal'].values[-1]

        #读取当前仓位
        curPos = context.stra_get_position(code)
        
        # 记录最新要添加的仓位
        if (position!=0) & ((df_all['position'].values[-2]==0) | (curPos==0)):
            self.position[self.current_time] = position
        
        # 如果信号连续 选择更新最新的时间
        if (position==df_all['position'].values[-2]) & (position!=0):
            self.position[self.current_time] = self.position[list(self.position.keys())[-1]]
            self.position.pop(list(self.position.keys())[-1])

        # 实时更新仓位 若仓位超时 则删除：
        for i in list(self.position.keys()):
            if minute_to_index()[self.current_time]-minute_to_index()[i]>=5:
                self.position.pop(i)
        print(self.position)
        all_position = sum(self.position.values())
        
        # 需要重写
        context.set_position(code, all_position)
        context.stra_log_text(f'{context.stra_get_time()} set {self.__code__} position {all_position}')       


    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        # print(f"{newTick.time} {stdCode}, {newTick.bid_price_0}, {newTick.ask_price_0}")
        # print(f"{newTick.time} {stdCode}, {newTick.total_volume}, {newTick.total_turnover}")
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
        # print(tmp)
        # print(tmp["time"][0])
        # tick_minute = tmp["time"][0]//100000%10000
        # current_time = context.stra_get_time()
        # if current_time == 931:
        #     current_time = 931
        # else:
        #     current_time = index_to_minute()[minute_to_index()[current_time]-1]
        # print(tick_minute)
        # print(current_time)
        if len(self.tick)==0:
            last_tick_time = tmp["time"][0]-1
        else:
            last_tick_time = self.tick['time'][-1]
        current_tick_time = tmp["time"][0]
        if current_tick_time > last_tick_time:
            # last_tick_time = current_tick_time
            # print(f'code:{self.__code__} tick_time:{tmp["time"][0]} ETF_T0_lxy ')
            self.tick = pl.concat([self.tick,tmp])
            self.tick.write_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/{self.__code__}.csv')
        # print(self.tick)
        
        return
    
    def on_session_end(self, context: CtaContext, curTDate: int):
        print('on session end')
        return 
    
    
    
    