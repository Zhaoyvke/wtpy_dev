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
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
import tsfresh

# 一些函数
def next_minute(minute):
    # minute -> 112203
    minute = int(minute)
    h, m, s = minute//10000, minute//100%100, minute%100
    time = datetime.datetime(2020,1,1,h,m,s)
    # nm = (m+1)%60+ 100*((m+1)//60+h)
    nm = int(time.strftime('%H%M'))
    if s!=0:
        time += datetime.timedelta(minutes=1)
    else:
        if nm==930 or nm==1300:
            time += datetime.timedelta(minutes=1)
    nm = int(time.strftime('%H%M'))
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
            if hour == 13 and minute == 0:
                continue
            # if hour == 14 and minute > 57:
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
    Y = part.filter(~part['log_return_tick'].is_nan())['log_return_tick']
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

def cross(df,col):
    return pd.DataFrame(np.sign(df['LastPx'] - df[col]).fillna(0))
def new_ret_5(part):
    y = np.argsort(part[:,0])
    x = part[y,1]
    return x[-len(x)//5:].mean()

def new_ret_1(part):
    y = np.argsort(part[:,0])
    x = part[y,1]
    if len(x)//5==0:
        return x.mean()
    else:
        return x[:len(x)//5].mean()
def new_ret_5_1(part):
    y = np.argsort(part[:,0])
    x = part[y,1]
    if len(x)//5==0:
        return x[-len(x)//5:].mean()-x.mean()
    else:
        return x[-len(x)//5:].mean()-x[:len(x)//5].mean()
def new_ret_1_5(part):
    y = np.argsort(part[:,0])
    x = part[y,1]
    if len(x)//5==0:
        return x.mean() - x[-len(x)//5:].mean()
    else:
        return x[:len(x)//5].mean() - x[-len(x)//5:].mean()
def volume_return(part):
    # part[:,0] return part[:,1] volume
    # 第1列根据第0列排序
    y = np.argsort(part[:,0])
    y_reverse = y[::-1]
    x = np.log(part[y,1]+1)
    cum_x = x.cumsum()
    x_reverse = np.log(part[y_reverse,1]+1)
    cum_x_reverse = x_reverse.cumsum()[::-1]
    factor = (cum_x - cum_x_reverse).cumsum()
    return factor[-1]

def amplitude_return(part):
    # part[:,0] return part[:,1] volume
    # 第1列根据第0列排序
    y = np.argsort(part[:,0])
    y_reverse = y[::-1]
    x = part[y,1]
    cum_x = x.cumsum()
    x_reverse = part[y_reverse,1]
    cum_x_reverse = x_reverse.cumsum()[::-1]
    factor = (cum_x - cum_x_reverse).cumsum()
    return factor[-1]

def cal_reverse_factor(ori_value):
    # print(ori_value)
    value = np.asarray(ori_value)[:,1]
    factor = np.zeros(len(value))
    k = 0.1
    # print(value)
    if value[0] != 0:
        factor[0] = value[0]
    for i in range(1,len(value)):
        if value[i] != 0:
            factor[i] = value[i]
        else:
            factor[i] = factor[i-1]*np.exp(-k)
    # df = pd.DataFrame(factor,columns='factor')
    # df['Index'] = ori_value.index
    return pd.DataFrame(factor,index=ori_value.index)
def get_snapshot_sz(df):
    t0 = datetime.datetime.now()
    # level1 对应可用字段
    # 代码（六位） 交易日 日期 时间（93020223）
    # 价格 开盘价 最高价 最低价 昨收价 总成交量 成交量 总成交额 成交额 总持 增仓
    # 买一-五价 买一-五量 卖一-五价 卖一-五量 计数
    df = df.sort(by='OrigTime')
    
    ticker = df['ticker'][0][-6:]
    df = df.with_columns(pl.lit(ticker).alias('ticker'))
    
    # df = df.with_columns(df['OrigTime'].cast(pl.Utf8).str.strptime(pl.Datetime,format='%Y%m%d%H%M%S%f').alias('datetime'))
    df = df.with_columns((df['OrigTime']//1000%1000000).alias('minute'))
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

    # df = df.fill_nan(0)
    df = df.fill_null(0)
    
    # 如果不成交的话 快照的ohlc就是0 不会更新
    ohlc_cols = ['OpenPx','HighPx','LowPx','LastPx']    
    for name in ohlc_cols:
        df = df.with_columns(pl.when(pl.col(f"{name}")==0).then(pl.col('PreClosePx')).otherwise(pl.col(f"{name}")).alias(f"{name}"))


    # Calculate log returns
    df = df.with_columns(np.log(pl.col('LastPx')).diff().alias('log_return_tick')) 

     # # 需要计算每个3s的volume
    df = df.with_columns(pl.col('TotalVolumeTrade').diff().alias('volume_tick')) # level1有这两个数据 到时候需要把这两行给注释掉
    df = df.with_columns(pl.when(pl.col('volume_tick').is_null()).then(pl.col('TotalVolumeTrade')).otherwise(pl.col('volume_tick')).alias('volume_tick'))

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
    
    df_min = df_min.with_columns((df_min['LastPx']-df_min['PreClosePx']).alias('PxChange1'))
    df_min = df_min.with_columns(pl.when(pl.col('LastPx').diff().is_null()).then(0).otherwise(pl.col('LastPx').diff()).alias('PxChange2'))
    
    df_min = df_min.with_columns(((pl.col('BidPx1')+pl.col('OfferPx1'))/2).alias('mid'))
    df_filter = df.filter(df['volume_tick']>0)
    df_feature = df_filter.groupby('minute').agg(
                    pl.col('LastPx').first().alias('open'),
                    pl.col('LastPx').max().alias('high'),
                    pl.col('LastPx').min().alias('low'),
                    pl.col('LastPx').last().alias('close'),
                    ).sort('minute')
    df_min = pl.concat([df_min,df_feature],how='align')

    df_min = df_min.with_columns(df_min['close'].fill_null(strategy='forward'))
    ohlc_cols = ['open','high','low']    
    for name in ohlc_cols:
        df_min = df_min.with_columns(pl.when(pl.col(f"{name}").is_null()).then(pl.col('close')).otherwise(pl.col(f"{name}")).alias(f"{name}"))

    
    # volume 直接取最后一行的话 其实拿的是最后一个tick的volume 而不是这一分钟的 所以还是用总的进行计算比较对
    df_min = df_min.with_columns(pl.col('TotalVolumeTrade').diff().alias('volume'))
    df_min = df_min.with_columns(pl.when(pl.col('volume').is_null()).then(pl.col('TotalVolumeTrade')).otherwise(pl.col('volume')).alias('volume'))
    df_min = df_min.with_columns(pl.col('TotalValueTrade').diff().alias('amount'))
    df_min = df_min.with_columns(pl.when(pl.col('amount').is_null()).then(pl.col('TotalValueTrade')).otherwise(pl.col('amount')).alias('amount'))
    
    df_min = df_min.with_columns((pl.col('amount')/pl.col('volume')).alias('vwap'))
    df_min = df_min.with_columns(pl.when(pl.col('vwap').is_nan()).then(pl.col('LastPx')).otherwise(pl.col('vwap')).alias('vwap'))
    
    
    df_min = df_min.with_columns(np.log(pl.col('vwap')).diff().alias('vwap_log_return'))
    df_min = df_min.with_columns(np.log(pl.col('wap1')).diff().alias('wap1_log_return'))
    
    df_min = df_min.filter(df_min['minute']>930)
    df_min = df_min.filter(pl.col('TotalVolumeTrade')>0)
    # print(df_min)
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
            # abs_diff和iqr_p相关系数比较高0.95
            # vol vars
            # abs_diff_v = (df_id['TotalVolumeTrade'] - df_id['TotalVolumeTrade'].mean()).abs().median()    
            # energy_v = (df_id['TotalVolumeTrade']**2).sum()
            # iqr_p_v = np.percentile(df_id['TotalVolumeTrade'],75) - np.percentile(df_id['TotalVolumeTrade'],25)
            abs_diff_v = (df_id['volume_tick'] - df_id['volume_tick'].mean()).abs().median()    
            energy_v = (df_id['volume_tick']**2).sum()
            iqr_p_v = np.percentile(df_id['volume_tick'],75) - np.percentile(df_id['volume_tick'],25)
            # abs_diff_v和iqr_p_v相关系数比较高0.88
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
    
    minute_len = len(df_min['minute'])
    vol_QMLE=[0]*(minute_len-1)
    vol = [0]*(minute_len-1)
    KK = 100
    # vol要改成年化的 目前是5min的 *4800/l可以改成日频
    Y = df.filter(~df['log_return_tick'].is_nan())['log_return_tick']
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
    df_min = df_min.with_columns((pl.col('TradeDate')*10000+pl.col('minute')).cast(pl.Utf8).str.strptime(pl.Datetime,format='%Y%m%d%H%M').alias('datetime'))

    df_min = df_min.with_columns((pl.col('PreClosePx')*1.1).alias('UpLimitPx'))
    df_min = df_min.with_columns((pl.col('PreClosePx')*0.9).alias('DownLimitPx'))
    df_min = df_min.drop(['volume_tick','log_return_tick'])
    df_min = df_min.fill_null(0)
    df_min = df_min.fill_nan(0)
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
    
    def __init__(self, name:str, ETFs:list, period:str):
        BaseCtaStrategy.__init__(self, name)
        # self.__code__ = f"{exchg}.ETF.{code}"
        # self.__c__ = code
        self.__etfs__ = ETFs
        self.__period__ = period    # m1
        self.__trdUnit__ = 100000
        self.__path__ = 'F:/deploy/ETF_T0_multi_underlying/'
        self.__time_period__ = int((datetime.datetime.today()-datetime.timedelta(days=10)).strftime('%Y%m%d'))
        self.load_models()
        self.position=dict()

    def load_models(self):
        # 需要load models
        self.__rolling_model__ = TabularPredictor.load(f'{self.__path__}model/ag-20240108_043013/',require_py_version_match=False,require_version_match=False)
        # self.__fix_model__ = TabularPredictor.load(f'{self.__path__}model/ag-20231024_022921/')
        tmp = pd.read_parquet(f'{self.__path__}model/autogluon_pred_wapreturn_rolling_noadj_myselectnew.parquet')
        self.pred_return = tmp[tmp.TradeDate>=self.__time_period__]
        self.df_ori_min1 = pd.read_parquet(f'{self.__path__}model/ETF_min_level1.parquet')
        # print(self.pred_return)
    def on_session_begin(self, context: CtaContext, curTDate: int):
        if os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}/min1.csv'):
            self.df = pd.read_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/min1.csv')
        else:
            self.df = pd.DataFrame(columns=self.df_ori_min1.columns).astype(self.df_ori_min1.dtypes) # 存储1min的数据

    
    def on_init(self, context:CtaContext):
        # print(f'subscribing ETF Bar data {self.__code__}')
        # context.stra_get_bars(self.__code__, self.__period__, 1, isMain = True)
        for exchg,code in self.__etfs__:
            code_name = f"{exchg}.ETF.{code}"
            print(f'subscribing ETF snapshot data {code_name}')
            context.stra_sub_ticks(code)
            self.position[int(code)] = dict()
        if not os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}'):
            os.makedirs(f'{self.__path__}daily_tick/{context.stra_get_date()}')
        if os.path.exists(f'{self.__path__}daily_tick/{context.stra_get_date()}/tick.csv'):
            tick_csv = pl.read_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/tick.csv')
            if len(tick_csv)>0:
                self.tick = tick_csv
            else:
                self.tick=pl.DataFrame() 
        else:
            self.tick=pl.DataFrame() 
        # print(self.tick)
        print('on_init end')
    
    def on_calculate(self, context:CtaContext):
        # print(f'{context.stra_get_time()} on_calc triggered')
        context.stra_log_text(f'{context.stra_get_time()} on_calc triggered')

        # code = self.__code__    #品种代码
        time = context.stra_get_time() # 拿到当前分钟 1231  其实是当前分钟的下一分钟
        # print(time)
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
            min5_before = minutes_list()[time_index-5]
        current_time = int(time)
        # min1_before = minutes_list()[time_index-1]
        # print(min5_before)
        # print(min1_before)
        # print(new_data.filter(pl.col('OrigTime')>=(date*1000000000+min5_before*100000)))
        # print(new_data.filter((pl.col('OrigTime')>=(date*1000000000+min5_before*100000))&(pl.col('OrigTime')<(date*1000000000+min1_before*100000))))
        for ticker in new_data.ticker.unique():
            if current_time == 931:
                part = new_data.filter((pl.col('OrigTime')<(date*1000000000+current_time*100000)) & (pl.col('ticker')==ticker))
            else:
                part = new_data.filter((pl.col('OrigTime')>=(date*1000000000+min5_before*100000)) & (pl.col('OrigTime')<(date*1000000000+current_time*100000)) & (pl.col('ticker')==ticker))
            # if len(part)>0:
            df_min = get_snapshot_sz(part).to_pandas()
            self.current_time = df_min['minute'].values[-1]
            df_min = df_min[self.df.columns]
            # else:
            #     df_min = pd.DataFrame(np.zeros((1,len(self.df.columns))),columns=self.df.columns)
            #     self.current_time = current_time
            if current_time == 931:
                self.df = self.df.append(df_min.iloc[-2,:].values,ignore_index=True)
            else:
                self.df = self.df.append(df_min.iloc[-1,:].values,ignore_index=True)
            
        self.df.to_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/min1.csv',index=False)
        # print(data.columns)
        df_min_today = self.df.copy()
        # print(df_min_today.ticker)

        df_min_today['log_return'] = df_min_today['wap1_log_return'].copy()
        df_min_today = df_min_today.drop(columns = ['wap1_log_return'])

        df = pd.concat([self.pred_return[df_min_today.columns],df_min_today],axis=0,ignore_index=True)
        df['ticker'] = df['ticker'].astype('int64')
        # 获取5min因子
        df['minute5'] = df.minute//5
        # BidPrice1 在滚动窗口100中的相对排名，其值在-1到1之间。
        df['bp_rank'] = df[['ticker','BidPx1']].groupby('ticker').rolling(100,min_periods=1).rank().reset_index().set_index('level_1')['BidPx1'].fillna(0)/100*2-1

        # 计算中间价的滚动100回报率
        df['mid_log_return'] = df[['ticker','TradeDate','mid']].groupby(['ticker','TradeDate'],group_keys=False).apply(log_return)['mid'].fillna(0)
        df['mid_rolling_return'] = df[['ticker','mid_log_return']].groupby('ticker').rolling(100,min_periods=1).sum().reset_index().set_index('level_1')['mid_log_return']
        df = df.drop(columns = ['mid_log_return'])

        # 计算OfferPrice1的20平均二阶中心差分 没有tsfresh算的快
        df['center_second_derivative_centra'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(20).apply(mean_second_derivative_centra,raw=True).fillna(0).reset_index().set_index('level_1')['OfferPx1']
        
        # 近20个卖一数据里最大的那个的位置 没有tsfresh算得快
        df['price_idxmax'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(20).apply(getattr(tsfresh.feature_extraction.feature_calculators,'first_location_of_maximum'),raw=True).reset_index().set_index('level_1')['OfferPx1'].fillna(0)*20+1


        # 计算快照数据中特定滚动窗口100内的价格波动范围
        df['depth_price_range'] = ((df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).max().reset_index().set_index('level_1')['OfferPx1']/df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).min().reset_index().set_index('level_1')['OfferPx1'])-1).fillna(0)
        
        # 偏度
        df['close_skew'] = df[['ticker','OfferPx1']].groupby('ticker').rolling(100,min_periods=1).skew().reset_index().set_index('level_1')['OfferPx1'].fillna(0)
        df['depth_price_skew'] = df[["BidPx5","BidPx4","BidPx3","BidPx2","BidPx1","OfferPx1","OfferPx2","OfferPx3","OfferPx4","OfferPx5"]].skew(axis=1)
        df['depth_volume_skew'] = df[["BidSize5","BidSize4","BidSize3","BidSize2","BidSize1","OfferSize1","OfferSize2","OfferSize3","OfferSize4","OfferSize5"]].skew(axis=1)


        df['volume_ma_1'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(10).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_2'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(20).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_3'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(60).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_4'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(90).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_5'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(120).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_6'] = (df[['ticker','volume']].groupby('ticker').rolling(5).mean().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(250).mean().reset_index().set_index('level_1')['volume']).fillna(0)

        df['price_ewma_5'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=5).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_10'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=10).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_20'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=20).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_60'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=60).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_120'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=120).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['price_ewma_250'] = df[['ticker','LastPx']].groupby('ticker').ewm(halflife=250).mean().reset_index().set_index('level_1')['LastPx'].fillna(0)

        df['return_std_250'] = df[['ticker','log_return']].groupby('ticker').rolling(250).std().reset_index().set_index('level_1')['log_return'].fillna(0)

        part = df[['ticker','log_return','volume']].groupby('ticker').rolling(120).corr().reset_index()
        df['corr_120'] = (part[part.level_2=='log_return'].set_index('level_1')['volume']).fillna(0)

        df['corr_120_mean'] = df[['ticker','corr_120']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['corr_120'].fillna(0)
        df = df.drop(columns=['corr_120'])
        
        
        # 蜡烛上下影线
        df['candle_up_shadow_line'] = df['high'] - df[['open','close']].max(axis=1)
        df['candle_down_shadow_line'] = df[['open','close']].min(axis=1) - df['low']
        # Walliam上下影线
        df['Walliam_up_shadow_line'] = df['high'] - df['close']
        df['Walliam_down_shadow_line'] = df['close'] - df['low']
       
        # Donchian channel
        df['donchian_up_channel'] = df[['ticker','high']].groupby('ticker').rolling(120).max().reset_index().set_index('level_1')['high'].fillna(0)
        df['donchian_down_channel'] = df[['ticker','low']].groupby('ticker').rolling(120).min().reset_index().set_index('level_1')['low'].fillna(0)
        df['donchian_channel'] = df['donchian_up_channel']-df['donchian_down_channel']
        df = df.drop(columns = ['donchian_up_channel','donchian_down_channel'])

        # 传统反转因子
        df['traditional_reverse'] = df[['ticker','LastPx']].groupby('ticker').diff(60).fillna(0)

        # 整点cross因子
        df['price_cross_5'] = df[['ticker','price_ewma_5','LastPx']].groupby('ticker').apply(cross,'price_ewma_5').reset_index().set_index('level_1')[0]
        df['price_cross_20'] = df[['ticker','price_ewma_20','LastPx']].groupby('ticker').apply(cross,'price_ewma_20').reset_index().set_index('level_1')[0]
        df['price_cross_60'] = df[['ticker','price_ewma_60','LastPx']].groupby('ticker').apply(cross,'price_ewma_60').reset_index().set_index('level_1')[0]
        
        # 波动率因子
        df['close_vol'] = df[['ticker','LastPx']].groupby('ticker').rolling(120).std().reset_index().set_index('level_1')['LastPx'].fillna(0)
        df['volatility_mean'] = df[['ticker','close_vol']].groupby(['ticker']).rolling(120).mean().reset_index().set_index('level_1')['close_vol'].fillna(0)
        df['volatility_std'] = df[['ticker','close_vol']].groupby(['ticker']).rolling(120).std().reset_index().set_index('level_1')['close_vol'].fillna(0)
        df['vol_UID'] = df['volatility_std']+df['volatility_mean']
        df = df.drop(columns = ['volatility_mean','volatility_std'])
        
        # 信息分布因子
        df['volume_z'] = (df[['ticker','volume']].groupby('ticker').rolling(120).std().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['volume']).fillna(0)
        df['net_ret_5'] = df[['ticker','volume_z','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_5,raw=True, engine='numba').reset_index().set_index('level_1')['volume_z'].fillna(0)
        df['net_ret_1'] = df[['ticker','volume_z','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_1,raw=True, engine='numba').reset_index().set_index('level_1')['volume_z'].fillna(0)
        # df['net_ret_5_1'] = df[['ticker','volume_z','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_5_1,raw=True, engine='numba').reset_index().set_index('level_1')['volume_z'].fillna(0)

        # 类比换手率因子 选择计算短期内成交量/长期成交量 同时考虑时间延迟的情况 并以此为基准进行分组计算组内平均收益
        # df['volume_ma_3'] = (df[['ticker','volume']].groupby('ticker').rolling(5).sum().reset_index().set_index('level_1')['volume']/df[['ticker','volume']].groupby('ticker').volume.rolling(60).sum().reset_index().set_index('level_1')['volume']).fillna(0)
        df['volume_ma_3_shift'] = df[['ticker','volume_ma_3']].groupby('ticker').shift(5).fillna(0)
        df['volume_net_ret_5_shift'] = df[['ticker','volume_ma_3_shift','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_5,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3_shift'].fillna(0)
        df['volume_net_ret_1_shift'] = df[['ticker','volume_ma_3_shift','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_1,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3_shift'].fillna(0)
        # df['volume_net_ret_1_5_shift'] = df[['ticker','volume_ma_3_shift','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_1_5,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3_shift'].fillna(0)
        df['volume_net_ret_5'] = df[['ticker','volume_ma_3','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_5,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3'].fillna(0)
        df['volume_net_ret_1'] = df[['ticker','volume_ma_3','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_1,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3'].fillna(0)
        # df['volume_net_ret_1_5'] = df[['ticker','volume_ma_3','log_return']].groupby('ticker').rolling(120, method='table').apply(new_ret_1_5,raw=True, engine='numba').reset_index().set_index('level_1')['volume_ma_3'].fillna(0)

        df['smart'] = np.abs(df['log_return'])/np.sqrt(df['volume']+1)

        # 成交量博弈-收益率因子    
        df['volume_return'] = df[['ticker','vwap_log_return','volume']].groupby('ticker').rolling(120, method='table').apply(volume_return,raw=True, engine='numba').reset_index().set_index('level_1')['volume'].fillna(0)
        df['volume_return_mean'] = df[['ticker','volume_return']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['volume_return'].fillna(0)
        df = df.drop(columns=['volume_return'])

        # 成交量博弈-日内相对位置因子
        df['vwap_max'] = df[['ticker','vwap']].groupby('ticker').rolling(120).max().reset_index().set_index('level_1')['vwap'].fillna(0)
        df['vwap_min'] = df[['ticker','vwap']].groupby('ticker').rolling(120).min().reset_index().set_index('level_1')['vwap'].fillna(0)
        df['rank_location'] = df['vwap']/df['vwap_min']-1 + df['vwap']/df['vwap_max']-1
        df['rank_location'].replace(np.inf,0, inplace=True)
        
        # 振幅博弈因子
        df['amplitude'] = (df['high'] - df['low'])/df['close']
        df['amplitude_return'] = df[['ticker','vwap_log_return','amplitude']].groupby('ticker').rolling(120, method='table').apply(amplitude_return,raw=True, engine='numba').reset_index().set_index('level_1')['amplitude'].fillna(0)
        df['amplitude_return_mean'] = df[['ticker','amplitude_return']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['amplitude_return'].fillna(0)
        df = df.drop(columns = ['amplitude_return'])
        
        # weighted_ask_prc weighted_bid_prc在五档内的位置
        tmp_ask = np.array(df[['OfferPx1','OfferPx2','OfferPx3','OfferPx4','OfferPx5']])
        tmp_ask_abs = tmp_ask - np.array(df['weighted_ask_prc']).reshape(-1,1)
        df['weighted_ask_prc_location'] = np.argmin(np.abs(tmp_ask_abs),axis=1)+1

        tmp_bid = np.array(df[['BidPx1','BidPx2','BidPx3','BidPx4','BidPx5']])
        tmp_bid_abs = tmp_bid - np.array(df['weighted_bid_prc']).reshape(-1,1)
        df['weighted_bid_prc_location'] = np.argmin(np.abs(tmp_bid_abs),axis=1)+1
         

        # 成交量反转因子
        df['volume_log'] = np.log(df.volume+1)
        df['volume_mean'] = df[['ticker','volume_log']].groupby('ticker').rolling(120).mean().reset_index().set_index('level_1')['volume_log']
        df['volume_std'] = df[['ticker','volume_log']].groupby('ticker').rolling(120).std().reset_index().set_index('level_1')['volume_log']
        conditions = [
            (df['volume_log'] > df['volume_mean']+2*df['volume_std'])
        ]

        choices = [1]

        df['volume_reverse'] = np.select(conditions, choices, default=0)
        
        # 每个不同的标的拟合出来的k不太一样 为了简便 就选k=0.1
        # 作为反转因子 当成交量被放大且上涨时 标记为-1 之后按照-exp(-kt) 若下跌 则标记为1 之后按照exp(-kt)
        df['price_ewma5_diff'] = df[['ticker','price_ewma_5']].groupby('ticker').diff().fillna(0)
        df['reverse'] = df['volume_reverse']*-np.sign(df['price_ewma5_diff'])
        df['reverse_factor'] = df[['ticker','reverse']].groupby('ticker',group_keys=False).apply(cal_reverse_factor)
        df = df.drop(columns = ['volume_log','volume_mean', 'volume_std', 'volume_reverse',])
         
        # apply transform
        df_today = df[df.TradeDate==int(date)].reset_index(drop=True)
        # print(df_today.bp_rank)
        y_pred_rolling = self.__rolling_model__.predict(df_today)        
        y_pred = y_pred_rolling    

        df_today['pred_return'] = y_pred
        
        df_all = pd.concat([self.pred_return,df_today],axis=0,ignore_index=True)

        df_all['pred_return_ewma_1'] = df_all.pred_return.ewm(halflife=1).mean()
    
        df_all['upthreshold'] = 0.00022
        df_all['downthreshold'] = -0.00022

        conditions = [
            (df_all['pred_return_ewma_1'] > df_all['upthreshold']),             # 条件1：做多
            (df_all['pred_return_ewma_1'] < df_all['downthreshold']),         # 条件2：做空
            (df_all['pred_return_ewma_1'] <= df_all['upthreshold']) & 
            (df_all['pred_return_ewma_1'] >= df_all['downthreshold']),           
        ]

        choices = [1, -1, 0]  # 分别对应条件1、条件2、条件3的标记

        df_all['position'] = np.select(conditions, choices, default=0)

        df_all['signal'] = df_all['position'].diff().fillna(0)

        df_all.to_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/pred_return.csv',index=False)

        for code in df_all.ticker.unique():
            df_position = df_all[df_all.ticker==code].reset_index(drop=True)
            print(f'time:{df_position["minute"].values[-1]} pred_return:{round(df_position["pred_return"].values[-1]*10000,3)} pred_return_ewma_1:{round(df_position["pred_return_ewma_1"].values[-1]*10000,3)}')

            # 计算仓位 
            if time>1456:
                position = 0
            else:
                position = df_position['position'].values[-1]*int(self.__trdUnit__/df_position['LastPx'].values[-1])
            signal = df_position['signal'].values[-1]

            print(f'设定仓位{position}')
            #读取当前仓位
            curPos = context.stra_get_position(code)
            print(f'当前仓位{curPos}')
            # 记录最新要添加的仓位
            if len(df_position) == 1:
                if (position!=0) & (curPos==0):
                    self.position[code][self.current_time] = position
            else:
                if (position!=0) & ((df_position['position'].values[-2]==0) | (curPos==0)):
                    self.position[code][self.current_time] = position
            
            # 如果信号连续 选择更新最新的时间
            print('判断信号连续 时间更新')
            print(f'历史信号-1 {df_position["position"].values[-1]}')
            if len(df_position) != 1:
                print(f'历史信号-2 {df_position["position"].values[-2]}')
            if len(df_position) == 1:
                if  (position!=0):
                    self.position[code][self.current_time] = self.position[code][list(self.position[code].keys())[-1]]
                    self.position[code].pop(list(self.position[code].keys())[-2])
            else:
                if (df_position['position'].values[-1]==df_position['position'].values[-2]) & (position!=0):
                    self.position[code][self.current_time] = self.position[code][list(self.position[code].keys())[-1]]
                    self.position[code].pop(list(self.position[code].keys())[-2])

            # 实时更新仓位 若仓位超时 则删除：
            for i in list(self.position[code].keys()):
                if minute_to_index()[self.current_time]-minute_to_index()[i]>=5:
                    self.position[code].pop(i)
            print(self.position[code])
            all_position = sum(self.position[code].values())
            
            context.set_position(code, all_position)
            context.stra_log_text(f'{context.stra_get_time()} set {code} position {all_position}')       


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
            self.tick.write_csv(f'{self.__path__}daily_tick/{context.stra_get_date()}/tick.csv')
        # print(self.tick)
        
        return
    
    def on_session_end(self, context: CtaContext, curTDate: int):
        print('on session end')
        return 
    def on_session_begin(self, context: CtaContext, curTDate: int):
        print('on session begin')
        return 
    
    
    
    