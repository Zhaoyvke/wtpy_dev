# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:50:47 2023

@author: Kai
"""
import WonderWizOption as wwo
import pandas as pd
import polars as pl
import numpy as np

from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import json
import pymysql
from wtpy import BaseIndexWriter

def month_in_name(name):
    name = name.split('月')[0]
    if '购' in name:
        month = int(name.split('购')[-1])
    else:
        month = int(name.split('沽')[-1])
    return month

def simpleMargin(df):
    # 计算单腿保证金
    s0 = df['underlyingPrice']
    k = df['strike']
    optionType = df['type']
    c = df['close']
    multiplier = 10000
    if optionType in ['c', 'C', 'Call', 'call']:  # call
        return (c + max(0.12 * s0 - max(k - s0, 0), 0.07 * s0)) * multiplier
    elif optionType in ['p', 'P', 'Put', 'put']:
        return min(c + max(0.12 * s0 - max(s0 - k, 0), 0.07 * k), k) * multiplier
    else:
        raise TypeError('Illegal ValueError')

def compute_margin(S0, K, cpflag, V, multiplier):
    if cpflag in ['c', 'C', 'Call', 'call']:  # call
        return (V + max(0.12*S0-max(K-S0,0), 0.07*S0)) * multiplier
    elif cpflag in ['p','P','Put','put']:   # put
        return min(V + max(0.12*S0 - max(S0-K,0), 0.07*K), K) * multiplier
    else:
        raise TypeError('Illegal ValueError')

def add_ETF(option):
    tmp = option.split('.')
    return tmp[0] + '.ETF.' + tmp[-1]

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
    Mysql模拟的交易记录写入器
    '''    
    def __init__(self, host, port, user, pwd, dbname, sqlfmt):
        self.__db_conn__ = pymysql.connect(host=host, user=user, passwd=pwd, db=dbname, port=port)
        self.__sql_fmt__ = sqlfmt
        
    def write_trade(self, STRATEGY, TIME, CODE, QTY):
        # 需要确认一下用什么形式。
        sql = self.__sql_fmt__.replace('$STRATEGY', STRATEGY).replace('$TIME', TIME).replace('$CODE', CODE).replace('$QTY', QTY)
        curConn = self.__db_conn__
        curBase = curConn.cursor()
        curBase.execute(sql)
        curConn.commit()

class MySQLPositionWriter(BaseIndexWriter):
    def __init__(self, host, port, user, pwd, dbname, sqlfmt):
        self.__db_conn__ = pymysql.connect(host=host, user=user, passwd=pwd, db=dbname, port=port)
        self.__sql_fmt__ = sqlfmt
    
    def write_position(self, STRATEGY, DATE, CODE, QTY):
        sql = self.__sql_fmt__.replace('$STRATEGY', STRATEGY).replace('$DATE', DATE).replace('$CODE', CODE).replace('$QTY', QTY)
        curConn = self.__db_conn__
        curBase = curConn.cursor()
        curBase.execute(sql)
        curConn.commit()

class term_structure(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', underlying='510050', period='m5', start_fund=1000000):
        BaseCtaStrategy.__init__(self, name)
        self.exchg = exchg                  # 交易所
        self.underlying = underlying        # 标的
        self.period = period                # 数据周期
        self.margin = 0                     # 保证金
        self.margin_ratio = 0.4             # 最大使用保证金比例，
        self.start_fund = start_fund        # 初始资金
        
        
        self.lots = 0                       # 交易几组spread
        self.und_price = 0                  # 标的价格
        self.trading_contracts = None
        self.contractInfo = None            
        self.mySQLTradeWriter = MySQLTradeWriter(   host = '106.14.221.29',
                                                    port = 5308,
                                                    user = 'opter',
                                                    pwd = 'wuzhi',
                                                    dbname = 'wz_optnew',
                                                    sqlfmt = 'REPLACE simulation_trade_record (strategy, time, code, qty) VALUES (\'$STRATEGY\', \'$TIME\', \'$CODE\', \'$QTY\')')
        self.mySQLPositionWriter = MySQLPositionWriter( host = '106.14.221.29',
                                                        port = 5308,
                                                        user = 'opter',
                                                        pwd = 'wuzhi',
                                                        dbname = 'wz_optnew',
                                                        sqlfmt = 'REPLACE simulation_eod_position (strategy, date, code, qty) VALUES (\'$STRATEGY\', \'$DATE\', \'$CODE\', \'$QTY\')')
        
    def get_trading_dates(self):
        with open('./XSHG_Trading_Dates.txt') as f:
            dates = f.read().split('\n')
        self.trading_dates = dict(zip(dates,range(len(dates))))

    def days_between(self, today, maturity):
        # today fmt: 20230515
        # maturity fmt： 20230524
        today = str(today)
        maturity = str(maturity)
        if today > maturity:
            return 0
        else:
            return 1 + self.trading_dates[maturity] - self.trading_dates[today] # 如果当日是到期日，返回1

    def on_init(self, context: CtaContext):
        print('Term Structure Started')
        self.__ctx__ = context
        self.get_trading_dates()       # 国内市场的交易日，用于计算到期时间
        # 订阅ETF做主K线驱动
        context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1, isMain=True)            

    def fit_vinf(self, iv, ttm, method='log'):
        if method == 'sqrt':
            func = lambda x: 1/np.sqrt(x)
        elif method == 'log':
            func = lambda x: -np.log(x)
        iv = np.array(iv)
        x = np.array(ttm)
        if x[0] == 0:
            x = x[1:]
            iv = iv[1:]
        x = np.hstack([np.ones((len(x),1)),x.reshape(len(x),1)])
        param,_,_,_ = np.linalg.lstsq(x,iv,rcond=None)
        err = iv - (x@param)        
        return param, err    

    def on_session_begin(self, context: CtaContext, curDate: int):
        S0 = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        # 更新今天可交易品种
        self.avail_contracts = context.stra_get_codes_by_underlying(f'{self.exchg}.{self.underlying}')   
        # 按月份分类下,分成几个Panel
        options = {}
        for contract in self.avail_contracts:
            _contract_info = context.stra_get_contractinfo(contract)
            this_option = {}
            this_option['code'] = _contract_info.code
            this_option['stdCode'] = _contract_info.stdCode # SSE.10004679
            this_option['name'] = _contract_info.name
            this_option['exchg'] = _contract_info.exchg
            this_option['maturity'] = _contract_info.expireDate
            this_option['month'] = str(this_option['maturity'])[2:6]
            this_option['K'] = _contract_info.strikePrice
            this_option['cpflag'] = 'C' if '购' in _contract_info.name else 'P'
            this_option['multiplier'] = 10000
            this_option['ttm'] = self.days_between(str(context.stra_get_date()), int(this_option['maturity']))/240
            options[_contract_info.code] = this_option
        self.options = pd.DataFrame(options).T.reset_index()
        # print(self.options)
        S0 = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        self.ttms = sorted(self.options['ttm'].unique())        
        trading_contracts = dict()      # 拿来交易的对象?应该只需要选出ATM的就行
        atm_puts = dict()       # ttm -> put
        atm_calls = dict()      # ttm -> call
        atm_options = dict()
        self.ATM_info = pd.DataFrame(index=np.arange(len(self.ttms)),columns=['F','K','C','P','iv_c','iv_p','iv','margin_c','margin_p','margin','vega','theta','delta_c','delta_p','pos_c','pos_p','ttm'])
        for i,ttm in enumerate(self.ttms):   # 取出当月所有的不同到期日
            trading_contracts[i] = self.options[self.options['ttm']==ttm]    # 某一个月份的数据
            atm_options[i] = trading_contracts[i][(trading_contracts[i]['K']-S0) == min(trading_contracts[i]['K']-S0)]
            # 选出ATM的期权            
            atm_options[i].loc[atm_options[i].index,'trading_code'] = ['.ETF.'.join(x.split('.')) for x in atm_options[i]['stdCode'].values]
            atm_options[i].loc[atm_options[i].index,'S0'] = S0
            atm_options[i].loc[:,'margin'] = 1
            atm_options[i].loc[:,'price'] = 0
            for idx in atm_options[i].index:
                S0 = atm_options[i].loc[idx,'S0']
                K = atm_options[i].loc[idx,'K']
                cpflag = atm_options[i].loc[idx,'cpflag']
                multiplier = atm_options[i].loc[idx,'multiplier']                
                V = context.stra_get_bars(atm_options[i].loc[idx,'trading_code'], self.period, 1).closes[0]
                atm_options[i].loc[idx,'price'] = V
                atm_options[i].loc[idx,'margin'] = compute_margin(S0, K, cpflag, V, multiplier)               
            # 选出C,P，应该只有一个
            atm_calls[i] = atm_options[i][atm_options[i]['cpflag']=='C']
            atm_puts[i] = atm_options[i][atm_options[i]['cpflag']=='P']
            atm_calls[i].reset_index(inplace=True)
            atm_puts[i].reset_index(inplace=True)
            # 计算implied volatility
            C, P = atm_calls[i]['price'].values[0], atm_puts[i]['price'].values[0]
            K = atm_calls[i]['K'].values[0]
            S = atm_calls[i]['S0'].values[0]
            # F = C + K - P
            F = S*K/(S-C+P)
            r = np.log((C-P)/(F-K)) / ttm            
            self.ATM_info.loc[i,['S','F','K','C','P','r']] = [S,F,K,C,P,r]
            if ttm > 0:
                self.ATM_info.loc[i, 'ttm']      = ttm
                self.ATM_info.loc[i, 'iv_c']     = wwo.blsImpv(S, K, r, ttm, C, cpflag='C')
                self.ATM_info.loc[i, 'iv_p']     = wwo.blsImpv(S, K, r, ttm, P, cpflag='P')
                self.ATM_info.loc[i, 'iv']       = 0.5*(self.ATM_info.loc[i, 'iv_c'] + self.ATM_info.loc[i, 'iv_p'])
                self.ATM_info.loc[i, 'delta_c']  = wwo.blsDelta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'], cpflag='C')
                self.ATM_info.loc[i, 'delta_p']  = wwo.blsDelta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'], cpflag='P')
                self.ATM_info.loc[i, 'margin_c'] = atm_calls[i]['margin'][0]
                self.ATM_info.loc[i, 'margin_p'] = atm_puts[i]['margin'][0]
                self.ATM_info.loc[i, 'pos_c']    = np.abs(self.ATM_info.loc[i, 'delta_p'])
                self.ATM_info.loc[i, 'pos_p']    = np.abs(self.ATM_info.loc[i, 'delta_c'])
                self.ATM_info.loc[i, 'vega']     = wwo.blsVega(S, K, r, ttm, self.ATM_info.loc[i, 'iv'])*(self.ATM_info.loc[i, 'pos_c'] + self.ATM_info.loc[i, 'pos_p'])
                self.ATM_info.loc[i, 'theta']    = wwo.blsTheta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'])*(self.ATM_info.loc[i, 'pos_c'] + self.ATM_info.loc[i, 'pos_p'])
                self.ATM_info.loc[i, 'delta']    = self.ATM_info.loc[i, 'pos_p']*self.ATM_info.loc[i,'delta_p'] + self.ATM_info.loc[i, 'pos_c']*self.ATM_info.loc[i,'delta_c']                
                self.ATM_info.loc[i, 'margin']   = self.ATM_info.loc[i, 'pos_c']*self.ATM_info.loc[i, 'margin_c'] + self.ATM_info.loc[i, 'pos_p']*self.ATM_info.loc[i, 'margin_p']
        # 拟合        
        _, err = self.fit_vinf(self.ATM_info['iv'].astype(float).values, np.array(self.ttms), method='sqrt')        


    def on_calculate(self, context: CtaContext):
        # 仅在1455交易
        # 标的的最新价格
        S0 = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        self.ttms = sorted(self.options['ttm'].unique())        
        trading_contracts = dict()      # 拿来交易的对象?应该只需要选出ATM的就行
        atm_puts = dict()       # ttm -> put
        atm_calls = dict()      # ttm -> call
        atm_options = dict()
        self.ATM_info = pd.DataFrame(index=np.arange(len(self.ttms)),columns=['F','K','C','P','iv_c','iv_p','iv','margin_c','margin_p','margin','vega','theta','delta_c','delta_p','pos_c','pos_p','ttm'])
        for i,ttm in enumerate(self.ttms):   # 取出当月所有的不同到期日
            trading_contracts[i] = self.options[self.options['ttm']==ttm]    # 某一个月份的数据
            atm_options[i] = trading_contracts[i][(trading_contracts[i]['K']-S0) == min(trading_contracts[i]['K']-S0)]
            # 选出ATM的期权            
            atm_options[i].loc[atm_options[i].index,'trading_code'] = ['.ETF.'.join(x.split('.')) for x in atm_options[i]['stdCode'].values]
            atm_options[i].loc[atm_options[i].index,'S0'] = S0
            atm_options[i].loc[:,'margin'] = 1
            atm_options[i].loc[:,'price'] = 0
            for idx in atm_options[i].index:
                S0 = atm_options[i].loc[idx,'S0']
                K = atm_options[i].loc[idx,'K']
                cpflag = atm_options[i].loc[idx,'cpflag']
                multiplier = atm_options[i].loc[idx,'multiplier']                
                V = context.stra_get_bars(atm_options[i].loc[idx,'trading_code'], self.period, 1).closes[0]
                atm_options[i].loc[idx,'price'] = V
                atm_options[i].loc[idx,'margin'] = compute_margin(S0, K, cpflag, V, multiplier)               
            # 选出C,P，应该只有一个
            atm_calls[i] = atm_options[i][atm_options[i]['cpflag']=='C']
            atm_puts[i] = atm_options[i][atm_options[i]['cpflag']=='P']
            atm_calls[i].reset_index(inplace=True)
            atm_puts[i].reset_index(inplace=True)
            # 计算implied volatility
            C, P = atm_calls[i]['price'].values[0], atm_puts[i]['price'].values[0]
            K = atm_calls[i]['K'].values[0]
            S = atm_calls[i]['S0'].values[0]
            # F = C + K - P
            F = S*K/(S-C+P)
            r = np.log((C-P)/(F-K)) / ttm            
            self.ATM_info.loc[i,['S','F','K','C','P','r']] = [S,F,K,C,P,r]
            if ttm > 0:
                self.ATM_info.loc[i, 'ttm']      = ttm
                self.ATM_info.loc[i, 'iv_c']     = wwo.blsImpv(S, K, r, ttm, C, cpflag='C')
                self.ATM_info.loc[i, 'iv_p']     = wwo.blsImpv(S, K, r, ttm, P, cpflag='P')
                self.ATM_info.loc[i, 'iv']       = 0.5*(self.ATM_info.loc[i, 'iv_c'] + self.ATM_info.loc[i, 'iv_p'])
                self.ATM_info.loc[i, 'delta_c']  = wwo.blsDelta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'], cpflag='C')
                self.ATM_info.loc[i, 'delta_p']  = wwo.blsDelta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'], cpflag='P')
                self.ATM_info.loc[i, 'margin_c'] = atm_calls[i]['margin'][0]
                self.ATM_info.loc[i, 'margin_p'] = atm_puts[i]['margin'][0]
                self.ATM_info.loc[i, 'pos_c']    = np.abs(self.ATM_info.loc[i, 'delta_p'])
                self.ATM_info.loc[i, 'pos_p']    = np.abs(self.ATM_info.loc[i, 'delta_c'])
                self.ATM_info.loc[i, 'vega']     = wwo.blsVega(S, K, r, ttm, self.ATM_info.loc[i, 'iv'])*(self.ATM_info.loc[i, 'pos_c'] + self.ATM_info.loc[i, 'pos_p'])
                self.ATM_info.loc[i, 'theta']    = wwo.blsTheta(S, K, r, ttm, self.ATM_info.loc[i, 'iv'])*(self.ATM_info.loc[i, 'pos_c'] + self.ATM_info.loc[i, 'pos_p'])
                self.ATM_info.loc[i, 'delta']    = self.ATM_info.loc[i, 'pos_p']*self.ATM_info.loc[i,'delta_p'] + self.ATM_info.loc[i, 'pos_c']*self.ATM_info.loc[i,'delta_c']                
                self.ATM_info.loc[i, 'margin']   = self.ATM_info.loc[i, 'pos_c']*self.ATM_info.loc[i, 'margin_c'] + self.ATM_info.loc[i, 'pos_p']*self.ATM_info.loc[i, 'margin_p']
        # 拟合        
        _, err = self.fit_vinf(self.ATM_info['iv'].astype(float).values, np.array(self.ttms), method='sqrt')        

        # 交易判断
        if not context.stra_get_time()>=1450:     return
        # 确认买卖合约
        pos = np.zeros(len(self.ttms))
        MM = np.argmax(err[1:])+1   # 最被低估的
        mm = np.argmin(err[1:])+1   # 最被高估的
        pos[MM] = -1*np.abs(self.ATM_info.loc[mm, 'vega'])
        pos[mm] =  1*np.abs(self.ATM_info.loc[MM, 'vega'])
        for i in range(len(self.ttms)):
            if np.isnan(pos[i]): 
                pos[i] = 0
        margin = sum([np.abs(pos[i]*self.ATM_info.loc[i, 'margin']) for i in range(len(self.ttms))])
        self.lots = np.floor(self.margin_ratio * self.start_fund / margin)
        new_positions = dict()
        for i in range(len(self.ttms)):
            if np.isnan(pos[i]):
                print('----------------------------------')
                print(pos[i])
            elif pos[i] == 0:
                continue
            elif pos[i] > 0:
                # add call and put 
                new_positions[atm_calls[i]['trading_code'][0]] = np.floor(self.lots*pos[i]*self.ATM_info.loc[i, 'pos_c'])
                new_positions[atm_puts[i]['trading_code'][0]] = np.floor(self.lots*pos[i]*self.ATM_info.loc[i, 'pos_p'])
            elif pos[i] < 0:
                new_positions[atm_calls[i]['trading_code'][0]] = np.floor(self.lots*pos[i]*self.ATM_info.loc[i, 'pos_c'])
                new_positions[atm_puts[i]['trading_code'][0]] = np.floor(self.lots*pos[i]*self.ATM_info.loc[i, 'pos_p'])
        self.und_price = S0
        if len(new_positions) != 4:
            print(pos)            
            # raise ValueError('less than 4 option chose')
            return
        for contract, qty in new_positions.items():
            if np.isnan(qty):                
                return
                
        # 先清空所有position,并记录交易，存放在一个大文件里面，然后加一栏日期
        positions = context.stra_get_all_position()
        print(positions)        
        if len(positions) > 0:
            for contract, _pos in positions.items():           
                if contract in new_positions.keys():
                    continue
                context.stra_log_text(f'Set {contract}\'s position to 0.')
                print(f'Set {contract}\'s position to 0.')
                context.stra_set_position(contract, 0)
                # _STRATEGY = f'{self.name}_sim'
                # _TIME = f'{context.stra_get_date()}{context.stra_get_time()}'
                # _CODE = f'{contract}'
                # _LOT = f'-{_pos}'
                # self.mySQLTradeWriter.write_trade(_STRATEGY, _TIME, _CODE, _LOT)
                
        # 再添加新持仓                
        for contract, lot in new_positions.items():
            print(f'Set {contract}\'s position to {lot}.')
            context.stra_set_position(contract, lot)
            context.stra_log_text(f'Set {contract}\'s position to {lot}.')
            # _STRATEGY = f'{self.name}_sim'
            # _TIME = f'{context.stra_get_date()}{context.stra_get_time()}'
            # _CODE = f'{contract}'
            # _QTY = f'{lot}'
            # self.mySQLTradeWriter.write_trade(_STRATEGY, _TIME, _CODE, _QTY)                
        print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')       
        return None

    def on_session_end(self, context: CtaContext, curDate: int):        
        # 输出日末持仓
        for contract,pos in context.stra_get_all_position():
            _STRATEGY = f'{self.name}_sim'
            _DATE = f'{context.stra_get_date()}'
            _CODE = f'{contract}'
            _QTY = f'{pos}'
            self.mySQLPositionWriter.write_position(_STRATEGY, _DATE, _CODE, _QTY)
        return None

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return None
