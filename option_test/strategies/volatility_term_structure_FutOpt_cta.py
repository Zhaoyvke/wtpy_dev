# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:50:47 2023
    模拟盘代码
    股指期权的波动率期限结构
    使用主力合约作为驱动
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
from MySqlIdxWriter import *

class volatility_term_structure_v0(BaseCtaStrategy):
    def __init__(self, name, exchg='CFFEX', underlying='IM', period='m5', start_fund=1000000, margin_ratio = 0.5):
        BaseCtaStrategy.__init__(self, name)
        self.exchg = exchg                              # 交易所
        self.underlying = underlying                    # 标的 510050/IH
        self.period = period                            # 数据周期
        self.max_margin_ratio = margin_ratio            # 最大使用保证金比例，
        self.start_fund = start_fund                    # 初始资金
        self.optCode = 'MO'*(underlying=='IM') + 'IO'*(underlying=='IF') + 'HO'*(underlying=='IH')        
        trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        self.sql_trade_writer = MySQLTradeWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', trade_sqlfmt)
        pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        self.sql_position_writer = MySQLPositionWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', pos_sqlfmt)        
        self.get_trading_dates()

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

    def compute_margin(self, S0, K, cpflag, V, multiplier):
        if cpflag in ['c', 'C', 'Call', 'call']:  # call
            return (V + max(0.12*S0-max(K-S0,0), 0.07*S0)) * multiplier
        elif cpflag in ['p','P','Put','put']:   # put
            return min(V + max(0.12*S0 - max(S0-K,0), 0.07*K), K) * multiplier

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
        # 订阅主力合约作为驱动
        # F_hot = context.stra_get_bars(f'{self.exchg}.{self.underlying}.HOT', self.period, 1, isMain = True).closes[0] 
        # 获取当日所有可交易合约
        print("---------On Session Begin----------------")
        # print(context.stra_get_codes_by_product(f'{self.exchg}.{self.optCode}'))    # 所有月份的        
        # self.avail_opt = context.stra_get_codes_by_underlying(f'{self.exchg}.{self.optCode}2308')   # 特定月份的
        # print(self.avail_opt)

        self.avail_contracts = context.stra_get_codes_by_product(f'{self.exchg}.{self.optCode}')        
        print(self.avail_contracts)
        
        options = {}
        for contract in self.avail_contracts:
            _contract_info = context.stra_get_contractinfo(contract)            
            # print(_contract_info.__dict__)
            # contract -> CFFEX.MO.2307-C-5700
            this_option = {}
            this_option['code'] = _contract_info.code               # 我猜是IO2309C4000这个形式
            this_option['stdCode'] = f'{self.exchg}.{".".join(_contract_info.code.split("-"))}'
            this_option['name'] = _contract_info.name
            this_option['exchg'] = _contract_info.exchg
            this_option['maturity'] = _contract_info.expiredate                     # 20230616
            this_option['month'] = str(this_option['maturity'])[2:6]                # 2306
            this_option['K'] = _contract_info.strikePrice                           # 4000
            this_option['cpflag'] = 'C' if 'C' in _contract_info.code else 'P'      # C/P
            # this_option['multiplier'] = _contract_info.multiplier                   # 在json中添加，应该默认是100
            this_option['multiplier'] = 100
            this_option['ttm'] = self.days_between(str(context.stra_get_date()), int(this_option['maturity']))/240  # 年化后的
            options[this_option['stdCode']] = this_option                           # 记录期权信息            
        self.options = pd.DataFrame(options).T                                      # 转为pandas dataframe
        print(self.options)
        # 然后分类存储？
        self.ttms = sorted(self.options['ttm'].unique())
        self.Fs = np.ones(len(self.ttms))*0                                         # implied Forward
        self.rs = np.zeros(len(self.ttms))                                          # interest rate
        self.options_by_month = dict()                                              # 分月存储,值是一个DataFrame
        for i,ttm in enumerate(self.ttms):
            # 取出所有的K
            Ks = np.sort(np.unique(self.options[self.options['ttm']==ttm]['K']))
            _options = self.options[self.options['ttm']==ttm]                       # 选出一部分
            _options.reset_index(inplace=True)
            self.options_by_month[i] = pd.DataFrame(index=Ks, columns=['C_code','P_code','C_bid','C_ask','P_bid','P_ask','C_mid',])
            for idx in _options.index:
                K = _options.loc[idx,'K']
                cpflag = _options.loc[idx,'cpflag']
                # code = 'CFFEX.'+_options.loc[idx, 'stdCode'][:2]+'.'+'.'.join(_options.loc[idx, 'stdCode'][2:].split('-'))
                # code = 'CFFEX.'+'MO'+'.'.join(_options.loc[idx, 'stdCode'][2:].split('-'))
                code = _options.loc[idx, 'stdCode']
                print(code)
                # 同时订阅一下tick
                tick = context.get_ticks(code, 1)                                   # 利用tick去更新报价
                print(tick)
                # print(tick.__dict__)
                self.options_by_month[i].loc[K,f'{cpflag}_code'] = code
                self.options_by_month[i].loc[K,f'{cpflag}_bid'] = tick['bid_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}_ask'] = tick['ask_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}_mid'] = (tick['ask_price_0'] + tick['bid_price_0']) / 2
            # 找到ATM的计算implied forward和implied rate
            K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C_mid']-self.options_by_month[i]['P_mid']))]
            C, P = self.options_by_month[i].loc[K_atm, 'C_mid'],  self.options_by_month[i].loc[K_atm, 'P_mid']
            # F = F_hot*K_atm/(F_hot-C+P)
            F = C+K-P
            # r = np.log(F/F_hot)/ttm
            r = 0
            self.Fs[i] = F
            self.rs[i] = r        
        F_hot = context.stra_get_bars(f'{self.exchg}.{self.underlying}.{np.min(self.options["month"])}', self.period, 1, isMain = True).closes[0] 

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return None

    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        # 用于更新报价，这个stdCode是什么格式？
        print(f'On tick, stdCode: {stdCode}')
        _ttm = self.options.loc[stdCode,'ttm']
        _i = np.where(self.ttms==_ttm)[0][0]
        _K = self.options.loc[stdCode, 'K']
        _cpflag = self.options.loc[stdCode, 'cpflag']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_bid'] = newTick['bid_price_0']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_ask'] = newTick['ask_price_0']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_mid'] = (newTick['bid_price_0'] + newTick['ask_price_0']) / 2
        return 

    def extract_atm_infos(self):        
        # 更新下implied forward and implied rate 
        _columns = ['code','mid','delta','gamma','vega','theta','pos','margin']
        df = pd.DataFrame(index=self.ttms, columns=['IV','F'] + [f'{cl}_{cp}' for cl in _columns for cp in ['c','p']] + ['vega_combo','gamma_combo','theta_combo','margin_combo'])
        for i,ttm in enumerate(self.ttms):
            Ks = self.options_by_month[i].index
            K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C_mid'] - self.options_by_month[i]['C_mid']))]
            C, P = self.options_by_month[i].loc[K_atm, 'C_mid'],  self.options_by_month[i].loc[K_atm, 'P_mid']
            F = C+K_atm-P
            r = 0
            civ = wwo.blsImpv(F, K_atm, r, ttm, C)
            piv = wwo.blsImpv(F, K_atm, r, ttm, P, cpflag='P')
            iv = (civ + piv)/2
            delta_c = wwo.blsDelta(F, K_atm, r, ttm, iv)
            delta_p = wwo.blsDelta(F, K_atm, r, ttm, iv, cpflag='P')
            gamma_c = wwo.blsGamma(F, K_atm, r, ttm, iv)
            gamma_p = wwo.blsGamma(F, K_atm, r, ttm, iv)
            vega_c = wwo.blsVega(F, K_atm, r, ttm, iv)
            vega_p = wwo.blsVega(F, K_atm, r, ttm, iv)
            theta_c = wwo.blsTheta(F, K_atm, r, ttm, iv)
            theta_p = wwo.blsTheta(F, K_atm, r, ttm, iv)
            margin_c = self.compute_margin(F, K_atm, 'C', C, 10000)
            margin_p = self.compute_margin(F, K_atm, 'P', P, 10000)
            df.loc[ttm, ['F','mid_c','mid_p','IV']] = [F,C,P,iv]
            df.loc[ttm, ['delta_c','delta_p']] = [delta_c,delta_p]
            df.loc[ttm, ['gamma_c','gamma_p']] = [gamma_c,gamma_p]
            df.loc[ttm, ['vega_c','vega_p']] = [vega_c,vega_p]
            df.loc[ttm, ['theta_c','theta_p']] = [theta_c,theta_p]
            df.loc[ttm, ['margin_c','margin_p']] = [margin_c,margin_p]
            df.loc[ttm,['pos_c','pos_p']] = np.abs(delta_p), np.abs(delta_c)
            df.loc[ttm,'vega_combo'] = np.abs(delta_p)*vega_c + np.abs(delta_c)*vega_p
            df.loc[ttm,'gamma_combo'] = np.abs(delta_p)*gamma_c + np.abs(delta_c)*gamma_p
            df.loc[ttm,'theta_combo'] = np.abs(delta_p)*theta_c + np.abs(delta_c)*theta_p
            df.loc[ttm,'margin_combo'] = np.abs(delta_p)*margin_c + np.abs(delta_c)*margin_p            
        return df

    def on_calculate(self, context: CtaContext):    
        # 日末交易下
        if not context.stra_get_time()>=1450:
            return    
        F_hot = context.stra_get_bars(f'{self.exchg}.{self.underlying}.{np.min(self.options["month"])}', self.period, 1, isMain = True).closes[0] 
        df = self.extract_atm_infos()
        _, err = self.fit_vinf(df['IV'].values, np.array(self.ttms), method='sqrt')          
        # 交易判断        
        pos = np.zeros(len(self.ttms))  # 买入卖出月份标记,后续加一个组合优化？
        MM = np.argmax(err[1:])+1   # 最被低估的
        mm = np.argmin(err[1:])+1   # 最被高估的
        # Vega中性
        pos[MM] = -1*np.abs(df.loc[self.ttms[MM], 'vega_all'])
        pos[mm] =  1*np.abs(df.loc[self.ttms[MM], 'vega_all'])
        for i in range(len(self.ttms)):
            if np.isnan(pos[i]): 
                pos[i] = 0
        margin_total = sum([np.abs(pos[i]*df.loc[self.ttms[i], 'margin']) for i in range(len(self.ttms))])
        self.lots = self.margin_ratio * self.start_fund / margin_total        
        trade_records = {}
        # 计算新的仓位
        new_positions = dict()
        for i in range(len(self.ttms)):
            if np.isnan(pos[i]):
                print('----------------------------------')
                print(i,pos[i])
            elif pos[i] == 0:
                continue
            elif pos[i] > 0:
                # add call and put 
                new_positions[df.loc[df.index[i],'code_c']] = np.floor(self.lots*pos[i]*df.loc[df.index[i], 'pos_c'])
                new_positions[df.loc[df.index[i],'code_p']] = np.floor(self.lots*pos[i]*df.loc[df.index[i], 'pos_p'])                
            elif pos[i] < 0:
                new_positions[df.loc[df.index[i],'code_c']] = np.floor(self.lots*pos[i]*df.loc[df.index[i], 'pos_c'])
                new_positions[df.loc[df.index[i],'code_p']] = np.floor(self.lots*pos[i]*df.loc[df.index[i], 'pos_p'])                
        for contract, qty in new_positions.items():
            if np.isnan(qty):     
                # 意外出现nan就不交易           
                return
        if context.stra_get_time() < 1451:            return
        # 先清空所有position,并记录交易，存放在一个大文件里面，然后加一栏日期
        trading_time = str(context.stra_get_date()*10000 + context.stra_get_time())
        self.sql_trade_writer.write_trade(self.name, trading_time, trade_records)
        
        positions = context.stra_get_all_position()
        print(positions)        
        if len(positions) > 0:
            for contract, _pos in positions.items():           
                if contract in new_positions.keys():
                    continue
                context.stra_log_text(f'Set {contract}\'s position to 0.')                
                context.stra_set_position(contract, 0)
                trade_records[contract] = _pos * -1
                
        # 再添加新持仓                
        for contract, lot in new_positions.items():
            trade_records[contract] = lot
            if contract in positions.keys():
                old_pos = positions[contract]
                trade_records[contract] = lot - old_pos            
            context.stra_set_position(contract, np.floor(lot))
            context.stra_log_text(f'Set {contract}\'s position to {lot}.')
            
        print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')       
        return None

    def on_session_end(self, context: CtaContext, curDate: int):        
        # 输出日末持仓
        fake_pos = context.stra_get_all_position()
        real_pos = []
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k] = v
        self.sql_position_writer.write_pos(self.name, str(context.stra_get_date()), real_pos)        
        return 