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
from wtpy import BaseIndexWriter
from MySqlIdxWriter import *

class SkewStatArb(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', underlying='510050', period='m5', moneyness=0.05, start_fund=100000, margin_ratio = 0.5):
        BaseCtaStrategy.__init__(self, name)
        self.file_name = name                           #
        self.exchg = exchg                              # 交易所
        self.underlying = underlying                    # 标的 510050
        self.period = period                            # 数据周期
        self.moneyness= moneyness                       # Call和Put对应的moneyness，默认选0.95=1-moneyness的
        self.margin_ratio = margin_ratio                # 最大使用保证金比例，
        self.start_fund = start_fund                    # 初始资金
        
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

    def on_session_begin(self, context: CtaContext, curDate: int):
        # 订阅主力合约作为驱动
        self.underlying_code = f'{self.exchg}.ETF.{self.underlying}'
        print(f'subscribing bars: [{self.underlying_code}], {self.period}')
        self.S = context.stra_get_bars(self.underlying_code, self.period, 1, isMain = True).closes[0] 
        S = context.stra_get_bars(self.underlying_code, self.period, 1).closes[0]         
        # 获取当日所有可交易合约
        self.avail_contracts = context.stra_get_codes_by_underlying(f'{self.exchg}.{self.underlying}')
        options = {}
        for contract in self.avail_contracts:
            _contract_info = context.stra_get_contractinfo(contract)
            this_option = {}
            this_option['code'] = _contract_info.code               # 我猜是IO2309C4000这个形式
            stdCode = f'{self.exchg}.ETFO.{_contract_info.code}'
            this_option['stdCode'] = stdCode
            this_option['name'] = _contract_info.name
            this_option['exchg'] = _contract_info.exchg
            this_option['maturity'] = _contract_info.expireDate                     # 20230616
            this_option['month'] = str(this_option['maturity'])[2:6]                # 2306
            this_option['K'] = _contract_info.strikePrice                           # 3.0
            this_option['cpflag'] = 'C' if '购' in _contract_info.name else 'P'      # C/P
            # this_option['multiplier'] = _contract_info.multiplier                   # 在json中添加，应该默认是10000
            this_option['multiplier'] = 10000                   # 在json中添加，应该默认是10000
            this_option['ttm'] = self.days_between(str(context.stra_get_date()), int(this_option['maturity']))/240  # 年化后的
            options[this_option['stdCode']] = this_option                           # 记录期权信息
        self.options = pd.DataFrame(options).T                        # 转为pandas dataframe
        print(self.options)
        # 然后分类存储？
        self.ttms = sorted(self.options['ttm'].unique())
        self.Fs = np.ones(len(self.ttms))*S                                         # implied Forward
        self.rs = np.zeros(len(self.ttms))                                          # interest rate
        self.options_by_month = dict()                                              # 分月存储,值是一个DataFrame
        for i,ttm in enumerate(self.ttms):
            # 取出所有的K
            Ks = np.sort(np.unique(self.options[self.options['ttm']==ttm]['K']))
            _options = self.options[self.options['ttm']==ttm]                       # 选出一部分
            _options.reset_index(inplace=True)            
            _greeks = ['iv','delta','gamma','vega','theta','vanna','volga','margin']
            self.options_by_month[i] = pd.DataFrame(index=Ks, columns=['C_code','P_code','C_bid','C_ask','P_bid','P_ask','C_mid'] + [f'{cp}_{g}' for cp in ['C','P'] for g in _greeks])
            for idx in _options.index:
                K = _options.loc[idx,'K']
                cpflag = _options.loc[idx,'cpflag']
                code = _options.loc[idx, 'stdCode']
                # 同时订阅一下tick
                context.sub_ticks(code)
                tick = context.get_ticks(code, 1)                                   # 利用tick去更新报价
                self.options_by_month[i].loc[K,f'{cpflag}_code'] = code
                self.options_by_month[i].loc[K,f'{cpflag}_bid'] = tick['bid_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}_ask'] = tick['ask_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}_mid'] = np.round((tick['ask_price_0'] + tick['bid_price_0']) / 2,5)
                print(code, tick['bid_price_0'], tick['ask_price_0'])
            # 找到ATM的计算implied forward和implied rate
            print(self.options_by_month[i])
            # K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C_mid']-self.options_by_month[i]['P_mid']))]
            K_atm = Ks[np.argmin(np.abs(Ks-S))] # 离S最近的K当作是ATM
            C, P = self.options_by_month[i].loc[K_atm, 'C_mid'],  self.options_by_month[i].loc[K_atm, 'P_mid']            
            F = C+K-P
            r = 0
            self.Fs[i] = F
            self.rs[i] = r
    
    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return

    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        # 用于更新报价，这个stdCode是什么格式？
        _ttm = self.options.loc[stdCode,'ttm']
        # print(self.ttms, _ttm)
        _i = np.argmin(np.abs(np.array(self.ttms)-_ttm))
        # print(_i)
        _K = self.options.loc[stdCode, 'K']
        _cpflag = self.options.loc[stdCode, 'cpflag']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_bid'] = newTick['bid_price_0']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_ask'] = newTick['ask_price_0']
        self.options_by_month[_i].loc[_K, f'{_cpflag}_mid'] = (newTick['bid_price_0'] + newTick['ask_price_0']) / 2
        return 

    def refresh_greeks(self):
        # on_calculate触发时先做计算
        for i,ttm in enumerate(self.ttms):
            # 取出所有的K
            Ks = np.sort(np.unique(self.options[self.options['ttm']==ttm]['K']))            
            _greeks = ['iv','delta','gamma','vega','theta','vanna','volga']
            K_atm = Ks[np.argmin(np.abs(Ks-self.S))] # 离S最近的K当作是ATM
            F = K_atm + self.options_by_month[i].loc[K_atm, 'C_mid'] - self.options_by_month[i].loc[K_atm, 'P_mid']
            self.Fs[i] = F            
            for K in Ks:
                if K < F:   # 使用PIV
                    iv = wwo.blsImpv(F, K, 0, ttm, self.options_by_month[i].loc[K_atm, 'P_mid'], cpflag='P')
                else:
                    iv = wwo.blsImpv(F, K, 0, ttm, self.options_by_month[i].loc[K_atm, 'P_mid'], cpflag='C')                    
                for cpflag in ['C','P']:
                    # delta, gamma, vega, theta, vanna, volga
                    self.options_by_month[i].loc[K, f'{cpflag}_iv'] = iv
                    self.options_by_month[i].loc[K, f'{cpflag}_delta'] = wwo.blsDelta(F, K, 0, ttm, iv, cpflag=cpflag) 
                    self.options_by_month[i].loc[K, f'{cpflag}_gamma'] = wwo.blsGamma(F, K, 0, ttm, iv) 
                    self.options_by_month[i].loc[K, f'{cpflag}_vega'] = wwo.blsVega(F, K, 0, ttm, iv) 
                    self.options_by_month[i].loc[K, f'{cpflag}_theta'] = wwo.blsTheta(F, K, 0, ttm, iv, cpflag=cpflag) 
                    self.options_by_month[i].loc[K, f'{cpflag}_vanna'] = wwo.blsVanna(F, K, 0, ttm, iv) 
                    self.options_by_month[i].loc[K, f'{cpflag}_volga'] = wwo.blsDelta(F, K, 0, ttm, iv, cpflag=cpflag) 
                    self.options_by_month[i].loc[K, f'{cpflag}_margin'] = self.compute_margin(self.S, K, cpflag, self.options_by_month[i].loc[K, f'{cpflag}_mid'], 10000)                    
            print(f'Refresh greek panel')
            print(self.options_by_month[i])

    def extract_target_smile(self, S):
        # S, ETF price
        # 先按5天做划分，后续改成7-5做个过度？        
        ttm = self.ttms[0]*(self.ttms[0]>5/240) + self.ttms[1]*(self.ttms[0]<=5/240)
        idx = np.where(np.array(self.ttms)==ttm)[0][0]
        # print(idx)
        opts = self.options_by_month[idx]        
        # print(opts)
        _columns = ['code','mid','iv','delta','gamma','vega','theta','margin','pos']
        pos = pd.DataFrame(index=['P','C','S'], columns=_columns)   # 存放最新仓位
        # print(pos)
        # 找出对应的行权价
        Ks = self.options_by_month[idx].index
        K_atm = Ks[np.argmin(np.abs(Ks-S))] # 离S最近的K当作是ATM
        K_l = Ks[np.argmin(np.abs(Ks-S*(1-self.moneyness)))]    # 单一
        K_u = Ks[np.argmin(np.abs(Ks-S*(1+self.moneyness)))]    # 单一，转成组合？上下两个
        # 买put卖call        
        pos.loc['P',_columns[:8]] = opts.loc[K_l,[f'P_{col}' for col in _columns[:8]]].values
        pos.loc['C',_columns[:8]] = opts.loc[K_u,[f'C_{col}' for col in _columns[:8]]].values
        # print(pos)
        # print(self.margin_ratio, self.start_fund)
        # print(opts.loc[K_u,'C_margin'])
        # pos.loc['C','pos'] =  np.floor(- self.margin_ratio * self.start_fund / opts.loc[K_u,'C_margin'])       # 卖        
        # pos.loc['P','pos'] = np.floor(- pos.loc['C','pos'] * opts.loc[K_u,'C_gamma'] / opts.loc[K_l,'P_gamma']) # 买
        pos.loc['C','pos'] =  np.floor(- self.margin_ratio * self.start_fund / pos.loc['C','margin'])       # 卖        
        pos.loc['P','pos'] = np.floor(- pos.loc['C','pos'] * pos.loc['C','gamma'] / pos.loc['P','gamma']) # 买
        net_delta = np.floor(pos.loc['C','pos']*pos.loc['C','delta'] + pos.loc['P','pos']*pos.loc['P','delta'] * 100) * 100
        pos.loc['S','pos'] = -net_delta
        pos.loc['S','code'] = self.underlying_code
        print(pos)
        # 返回目标仓位的dataframe
        return pos

    def on_calculate(self, context: CtaContext):   
        print(f"{self.underlying} on calculate trigger, {context.stra_get_time()}")         
        self.S = context.stra_get_bars(self.underlying_code, self.period, 1).closes[0]          
        # 更新Greeks Panel的信息
        self.refresh_greeks()

        # 继续修改
        pos = self.extract_target_smile(self.S)
        new_positions = dict()
        trade_records = dict()
        for idx in pos.index:
            if pos.loc[idx, 'pos'] != 0:
                new_positions[pos.loc[idx,'code']] = pos.loc[idx, 'pos']
        positions = context.stra_get_all_position()
        print(positions)        
        print(new_positions)                  

        # 日末交易
        if context.stra_get_time() < 1451:              return
        if context.stra_get_time() > 1456:              return
        # 先清空所有position,并记录交易，存放在一个大文件里面，然后加一栏日期
        trading_time = str(context.stra_get_date()*10000 + context.stra_get_time())
        print(trading_time)                
        if len(positions) > 0:
            for contract, _pos in positions.items():           
                if contract in new_positions.keys():
                    continue
                context.stra_log_text(f'Set {contract}\'s position to 0.')                
                context.stra_set_position(contract, 0)
                if (_pos!=0):
                    trade_records[contract] = _pos * -1
        # 再添加新持仓                
        for contract, lot in new_positions.items():
            trade_records[contract] = lot
            if contract in positions.keys():
                old_pos = positions[contract]
                if (lot - old_pos != 0):
                    trade_records[contract] = lot - old_pos
            context.stra_set_position(contract, np.floor(lot))
            context.stra_log_text(f'Set {contract}\'s position to {lot}.')
        print(type(trading_time), trading_time)
        print(trade_records)
        print(type(self.file_name), self.file_name)
        self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)
        print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')       
        return None

    def on_session_end(self, context: CtaContext, curDate: int):
        # 输出日末持仓
        fake_pos = context.stra_get_all_position()
        real_pos = dict()
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k] = v
        self.sql_position_writer.write_pos(self.file_name, str(context.stra_get_date()), real_pos)        
        return 