# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:50:47 2023
    
@author: OptionTeam
"""

from OptionInfo import *
from StressTest import *
from SupplementaryFunc import *

import WonderWizOption as wwo
import pandas as pd
import polars as pl
import numpy as np

from wtpy import BaseCtaStrategy
from wtpy import CtaContext
from MySqlIdxWriter import *

class VolTS(BaseCtaStrategy):
    def __init__(self, name, exchg='CFFEX', optionCode='HO', period='m5', start_fund=1000000):
        BaseCtaStrategy.__init__(self, name)                       
        self.file_name = name
        self.exchg = exchg
        self.period = period
        if optionCode == 'MO' or optionCode == 'IM':
            self.optionCode = 'MO'
            self.underlying = 'IM'
        elif optionCode == 'IO' or optionCode == 'IF':
            self.optionCode = 'IO'
            self.underlying = 'IF'
        elif optionCode == 'HO' or optionCode == 'IH':
            self.optionCode = 'HO'
            self.underlying = 'IH'        
        self.und_price = 0
        self.trading_contracts = None
        self.contractInfo = None
        self.margin = 0
        self.margin_ratio = 0.4
        self.start_fund = start_fund
        self.lots = 0
        self.position_yesterday = []
        self.position_today = []
        self.long_month = ""
        self.short_month = ""
        trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        self.sql_trade_writer = MySQLTradeWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', trade_sqlfmt)
        pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        self.sql_position_writer = MySQLPositionWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', pos_sqlfmt)                
        self.get_trading_dates()
        self.traded = False
        print('__init__ finished')
        
    def on_init(self, context: CtaContext):
        print(f'Volatility term structure strategy {self.optionCode} started')
        self.__ctx__ = context
        und_bars = context.stra_get_bars(f'CFFEX.{self.underlying}.2307', self.period, 1, isMain=True)

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
        x = np.array(ttm)/240
        if x[0] == 0:
            x = x[1:]
            iv = iv[1:]
        x = np.hstack([np.ones((len(x),1)),x.reshape(len(x),1)])
        param,_,_,_ = np.linalg.lstsq(x,iv,rcond=None)
        err = iv - (x@param)        
        return param, err    

    def on_session_begin(self, context: CtaContext, curDate: int):
        # print('******************************************************************')
        print('[%s] on_session_begin' % (curDate))   
        # 获取当日所有可交易合约        
        self.avail_contracts = context.stra_get_codes_by_product(f'{self.exchg}.{self.optionCode}')        
        option_infos = {}
        for contract in self.avail_contracts:
            _contract_info = context.stra_get_contractinfo(contract)
            this_option = {}
            this_option['code'] = _contract_info.code               
            stdCode = f'{self.exchg}.{".".join(_contract_info.code.split("-"))}' 
            this_option['stdCode'] = stdCode
            this_option['name'] = _contract_info.name
            this_option['exchg'] = _contract_info.exchg
            this_option['maturity'] = _contract_info.expiredate                         # 20230616
            this_option['month'] = _contract_info.delivermonth                          # 06
            this_option['K'] = _contract_info.strikePrice                               # 3.0
            this_option['cpflag'] = _contract_info.optiontype                           # C/P
            # this_option['multiplier'] = _contract_info.multiplier                     # 在json中添加，应该默认是100
            this_option['multiplier'] = 100                                             # 在json中添加，应该默认是100
            this_option['ttm'] = self.days_between(str(context.stra_get_date()), int(this_option['maturity']))/240  # 年化后的
            option_infos[this_option['stdCode']] = this_option                               # 记录期权信息        

        self.option_infos = pd.DataFrame(option_infos).T                        # 转为pandas dataframe
        # 然后分类存储？
        S = 0
        self.ttms = sorted(self.option_infos['ttm'].unique())
        self.months = sorted(self.option_infos['month'].unique())
        self.Fs = np.ones(len(self.ttms))*S                                         # implied Forward
        self.rs = np.zeros(len(self.ttms))                                          # interest rate
        self.options_by_month = dict()                                              # 分月存储,值是一个DataFrame
        for i,ttm in enumerate(self.ttms):
            # 取出所有的K
            Ks = np.sort(np.unique(self.option_infos[self.option_infos['ttm']==ttm]['K']))
            _options = self.option_infos[self.option_infos['ttm']==ttm]                       # 选出一部分
            _options.reset_index(inplace=True)            
            # 实盘或者模拟盘
            columns=['C_code','P_code','C_bid','C_ask','C','P_bid','P_ask','P']            # C,P-> mid prices
            # 回测
            # columns = ['C_code','P_code','C','P']
            _greeks = ['iv','delta','gamma','vega','theta','vanna','volga','margin','pos']
            self.options_by_month[i] = pd.DataFrame(index=Ks, columns=columns + [f'{cp}_{g}' for cp in ['C','P'] for g in _greeks])
            for idx in _options.index:
                K = _options.loc[idx,'K']
                cpflag = _options.loc[idx,'cpflag']
                code = _options.loc[idx, 'stdCode']
                self.options_by_month[i].loc[K,f'{cpflag}_code'] = code                
                # 同时订阅一下tick（实盘或者模拟盘）
                context.sub_ticks(code)
                tick = context.get_ticks(code, 1)                                   # 利用tick去更新报价
                self.options_by_month[i].loc[K,f'{cpflag}_bid'] = tick['bid_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}_ask'] = tick['ask_price_0']
                self.options_by_month[i].loc[K,f'{cpflag}'] = np.round((tick['ask_price_0'] + tick['bid_price_0']) / 2,5)
            # 找到ATM的计算implied forward和implied rate            
            K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C']-self.options_by_month[i]['P']))]
            C, P = self.options_by_month[i].loc[K_atm, 'C'],  self.options_by_month[i].loc[K_atm, 'P']            
            F = C+K-P
            r = 0
            self.Fs[i] = F
            self.rs[i] = r
        print('[%s] on_session_begin Finished' % (curDate))
        # print('******************************************************************')
        
    def refresh_option_info(self, context: CtaContext):
        # 更新最新的价格信息
        # self.ttms, self.Fs, self.rs
        # 使用implied forward的话，只需ttms
        for i in self.options_by_month.keys():
            Ks = self.options_by_month[i].index
            for K in Ks:
                # 更新价格
                C_code = self.options_by_month[i].loc[K, 'C_code']
                P_code = self.options_by_month[i].loc[K, 'P_code']
                c_bar = context.stra_get_bars(C_code, self.period, 1)
                p_bar = context.stra_get_bars(P_code, self.period, 1)
                if not c_bar is None:
                    self.options_by_month[i].loc[K, 'C'] = c_bar.closes[0]
                if not p_bar is None:
                    self.options_by_month[i].loc[K, 'P'] = p_bar.closes[0]
            # 计算Greeks
            K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C']-self.options_by_month[i]['P']))]
            # implied forward
            F = K_atm + self.options_by_month[i].loc[K_atm,'C'] - self.options_by_month[i].loc[K_atm,'P']
            tau = self.ttms[i]
            for K in Ks:
                if K < F:   # use put
                    iv = wwo.blsImpv(F, K, 0, tau, self.options_by_month[i].loc[K,'P'], cpflag='P')
                else:
                    iv = wwo.blsImpv(F, K, 0, tau, self.options_by_month[i].loc[K,'C'])
                self.options_by_month[i].loc[K,'C_iv'] = iv
                self.options_by_month[i].loc[K,'P_iv'] = iv 
                self.options_by_month[i].loc[K,'C_delta'] = wwo.blsDelta(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_delta'] = wwo.blsDelta(F, K, 0, tau, iv, cpflag='P')
                self.options_by_month[i].loc[K,'C_gamma'] = wwo.blsGamma(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_gamma'] = wwo.blsGamma(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'C_vega'] = wwo.blsVega(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_vega'] = wwo.blsVega(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'C_theta'] = wwo.blsTheta(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_theta'] = wwo.blsTheta(F, K, 0, tau, iv, cpflag='P')
                self.options_by_month[i].loc[K,'C_vanna'] = wwo.blsVanna(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_vanna'] = wwo.blsVanna(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'C_volga'] = wwo.blsVolga(F, K, 0, tau, iv)
                self.options_by_month[i].loc[K,'P_volga'] = wwo.blsVolga(F, K, 0, tau, iv)
                # margin and position
                self.options_by_month[i].loc[K,'C_margin'] = 100000
                self.options_by_month[i].loc[K,'P_margin'] = 100000
                
    
    def extract_atm_iv(self):
        atm_ivs = np.zeros(len(self.ttms))
        for i in self.options_by_month.keys():
            Ks = self.options_by_month[i].index
            K_atm = Ks[np.argmin(np.abs(self.options_by_month[i]['C']-self.options_by_month[i]['P']))]
            atm_ivs[i] = self.options_by_month[i].loc[K_atm, 'C_iv']
        return atm_ivs

    def generate_new_position(self, err):
        # err 拟合的误差     
        # 获取当前仓位
        positions = self.__ctx__.stra_get_all_position()
        print(positions)   
        # 不交易第一个月的
        MM = np.argmax(err[1:])+1   # 最被高估的，卖出
        mm = np.argmin(err[1:])+1   # 最被低估的, 买入
        a = 1
        if len(positions) == 0 or (self.long_month != self.months[mm] and self.short_month != self.months[MM]):
            print("-------------Change All Position----------------------")
            self.long_month = self.months[mm]
            self.short_month = self.months[MM]
            # 找到对应月份的ATM合约，先计算卖出月份的合约组计算margin
            Ks = self.options_by_month[mm].index
            K_mm = Ks[np.argmin(np.abs(self.options_by_month[mm]['C']-self.options_by_month[mm]['P']))]
            # Make it delta neutral
            margin_p_mm = np.abs(self.options_by_month[mm].loc[K_mm,'C_delta']) * self.options_by_month[mm].loc[K_mm,'P_margin']
            margin_c_mm = np.abs(self.options_by_month[mm].loc[K_mm,'P_delta']) * self.options_by_month[mm].loc[K_mm,'C_margin']
            tmp_lots = self.margin_ratio * self.start_fund / (margin_p_mm + margin_c_mm)
            qty_p_mm = np.abs(np.floor(self.options_by_month[mm].loc[K_mm,'C_delta']*tmp_lots))  # 对应卖出PUT的手数
            qty_c_mm = np.abs(np.floor(self.options_by_month[mm].loc[K_mm,'P_delta']*tmp_lots))  # 对应卖出CALL的手数        
            code_c_mm = self.options_by_month[mm].loc[K_mm,'C_code']
            code_p_mm = self.options_by_month[mm].loc[K_mm,'P_code']
            # vega_mm = self.options_by_month[mm].loc[K_mm,'C_vega']*qty_c_mm + self.options_by_month[mm].loc[K_mm,'P_vega']*qty_p_mm
            theta_mm = self.options_by_month[mm].loc[K_mm,'C_theta']*qty_c_mm + self.options_by_month[mm].loc[K_mm,'P_theta']*qty_p_mm        
            # MM 的部分
            Ks = self.options_by_month[MM].index
            K_MM = Ks[np.argmin(np.abs(self.options_by_month[MM]['C']-self.options_by_month[MM]['P']))]        
            theta_MM = -self.options_by_month[MM].loc[K_MM,'C_theta']*self.options_by_month[MM].loc[K_MM,'P_delta'] + self.options_by_month[MM].loc[K_MM,'P_theta']*self.options_by_month[MM].loc[K_MM,'C_delta']
            tmp_lots2 = theta_mm / theta_MM
            qty_p_MM = -np.abs(np.floor(self.options_by_month[MM].loc[K_MM,'C_delta']*tmp_lots2))  # 对应买入PUT的手数
            qty_c_MM = -np.abs(np.floor(self.options_by_month[MM].loc[K_MM,'P_delta']*tmp_lots2))  # 对应买入CALL的手数                
            code_c_MM = self.options_by_month[MM].loc[K_MM,'C_code']
            code_p_MM = self.options_by_month[MM].loc[K_MM,'P_code']
            codes = [code_c_mm,code_p_mm,code_c_MM,code_p_MM]
            pos = pd.DataFrame([qty_c_mm,qty_p_mm,qty_c_MM,qty_p_MM],index=codes,columns=['qty'])
            pos = pos.groupby(pos.index).sum()
            return pos
        elif self.long_month == self.months[mm] and self.short_month == self.months[MM]:
            print("-------------Keep Same Position----------------------")
            # 不用换月
            codes,qtys = [],[]
            for code,qty in positions.items():
                if qty != 0:
                    codes.append(code)
                    qtys.append(qty)
            pos = pd.DataFrame(qtys,index=codes,columns=['qty'])
            return pos
        elif self.long_month != self.months[mm] and self.short_month == self.months[MM]:
            print("-------------Change Long Position----------------------")
            # 需要换做多的月份
            codes,qtys = [],[]
            for code,qty in positions.items():
                if qty < 0:
                    codes.append(code)
                    qtys.append(qty)
            # 使用theta中性对冲            
            K_MM = float(codes[-1].split('.')[-1])
            if 'C' in codes[0]:
                lots_MM_c, lots_MM_p = qtys
            else:
                lots_MM_p, lots_MM_c = qtys
            theta_MM = self.options_by_month[MM].loc[K_MM,'C_theta'] * lots_MM_c + self.options_by_month[MM].loc[K_MM,'P_theta'] * lots_MM_p
            # 找到对应月份的ATM合约，先计算卖出月份的合约组计算margin
            Ks = self.options_by_month[mm].index
            K_mm = Ks[np.argmin(np.abs(self.options_by_month[mm]['C']-self.options_by_month[mm]['P']))]
            # Make it delta neutral
            theta_mm = np.abs(self.options_by_month[mm].loc[K_mm,'C_theta']*self.options_by_month[mm].loc[K_mm,'P_delta']) + np.abs(self.options_by_month[mm].loc[K_mm,'P_theta']*self.options_by_month[mm].loc[K_mm,'C_delta'])
            lots_mm = theta_MM/theta_mm
            qty_p_mm = np.abs(np.floor(self.options_by_month[mm].loc[K_mm,'C_delta']*lots_mm))  # 对应卖出PUT的手数
            qty_c_mm = np.abs(np.floor(self.options_by_month[mm].loc[K_mm,'P_delta']*lots_mm))  # 对应卖出CALL的手数
            code_c_mm = self.options_by_month[mm].loc[K_mm,'C_code']
            code_p_mm = self.options_by_month[mm].loc[K_mm,'P_code']
            codes = codes + [code_c_mm, code_p_mm]
            qtys  = qtys  + [qty_c_mm,   qty_p_mm]
            pos = pd.DataFrame(qtys,index=codes,columns=['qty'])
            return pos            
        elif self.long_month == self.months[mm] and self.short_month != self.months[MM]:
            print("-------------Change Short Position----------------------")
            # 需要换做空的月份
            codes,qtys = [],[]
            for code,qty in positions.items():
                if qty > 0:
                    codes.append(code)
                    qtys.append(qty)
            # 使用theta中性对冲
            K_mm = float(codes[-1].split('.')[-1])
            if 'C' in codes[0]:
                lots_mm_c, lots_mm_p = qtys
            else:
                lots_mm_p, lots_mm_c = qtys
            # lots_mm = qtys[-1]
            theta_mm = self.options_by_month[mm].loc[K_mm,'C_theta'] * lots_mm_c + self.options_by_month[mm].loc[K_mm,'P_theta'] * lots_mm_p
            # 找到对应月份的ATM合约，先计算卖出月份的合约组计算margin
            Ks = self.options_by_month[MM].index
            K_MM = Ks[np.argmin(np.abs(self.options_by_month[MM]['C']-self.options_by_month[MM]['P']))]
            # Make it delta neutral
            theta_MM = np.abs(self.options_by_month[MM].loc[K_MM,'C_theta']*self.options_by_month[MM].loc[K_MM,'P_delta']) + np.abs(self.options_by_month[MM].loc[K_MM,'P_theta']*self.options_by_month[MM].loc[K_MM,'C_delta'])
            lots_MM = np.abs(theta_mm/theta_MM)
            qty_p_MM = np.abs(np.floor(self.options_by_month[MM].loc[K_MM,'C_delta']*lots_MM))  # 对应卖出PUT的手数
            qty_c_MM = np.abs(np.floor(self.options_by_month[MM].loc[K_MM,'P_delta']*lots_MM))  # 对应卖出CALL的手数
            code_c_MM = self.options_by_month[MM].loc[K_MM,'C_code']
            code_p_MM = self.options_by_month[MM].loc[K_MM,'P_code']
            codes = codes + [code_c_MM, code_p_MM]
            qtys  = qtys  + [-qty_c_MM, -qty_p_MM]
            pos = pd.DataFrame(qtys,index=codes,columns=['qty'])
            return pos

    def on_calculate(self, context: CtaContext):
        # print(f'[{context.stra_get_time()}] On Calculate Triggered!')
        # if (not context.stra_get_time()>1450) or (context.stra_get_time() > 1455):
        #     return
        if self.traded:
            return
        # 标的的最新价格
        self.refresh_option_info(context)
        # 取出ATM iv或者sigma0?
        atm_ivs = self.extract_atm_iv()
        _, err = self.fit_vinf(atm_ivs, np.array(self.ttms), method='log')
        # print(f'[ERR]: {err}')
        # 确认买卖合约        
        # 做个vega hedging 
        pos = self.generate_new_position(err)
        print('================Generated Positions============================')
        print(pos)
        # print(f"THETA: {ATM_info.loc[mm, 'theta']},{ATM_info.loc[MM, 'theta']}") # all less than 0 
        
        # 先清空所有position,并记录交易，存放在一个大文件里面，然后加一栏日期
        trading_time = str(context.stra_get_date()*10000 + context.stra_get_time())        
        trade_records = {}

        positions = context.stra_get_all_position()
        new_positions = {k:pos.loc[k,'qty'] for k in pos.index}        
        # print(positions)        
        if len(positions) > 0:
            for contract, _ in positions.items():
                if not contract in new_positions:
                    context.stra_set_position(contract, 0)
                    context.stra_log_text(f'Set {contract}\'s position to 0.')
                    # 添加交易记录
                    trade_records[contract] = int(-qty)

        # 再添加新持仓                
        for contract, qty in new_positions.items():
            if contract in positions:
                diff = int(qty - positions[contract])
                if diff != 0:
                    trade_records[contract] = diff        
            if qty != 0:
                context.stra_set_position(contract, qty)
            context.stra_log_text(f'Set {contract}\'s position to {qty}.')
            # 添加交易记录
        print(f'Trade records :\n {trade_records}')
        self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)
        print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')
        self.position_today = new_positions
        print(new_positions)
        self.traded = True
        print(f'{context.stra_get_date()}, {context.stra_get_time()} Self.Traded: {self.traded}')
        self.export_current_position()
        return None

    
    def export_current_position(self):
        context = self.__ctx__
        fake_pos = context.stra_get_all_position()
        real_pos = {}
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k] = v
        self.sql_position_writer.write_pos(self.name, str(context.stra_get_date()), real_pos)

    def on_session_end(self, context: CtaContext, curDate: int):
        self.export_current_position()
        print(f'[{curDate}] on_session_end ')        
        return None

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return None
