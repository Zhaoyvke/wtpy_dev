# -*- coding: utf-8 -*-
"""
Created on 20230629
    SSR statistical arbitrage
    for simulation / real trade
@author: OptionTeam
"""
import WonderWizOption as wwo
import pandas as pd
import polars as pl
import numpy as np

from wtpy import BaseCtaStrategy
from wtpy import CtaContext
from MySqlIdxWriter import *

class skew_stickiness_ratio(BaseCtaStrategy):
    def __init__(self, name, exchg='CFFEX', optionCode='MO', period='m5', start_fund=1000000):
        BaseCtaStrategy.__init__(self, name)
        self.file_name = name
        self.exchg = exchg
        # self.optionCode = optionCode
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
        trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        self.sql_trade_writer = MySQLTradeWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', trade_sqlfmt)
        pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        self.sql_position_writer = MySQLPositionWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', pos_sqlfmt)                
        self.get_trading_dates()
        self.traded = False
        print('__init__ finished')
        
    def on_init(self, context: CtaContext):
        print(f'Skew Stickiness Ratio StatArb {self.underlying} started')
        self.__ctx__ = context
        self.expireContracts = set()        
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

    def on_session_begin(self, context: CtaContext, curDate: int):
        print(f'[{curDate}] on_session_begin')  
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

    def refresh_option_info(self, context: CtaContext):
        # 更新最新的价格信息
        # self.ttms, self.Fs, self.rs
        # 使用implied forward的话，只需ttms
        for i in self.options_by_month.keys():
            Ks = self.options_by_month[i].index
            for K in Ks:
                # 更新价格
                for cpflag in ['C', 'P']:
                    code = self.options_by_month[i].loc[K, f'{cpflag}_code']
                    tick = context.get_ticks(code,1)
                    if not tick is None:
                        self.options_by_month[i].loc[K,f'{cpflag}_bid'] = tick['bid_price_0']
                        self.options_by_month[i].loc[K,f'{cpflag}_ask'] = tick['ask_price_0']
                        self.options_by_month[i].loc[K,f'{cpflag}'] = np.round((tick['ask_price_0'] + tick['bid_price_0']) / 2,5)                
                    else:
                        bar = context.stra_get_bars(code, self.period, 1)
                        if not bar is None:
                            self.options_by_month[i].loc[K, f'{cpflag}'] = bar.closes[0]
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
                # margin and position   假设100000一手
                self.options_by_month[i].loc[K,'C_margin'] = 100000
                self.options_by_month[i].loc[K,'P_margin'] = 100000
                    
    def generate_new_position(self):
        # 交易的月份
        mm = 0 + 1*(self.ttms[0] < 5/240)        
        Ks = self.options_by_month[mm].index        
        K_atm = Ks[np.argmin(np.abs(self.options_by_month[mm]['C']-self.options_by_month[mm]['P']))]
        impF = K_atm + self.options_by_month[mm].loc[K_atm, 'C'] - self.options_by_month[mm].loc[K_atm, 'P']
        moneyness = 0.05    # 应该能区分
        Kp = Ks[np.argmin(np.abs(Ks-impF*(1-moneyness)))]
        Kc = Ks[np.argmin(np.abs(Ks-impF*(1+moneyness)))]        
        code_c = self.options_by_month[mm].loc[Kc,'C_code']
        code_p = self.options_by_month[mm].loc[Kp,'P_code']        
        code_c_atm = self.options_by_month[mm].loc[K_atm,'C_code']        
        code_p_atm = self.options_by_month[mm].loc[K_atm,'P_code']        
        # if (Kp == Kc): return None
        # 卖出Call，买入Put，组成risk reversal?
        call_lots = - np.floor(self.margin_ratio * self.start_fund / self.options_by_month[mm].loc[Kc,'C_margin'] / 2)
        # gamma hedge
        put_lots = np.floor(np.abs(call_lots * self.options_by_month[mm].loc[Kc, 'C_gamma'] / self.options_by_month[mm].loc[Kp, 'P_gamma']))
        net_delta = put_lots * self.options_by_month[mm].loc[Kp,'P_delta'] + call_lots * self.options_by_month[mm].loc[Kc,'C_delta']
        hedge_qty = np.round(np.abs(net_delta)) # > 0
        print(f'[Delta] {net_delta+hedge_qty}')
        codes = [code_c,code_p,code_c_atm,code_p_atm]
        pos = pd.DataFrame([call_lots, put_lots, -hedge_qty, hedge_qty],index=codes,columns=['qty'])      
        pos = pos.groupby(pos.index).sum()  
        return pos

    def on_calculate(self, context: CtaContext):
        # 仅在特定时刻操作
        # if (not context.stra_get_time()>1430) or (context.stra_get_time() > 1435):
        #     return
        print(f'{context.stra_get_date()}, {context.stra_get_time()} Self.Traded: {self.traded}')
        if self.traded:
            return
        # 标的的最新价格
        self.refresh_option_info(context)
        # 取出ATM iv或者sigma0?
        pos = self.generate_new_position()
        print(pos)
        
        # ------------------------------交易决策----------------------------
        # 先清空所有position,并记录交易，存放在一个大文件里面，然后加一栏日期
        trading_time = str(context.stra_get_date()*10000 + context.stra_get_time())        
        trade_records = {}

        positions = context.stra_get_all_position()
        new_positions = {k:pos.loc[k,'qty'] for k in pos.index}
        # print(positions)        
        if len(positions) > 0:
            for contract, qty in positions.items():
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
        real_pos = dict()
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k] = v
        self.sql_position_writer.write_pos(self.name, str(context.stra_get_date()), real_pos)

    def on_session_end(self, context: CtaContext, curDate: int):
        # PNL Attribution
        # 输出日末持仓
        self.export_current_position()
        print(f'[{curDate}] on_session_end ')
        print('******************************************************************')
        return None

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return None
