# -*- coding: utf-8 -*-
"""
straddle策略

信号:颜老师的T2信号

期权 ITM和OTM  long

每次交易 购买手数使portfolio delta接近0

根据margin和delta设定自动调补仓位（组合保证金）由self.daily_adj决定


"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

import wtpy
import AttributionAnalysis
from OptionInfo import *
from StressTest import *
from SupplementaryFunc import *
from MySqlIdxWriter import *
from wtpy import BaseCtaStrategy, CtaContext

import pymysql

class StraddleT2(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', underlying='510300', period='m1', start_fund=100000, short_delta=0.4,
                 long_delta=0.2):
        BaseCtaStrategy.__init__(self, name)
        self.file_name = name
        self.exchg = exchg
        self.underlying = underlying
        self.period = period
        # self.enter = enter
        # self.exit = exit
        self.und_price = 0

        #是否需要交易
        self.isTrading = False

        #上一条K线的信号出来后，变成TRue, 在当前K线交易
        self.hasTrading = False

        #是否移仓到下个月
        self.isRoll = False

        #每日可交易期权信息
        self.contractInfo = None

        self.short_delta = short_delta  # 正数， put的就为负
        self.long_delta = long_delta


        self.vix = []

        self.margin = 0
        self.margin_ratio = 0.4

        self.und_change = 0.02

        # 是否需要每日调仓
        self.daily_adjust = False

        self.today_trading_positions = {}

        # self.loss_point = []

        self.trading_dates = get_trading_dates()
        # self.trading_time = None

        trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        self.sql_trade_writer = MySQLTradeWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', trade_sqlfmt)

        pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        self.sql_position_writer = MySQLPositionWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', pos_sqlfmt)


    def on_init(self, context: CtaContext):
        print('IronCondor Started')
        return None

    def on_session_begin(self, context: CtaContext, curDate: int):
        print('******************************************************************')
        print('[%s] on_session_begin' % (curDate))
        und_bars = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1, isMain=True)
        print('---------------Underlying----------------------')
        print(und_bars.closes)
        avail_contracts = context.stra_get_codes_by_underlying(f'{self.exchg}.{self.underlying}')
        self.contractInfo = pd.DataFrame(self.complete_option_info(avail_contracts, context)).T.reset_index()
        self.contractInfo.rename(columns={'index': 'code'}, inplace=True)
        self.contractInfo['date'] = context.stra_get_date()
        if self.contractInfo.shape[0] > 0:
            print('Contracts Info collected')
        else:
            print('Failed to collect contracts info')
        self.contractInfo = pl.DataFrame(self.contractInfo)
        last_und_info = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1)
        # 在日初添加标的价格为昨天的收盘价
        self.contractInfo = self.contractInfo.with_columns(pl.lit(last_und_info.closes[-1]).alias('underlyingPrice'))
        if len(context.stra_get_all_position()) != 0:
            for k, v in context.stra_get_all_position().items():
                if v != 0:
                    # tmp = k.split('.')
                    # contract = tmp[0] + '.' + tmp[2]
                    contract = sub_ETF(k)
                    this_contractInfo = self.contractInfo.filter(pl.col('code') == contract)
                    #如果还有一天到期，就开始移仓
                    ttm = this_contractInfo['timeToMaturity'][0]
                    if ttm <= 1:
                        print(contract, ': ', ttm)
                        self.isRoll = True
            if self.isRoll:
                print('One or more options approach maturity, start rolling over......')
            else:
                print('All options in hand are available.')

        return None

    def complete_option_info(self, contracts, context: CtaContext):
        '''
        填充期权信息
        return a dictionary
        '''
        all_optionInfo = {}
        # prev_date = get_previous_trading_date(context.stra_get_date(), self.trading_dates)
        for contract in contracts:
            print(f'loading {contract} data')
            try:
                bar_info = context.stra_get_bars('.ETFO.'.join(contract.split('.')), self.period, 1)
            except:
                continue
            if bar_info is None:
                continue

            name = context.stra_get_contractinfo(contract).name
            thisOption = {}
            #计算距离到期日天数
            # import pdb
            # pdb.set_trace()
            month = str(month_in_name(name))
            ttm = timeToMaturity(str(context.stra_get_date()), month, self.trading_dates)
            thisOption['timeToMaturity'] = ttm
            if ttm < 0:
                print('Error: one or more options have time to maturity less than 0 !!!')
            thisOption['strike'] = context.stra_get_contractinfo(contract).strikePrice
            thisOption['type'] = 'call' if '购' in name else 'put'
            # 添加上一条k线的收盘价
            thisOption['close'] = bar_info.closes[0]
            # 利率先设置为0
            thisOption['interest_rate'] = 0
            thisOption['multiplier'] = 10000
            all_optionInfo[contract] = thisOption ##code中间没有ETF

        return all_optionInfo

    def on_calculate(self, context: CtaContext):
        # print(f'{context.stra_get_date()}, {context.stra_get_time()} On calculate triggered')
        # cur_und_price = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        cur_und_price = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        print(f'Current underlyingPrice is {cur_und_price}')
        print(f'Previous underlyingPrice is {self.und_price}')

        hasPositions = False
        positions = context.stra_get_all_position()
        if len(positions) != 0:
            for k, v in positions.items():
                if v != 0:
                    hasPositions = True
                    print(f'Current Positions includes: ')
                    print(f'..........{k}: {v}')


        if self.contractInfo is None:
            print(f'No contracts information available...')
            return None



        # 更新当前标的价格
        self.contractInfo = self.contractInfo.with_columns(pl.lit(cur_und_price).alias('underlyingPrice'))




        db_conn = pymysql.connect(host='106.14.221.29', user='opt_scanner', passwd='wuzhi', db='wz_option', port=5308)

        cur = db_conn.cursor()
        cur.execute("select target from tb_signals where groupid = 'T2'")
        signal = cur.fetchall()[0][0]
        db_conn.commit()

        # 如果有信号，则开始交易
        print(f'T2 signal is {signal}')
        if (signal != 0)  & (hasPositions == False):
            self.isTrading = True
            print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')
            print(f'{context.stra_get_time()} Signal showed off, begin transaction')


        # 计算VIX 根据VIX调整delta的大小
        # near_opt, next_opt = get_near_next_options(self.contractInfo)
        # self.vix.append(impVIXTwoSide(near_opt, next_opt))

        # if self.vix[-1] > 20:
        #     self.short_delta, self.long_delta = 0.25, 0.1
        # elif self.vix[-1] <= 19:
        self.short_delta, self.long_delta = 0.25, 0.1

        # print(f'Delta set up {self.short_delta}, {self.long_delta} according to VIX {self.vix}')
        def calc_IV_delta_gamma(strike, interest_rate, timeToMaturity, option_type, close):
            thisOption = Option(cur_und_price, strike, interest_rate, timeToMaturity, 0,
                                option_type)
            IV = thisOption.getIV(close)
            thisOption.sigma = IV
            delta = thisOption.getDelta()
            gamma = thisOption.getGamma()

            return (IV, delta, gamma)

        # 开始交易
        if self.isTrading:
            print('....Start option selection....')



            # 筛选出当月期权
            ttms = sorted(self.contractInfo['timeToMaturity'].unique())
            target_ttm = ttms[0] if ttms[0] >= 2 else ttms[1]
            trading_contracts = self.contractInfo.filter(pl.col('timeToMaturity') == target_ttm)

            # 更新当前的所有合约的价格
            trading_contracts = trading_contracts.with_columns(pl.col('code').apply(
                lambda x: context.stra_get_bars('.ETFO.'.join(x.split('.')), self.period, 1).closes[
                    0]).alias(
                'close'))

            # 计算风险暴露
            exposures = trading_contracts.apply(lambda x: calc_IV_delta_gamma(x[2], x[5], x[1], x[3], x[4]))
            exposures = exposures.rename({'column_0': 'IV', 'column_1': 'delta', 'column_2': 'gamma'})
            trading_contracts = pl.concat([trading_contracts, exposures], how="horizontal")


            # 挑出看跌期权
            put_contracts = trading_contracts.filter(pl.col('type') == 'put').filter(pl.col('strike') < cur_und_price)
            put_strikes = put_contracts['strike'].unique()

            target_strike = find_n_closest_value(put_strikes, cur_und_price, 1)
            long_put_option = put_contracts.filter(pl.col('strike') == target_strike)['code'][0]

            # 挑出看涨期权
            call_contracts = trading_contracts.filter(pl.col('type') == 'call').filter(pl.col('strike') > cur_und_price)
            call_strikes = call_contracts['strike'].unique()

            target_strike = find_n_closest_value(call_strikes, cur_und_price, 1)
            long_call_option = call_contracts.filter(pl.col('strike') == target_strike)['code'][0]
            print('....End option selection....')


            # 计算组合风险敞口
            delta_exposed_put = trading_contracts.filter(pl.col('code') == long_put_option)['delta'][0]
            delta_exposed_call = trading_contracts.filter(pl.col('code') == long_call_option)['delta'][0]

            print(f'exposures: {delta_exposed_call}, {delta_exposed_put}')

            m1 = trading_contracts.filter(pl.col('code') == long_put_option)['close'][0] * 10000
            m2 = trading_contracts.filter(pl.col('code') == long_call_option)['close'][0] * 10000

            print(f'prices: put: {m1}, call: {m2}')

            # 调整手数使得delta尽可能小 并且保证金率大致在self.margin_ratio附近
            call_lots = np.round(self.margin_ratio * self.start_fund * delta_exposed_put / (
                        m1 * delta_exposed_put - m2 * delta_exposed_call))
            put_lots = np.round(self.margin_ratio * self.start_fund * delta_exposed_call / (
                        m2 * delta_exposed_call - m1 * delta_exposed_put))

            # 计算保证金
            margin = np.abs(m1) * call_lots + np.abs(m2) * put_lots
            self.margin = margin
            print(f'Margin is {margin}')
            print(f'Left delta exposed is {call_lots * delta_exposed_call + put_lots * delta_exposed_put}')

            # lots = np.floor(self.margin_ratio * self.start_fund / self.margin)

            new_positions = {add_ETF(long_call_option): call_lots,
                             add_ETF(long_put_option): put_lots
                             }

            trade_records = {}

            if len(new_positions) != 2:
                positions = context.stra_get_all_position()
                for k, v in positions.items():
                    context.stra_set_position(k, 0)
                print('less than 2 option chosen, clear positions....')
                trade_records[k.split('.')[-1]] = v * -1
            else:
                positions = context.stra_get_all_position()
                if len(positions) == 0:
                    pass
                else:
                    for k, v in positions.items():
                        if v != 0:
                            context.stra_set_position(k, 0)
                            trade_records[k.split('.')[-1]] = v * -1

                for k, v in new_positions.items():
                    context.stra_set_position(k, v)
                    if k.split('.')[-1] in trade_records.keys():
                        trade_records[k.split('.')[-1]] += v
                    else:
                        trade_records[k.split('.')[-1]] = v



                print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')

            # print(self.name, type(self.name))
            # time = str(context.stra_get_date())+str(context.stra_get_time())
            # print(time,type(time))
            # print(trade_records, type(trade_records))

            trading_time = str(context.stra_get_time()) if len(str(context.stra_get_time())) == 4 else '0' + str(context.stra_get_time())
            trading_time = str(context.stra_get_date()) + trading_time

            self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)


            self.isTrading = False
            self.hasTrading = True
            self.und_price = cur_und_price


        if (signal == 0) & (hasPositions == True):
           trade_records = {}
           positions = context.stra_get_all_position()
           if len(positions) != 0:
               for k, v in positions.items():
                   context.stra_set_position(k, 0)
                   if v != 0:
                       trade_records[k.split('.')[-1]] = v * -1

               trading_time = str(context.stra_get_time()) if len(str(context.stra_get_time())) == 4 else '0' + str(context.stra_get_time())
               trading_time = str(context.stra_get_date()) + trading_time

               self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)

               print('Positions all cleared...')




        return None

    def on_session_end(self, context: CtaContext, curDate: int):

        fake_pos = context.stra_get_all_position()
        real_pos = {}
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k.split('.')[-1]] = v
        print(f'{self.name}_ {self.underlying} ready to finish...')
        print(f'Today positions: {real_pos}')
        self.sql_position_writer.write_pos(self.file_name, str(context.stra_get_date()), real_pos)

        # print('----------------------')

        # pos = []
        # options = []

        # last_und_close = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[-1]

        # # 在日末更新contractInfo 中的标的价格， 最后一根K线的close
        # self.contractInfo = self.contractInfo.with_columns(pl.lit(last_und_close).alias('underlyingPrice'))

        # for k, v in real_pos.items():
        #     latest_close = context.stra_get_bars(k, self.period, 1).closes[0]
        #     code = k.replace('ETF.', '')
        #     info = self.contractInfo.filter(pl.col('code') == code)

        #     pos.append(v)

        #     this_option = data_to_option(info)[0]

        #     this_option.price = latest_close
        #     options.append(this_option)

        # st, _ = StressTest_Portfolio(options).vanilla(10000, pos)

        # print('Daily Stress Test generated.')
        # print('----------------------')


        # self.trading_contracts = None
        # print('Trading contracts cleared')
        # print('----------------------')
        print('[%s] on_session_end' % (curDate))
        print('******************************************************************')
        print()

        return None

    # def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
    #     return None
