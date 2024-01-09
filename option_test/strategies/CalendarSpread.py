# -*- coding: utf-8 -*-
'''
Calendar Spread策略

信号: 标的价格自上次交易变化1.5%

卖近月strike最接近und price的call&put, 买远月
每次交易 购买手数使portfolio delta接近0
根据margin设定自动调补仓位（组合保证金），保证margin使用率为25%左右

输出stressTest 自动

'''

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



class CalendarSpread(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', underlying='510050', period='d1', start_fund=100000):
        BaseCtaStrategy.__init__(self, name)
        self.lots = 0
        self.file_name = name
        self.exchg = exchg
        self.underlying = underlying
        self.period = period
        # self.enter = enter
        # self.exit = exit
        self.und_price = 0

        self.isTrading = True
        self.isRoll = False
        self.daily_adj = False

        self.contractInfo = None

        self.margin = 0
        self.margin_ratio = 0.4

        self.start_fund = start_fund
        self.und_change = 0.02

        self.trading_dates = get_trading_dates()
        # self.trading_time = None

        trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        self.sql_trade_writer = MySQLTradeWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', trade_sqlfmt)

        pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        self.sql_position_writer = MySQLPositionWriter("106.14.221.29", 5308, 'opter', 'wuzhi', 'wz_optnew', pos_sqlfmt)


    def on_init(self, context: CtaContext):
        print('Calendar Spread Started')

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
        cur_und_price = context.stra_get_bars(self.exchg + '.ETF.' + self.underlying, self.period, 1).closes[0]
        print(f'Current underlyingPrice is {cur_und_price}')
        print(f'Previous underlyingPrice is {self.und_price}')

        positions = context.stra_get_all_position()
        if len(positions) != 0:
            print(f'Current Positions are: ')
            for k, v in positions.items():
                print(f'..........{k}: {v}')

        if self.contractInfo is None:
            print(f'No contracts information available...')
            return None

        # 信号
        if abs(self.und_price / cur_und_price - 1) > self.und_change:
            self.isTrading = True
            # print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')
            print(f'{context.stra_get_time()} Signal showed off, begin transaction')




        # print(self.isTrading)
        #进行交易
        if self.isTrading:
            print('....Start option selection....')
            ttms = sorted(self.contractInfo['timeToMaturity'].unique())
            ttms = ttms if ttms[0] >= 2 else ttms[1:]

            this_month_trading_contracts = self.contractInfo.filter(pl.col('timeToMaturity') == ttms[0])
            next_month_trading_contracts = self.contractInfo.filter(pl.col('timeToMaturity') == ttms[1])

            short_options = this_month_trading_contracts.filter(
                (pl.col('strike') - pl.col('underlyingPrice')).apply(np.abs)
                == min(
                    np.abs(this_month_trading_contracts['strike'] - this_month_trading_contracts['underlyingPrice'])))
            short_options = short_options.with_columns(pl.col('code').apply(
                lambda x: context.stra_get_bars('.ETFO.'.join(x.split('.')), self.period, 1).closes[
                    0]).alias(
                'close'))

            ##计算margin
            short_options = short_options.select(pl.col('*'),
                                                 pl.struct(short_options.columns).apply(singleDeposit).alias('margin'))

            long_options = next_month_trading_contracts.filter(
                (pl.col('strike') - pl.col('underlyingPrice')).apply(np.abs)
                == min(
                    np.abs(next_month_trading_contracts['strike'] - next_month_trading_contracts['underlyingPrice'])))
            long_options = long_options.with_columns(pl.col('code').apply(
                lambda x: context.stra_get_bars('.ETFO.'.join(x.split('.')), self.period, 1).closes[
                    0]).alias(
                'close'))
            long_options = long_options.select(pl.col('*'),
                                               pl.struct(long_options.columns).apply(singleDeposit).alias(
                                                   'margin'))

            print('....End option selection....')

            short_call_option = short_options.filter(pl.col('type') == 'call')['code'][0]
            long_call_option = long_options.filter(pl.col('type') == 'call')['code'][0]
            short_put_option = short_options.filter(pl.col('type') == 'put')['code'][0]
            long_put_option = long_options.filter(pl.col('type') == 'put')['code'][0]

            call_margin = short_options.filter(pl.col('type') == 'call')['margin'][0]
            put_margin = short_options.filter(pl.col('type') == 'put')['margin'][0]

            if call_margin > put_margin:
                margin = call_margin + short_options.filter(pl.col('type') == 'put')['close'][0] * 10000
            else:
                margin = put_margin + short_options.filter(pl.col('type') == 'call')['close'][0] * 10000

            print(f'Margin is {margin}')

            self.lots = np.floor(self.margin_ratio * self.start_fund / margin)

            new_positions = {add_ETF(short_call_option): -1 * self.lots,
                             add_ETF(long_call_option): self.lots,
                             add_ETF(short_put_option): -1 * self.lots,
                             add_ETF(long_put_option): self.lots}

            trade_records = {}

            if len(new_positions) != 4:
                positions = context.stra_get_all_position()
                for k, v in positions.items():
                    context.stra_set_position(k, 0)
                print('less than 4 option chosen, clear positions....')
                trade_records[k] = v * -1
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
                    if k in trade_records.keys():
                        trade_records[k.split('.')[-1]] += v
                    else:
                        trade_records[k.split('.')[-1]] = v

                print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')

            trading_time = str(context.stra_get_time()) if len(str(context.stra_get_time())) == 4 else '0' + str(context.stra_get_time())
            trading_time = str(context.stra_get_date()) + trading_time

            self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)

            self.isTrading = False
            self.und_price = cur_und_price

        else:
            if self.isRoll:
                positions = context.stra_get_all_position()
                options_code = []

                for k, v in positions.items():
                    if v != 0:
                        code = sub_ETF(k)
                        options_code.append(code)
                        if v > 0:
                            if self.contractInfo.filter(pl.col('code') == code)['type'][0] == 'call':
                                long_call_option = code
                                long_call_lots = v
                            else:
                                long_put_option = code
                                long_put_lots = v
                        else:
                            if self.contractInfo.filter(pl.col('code') == code)['type'][0] == 'call':
                                short_call_option = code
                                short_call_lots = v
                            else:
                                short_put_option = code
                                short_put_lots = v

                ttms = self.contractInfo['timeToMaturity'].unique()

                if ttms[0] > 1:
                    target_ttm, target_ttm_next = ttms[0],ttms[1]
                else:
                    target_ttm, target_ttm_next = ttms[1],ttms[2]

                long_call_strike = self.contractInfo.filter(pl.col('code') == long_call_option)['strike'][0]
                long_call_option = self.contractInfo.filter(pl.col('timeToMaturity') == target_ttm_next).filter((pl.col('strike') - long_call_strike).abs() == (pl.col('strike') - long_call_strike).abs().min()).filter(pl.col('type') == 'call')['code'][0]

                long_put_strike = self.contractInfo.filter(pl.col('code') == long_put_option)['strike'][0]
                long_put_option = self.contractInfo.filter(pl.col('timeToMaturity') == target_ttm_next).filter((pl.col('strike') - long_put_strike).abs() == (pl.col('strike') - long_put_strike).abs().min()).filter(pl.col('type') == 'put')['code'][0]

                short_call_strike = self.contractInfo.filter(pl.col('code') == short_call_option)['strike'][0]
                short_call_option = self.contractInfo.filter(pl.col('timeToMaturity') == target_ttm).filter((pl.col('strike') - short_call_strike).abs() == (pl.col('strike') - short_call_strike).abs().min()).filter(pl.col('type') == 'call')['code'][0]

                short_put_strike = self.contractInfo.filter(pl.col('code') == short_put_option)['strike'][0]
                short_put_option = self.contractInfo.filter(pl.col('timeToMaturity') == target_ttm).filter((pl.col('strike') - short_put_strike).abs() == (pl.col('strike') - short_put_strike).abs().min()).filter(pl.col('type') == 'put')['code'][0]

                new_positions = {add_ETF(short_call_option): short_call_lots,
                                add_ETF(long_call_option): long_call_lots,
                                add_ETF(short_put_option): short_put_lots,
                                add_ETF(long_put_option): long_put_lots
                                }

                positions = context.stra_get_all_position()

                trade_records = {}

                if len(new_positions) != 4:
                    positions = context.stra_get_all_position()
                    for k, v in positions.items():
                        context.stra_set_position(k, 0)
                    print('less than 4 option chosen, clear positions....')
                    trade_records[k] = v * -1
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
                        if k in trade_records.keys():
                            trade_records[k.split('.')[-1]] += v
                        else:
                            trade_records[k.split('.')[-1]] = v

                print(f'{context.stra_get_date()}, {context.stra_get_time()} set new Position')


                trading_time = str(context.stra_get_time()) if len(str(context.stra_get_time())) == 4 else '0' + str(context.stra_get_time())
                trading_time = str(context.stra_get_date()) + trading_time

                self.sql_trade_writer.write_trade(self.file_name, trading_time, trade_records)


                self.isRoll = False


            # if self.daily_adj:
            #     positions = context.stra_get_all_position()
            #     short_options_code = []

            #     for k, v in positions.items():
            #         if v <= 0:
            #             short_options_code.append(sub_ETF(k))

            #     short_options = self.contractInfo.filter(pl.col('code').is_in(short_options_code))

            #     short_options = short_options.with_columns(pl.col('code').apply(
            #         lambda x:
            #         context.stra_get_bars(add_ETF(x), self.period, 1).closes[
            #             0]).alias(
            #         'close'))
            #     short_options = short_options.with_columns(pl.lit(cur_und_price).alias('underlyingPrice'))
            #     short_options = short_options.select(pl.col('*'),
            #                                         pl.struct(short_options.columns).apply(singleDeposit).alias(
            #                                             'margin'))

            #     call_margin = short_options.filter(pl.col('type') == 'call')['margin'][0]
            #     put_margin = short_options.filter(pl.col('type') == 'put')['margin'][0]

            #     if call_margin > put_margin:
            #         margin = call_margin + short_options.filter(pl.col('type') == 'put')['close'][0] * 10000
            #     else:
            #         margin = put_margin + short_options.filter(pl.col('type') == 'call')['close'][0] * 10000

            #     lots = np.floor(self.margin_ratio * self.start_fund / margin)

            #     if (self.lots - lots) > 0:
            #         print('Margin above water, reduce lots')
            #     elif (self.lots - lots) < 0:
            #         print('Margin under water, increase lots')
            #     else:
            #         print('Margin remains same, no operation on lots')

            #     if (self.lots - lots) != 0:
            #         self.lots = lots
            #         for k, v in positions.items():
            #             if v < 0:
            #                 context.stra_set_position(k, self.lots * -1)
            #             elif v > 0:
            #                 context.stra_set_position(k, self.lots)
            #             else:
            #                 pass

            #     self.hasTrading = True

            #     self.trading_time = fill_time(context.stra_get_date(), context.stra_get_time())
            #     if self.trading_time[-6:] == '150000':
            #         self.trading_time = fill_time(get_next_trading_date(context.stra_get_date(), self.trading_dates), '930')

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

        print('----------------------')

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
