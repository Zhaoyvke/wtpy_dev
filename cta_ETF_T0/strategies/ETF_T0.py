"""
ETF_


"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import wtpy
import filters
import ehlers

from wtpy import BaseCtaStrategy, CtaContext
from MySqlIdxWriter import *


class ETF_T0_v0(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', etf='510050', period='m1', start_fund=10000000):
        BaseCtaStrategy.__init__(self, name)
        self.file_name = name
        self.exchg = exchg
        self.etf = etf
        self.period = period
        self.start_fund = start_fund
        self.instrument = self.exchg + '.ETF.' + self.etf
        self.multiplier = 100
        self.trend = 0
        # 读取sql配置
        # with open("database.json") as f:
        #     db_info = json.load(f)
        # trade_sqlfmt = """replace wz_optnew.simulation_trade_record (strategy, time, contract, qty) values ('$STRATEGY','$TIME','$CODE','$QTY')"""
        # self.sql_trade_writer = MySQLTradeWriter(db_info['host'], db_info['port'], db_info['user'], db_info['pwd'], db_info['dbname'], trade_sqlfmt)
        # pos_sqlfmt = """replace wz_optnew.simulation_eod_position (strategy, date, contract, qty) values ('$STRATEGY','$DATE','$CODE','$QTY')"""
        # self.sql_position_writer = MySQLPositionWriter(db_info['host'], db_info['port'], db_info['user'], db_info['pwd'], db_info['dbname'], pos_sqlfmt)

    def on_init(self, context: CtaContext):
        main_bars  = context.stra_get_bars(self.instrument, self.period, 360, isMain=True)
        main_ticks = context.stra_get_ticks(self.instrument, 1) 

        # subscribe ETF ticks
        # context.sub_ticks(self.instrument)
        # print(f"{self.instrument} tick subscribed!")
        return None

    def on_session_begin(self, context: CtaContext, curDate: int):
        print('******************************************************************')
        print(f'{curDate} {self.etf} on_session_begin')
        # etf = self.exchg + '.ETF.' + self.underlying
        return None
    
    def on_calculate(self, context: CtaContext):
        print(f"{context.get_time()} {self.instrument} on calculate trigger")
        N = 360
        L = 90
        bars = context.stra_get_bars(self.instrument, self.period, N)
        p = bars.closes
        r = np.diff(np.log(p))
        trend = filters.Lanczos(r, L)
        self.trend = trend[-1]
        pos = np.sign(trend[-1])        
        context.stra_set_position(self.instrument, self.multiplier*pos)
        return None

    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        # print(f"{newTick.time} {stdCode}, {newTick.bid_price_0}, {newTick.ask_price_0}, {self.trend}")
        return 
    

    def on_bar(self, context: CtaContext, stdCode: str, period: str, newBar: dict):
        print(f"[{stdCode}] {context.stra_get_time()}, Local time: {datetime.now()}")
        return 

    def on_session_end(self, context: CtaContext, curDate: int):
        # real_pos = self.export_current_position(context)
        # print(f'Today positions: {real_pos}')
        print('----------------------')
        return None
    
    def export_current_position(self, context: CtaContext):
        fake_pos = context.stra_get_all_position()
        real_pos = {}
        for k, v in fake_pos.items():
            if v != 0:
                real_pos[k] = v
        self.sql_position_writer.write_pos(self.file_name, str(context.stra_get_date()), real_pos)

        return real_pos
