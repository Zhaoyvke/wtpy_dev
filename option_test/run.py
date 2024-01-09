# -*- coding: utf-8 -*-
"""
Created on Thu Mar  14:17 2023

@author: wzer
"""

from wtpy import WtEngine, EngineType
from strategies.IronCondor import *
from strategies.CalendarSpread import *
from strategies.StraddleT1 import *
from strategies.StraddleT2 import *

# import sys


if __name__ == '__main__':
    engine = WtEngine(EngineType.ET_CTA)
    engine.init('F:/wt_opt_sim/common/', "config.yaml")

    unds = {'510050':'SSE', '510300':'SSE', '510500':'SSE', '159901':'SZSE', '159915':'SZSE', '159919':'SZSE', '159922':'SZSE'}#/, '588080':'SSE', '588000':'SSE'}

    for und, exchg in unds.items():
        freq = 'm5'

        name = f"""{und}_IronCondor"""
        print(f'stra name: {name}')
        start_fund = 100000

        #
        straInfo = IronCondor(name=name)
        straInfo.exchg = exchg
        straInfo.underlying = und
        straInfo.period = freq
        straInfo.start_fund = start_fund
        straInfo.margin_ratio = 0.25

        engine.add_cta_strategy(straInfo)

    for und, exchg in unds.items():
        freq = 'm5'

        name = f"""CalendarSpread_{und}_{freq}"""
        print(f'stra name: {name}')
        start_fund = 100000

        #
        straInfo = CalendarSpread(name=name)
        straInfo.exchg = exchg
        straInfo.underlying = und
        straInfo.period = freq
        straInfo.start_fund = start_fund
        straInfo.margin_ratio = 0.25

        engine.add_cta_strategy(straInfo)

    freq = 'm1'
    und = '510300'
    name = f"""StraddleT1_{und}_{freq}"""
    print(f'stra name: {name}')
    start_fund = 100000
    straInfo = StraddleT1(name=name)
    straInfo.exchg = 'SSE'
    straInfo.underlying = '510300'
    straInfo.period = freq
    straInfo.start_fund = start_fund
    straInfo.margin_ratio = 0.25

    engine.add_cta_strategy(straInfo)

    freq = 'm1'
    und = '510050'
    name = f"""StraddleT2_{und}_{freq}"""
    print(f'stra name: {name}')
    start_fund = 100000
    straInfo = StraddleT2(name=name)
    straInfo.exchg = 'SSE'
    straInfo.underlying = '510050'
    straInfo.period = freq
    straInfo.start_fund = start_fund
    straInfo.margin_ratio = 0.25

    engine.add_cta_strategy(straInfo)


    engine.run()

    kw = input('press any key to exit\n')



