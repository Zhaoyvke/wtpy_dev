# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:36:18 2023

@author: OptionTeam
"""

from wtpy import WtEngine, EngineType
from strategies.ETF_T0 import ETF_T0_v0
from strategies.ETF_T0_lxy import ETF_T0_lxy
from strategies.ETF_T0_lxy_OFI import ETF_T0_lxy_OFI

if __name__ == '__main__':
    engine = WtEngine(EngineType.ET_CTA)
    engine.init('./common/', 'config.yaml')
    freq = 'm1'

    ETFs = [['SSE','510050'],
            ['SSE','510300'],
            ['SSE','510500'],
            ['SSE','512880'],
            ['SSE','588000'],
            ['SSE','588080'],
            ['SZSE','159901'],
            ['SZSE','159915'],
            ['SZSE','159919'],
            ['SZSE','159922']]
    
    
    # for exchg,etf in ETFs:
    #     name = f'T0_{etf}'
    #     print(f'stra name: {name}')
    #     straInfo = ETF_T0_v0(name, exchg, etf, freq)
    #     engine.add_cta_strategy(straInfo)
    
    # LXY T0
    # ETFs = [['SZSE','159901'],
    #         ['SZSE','159915'],
    #         ['SZSE','159919'],
    #         ['SZSE','159922'],
    #         # ['SZSE','159845'],
    #         # ['SZSE','159949'],
    #         # ['SZSE','159995'],
    #         # ['SZSE','159841'],
    #         # # ['SZSE','159859'],
    #         # ['SZSE','159633'],
    #         # ['SZSE','159928'],
    #          ]
    for exchg,etf in ETFs:
        name = f'T0_LXY_{etf}'
        print(f'stra name: {name}')
        straInfo = ETF_T0_lxy(name, exchg, etf, freq)        
        engine.add_cta_strategy(straInfo)
        break

        # name = f'T0_LXY_{etf}_OFI'
        # print(f'stra name: {name}')
        # straInfo = ETF_T0_lxy_OFI(name, exchg, etf, freq)
        # engine.add_cta_strategy(straInfo)
        
    engine.run()
    kw = input('press any key to exit\n')

    
