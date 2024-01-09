# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:36:18 2023

@author: OptionTeam
"""

from wtpy import WtEngine, EngineType
from strategies.ETF_T0_lxy import ETF_T0_lxy

if __name__ == '__main__':
    engine = WtEngine(EngineType.ET_CTA)
    engine.init('F:/deploy/ETF_T0_multi_underlying/', 'config.yaml')
    freq = 'm1'

    ETFs = [['SSE','510050'],
            ['SSE','510300'],
            ['SSE','510500'],
            ['SSE','588000'],
            ['SSE','588080'],
            # ['SSE','512480'],
            # ['SSE','513120'],
            # ['SSE','512100'],
            # ['SSE','512880'],
            # ['SSE','510310'],
            # ['SSE','510330'],
            # ['SSE','518880'],
            # ['SSE','560010'],
            # ['SSE','513050'],
            
            ['SZSE','159901'],
            ['SZSE','159915'],
            ['SZSE','159919'],
            ['SZSE','159922'],
            ['SZSE','159845'],
            # ['SZSE','159633'],
            # ['SZSE','159629'],
            # ['SZSE','159949'],
            ]
    
    # for exchg,etf in ETFs:
    #     name = f'T0_LXY_{etf}'
    #     print(f'stra name: {name}')
    #     straInfo = ETF_T0_lxy(name, exchg, etf, freq)        
    #     engine.add_cta_strategy(straInfo)
    #     # break
    straInfo = ETF_T0_lxy('T0_ETF_multi_underlying', ETFs, freq)        
    engine.add_cta_strategy(straInfo)
    engine.run()
    kw = input('press any key to exit\n')

    
