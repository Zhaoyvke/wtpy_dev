# -*- coding: utf-8 -*-
"""
Iron Condor策略

信号: 标的价格自上次交易变化1.5%

期权 long由self.long_delta决定, short由self.short_delta决定

每次交易 购买手数使portfolio delta接近0

根据margin和delta设定自动调补仓位（组合保证金）由self.daily_adj决定

输出stressTest 自动

""" 
from wtpy import BaseCtaStrategy, CtaContext

class IronCondor(BaseCtaStrategy):
    def __init__(self, name, exchg='SSE', underlying='510050', period='m1'):
        BaseCtaStrategy.__init__(self, name)
        self.file_name = name
        self.exchg = exchg
        self.underlying = underlying
        self.period = period

    def on_init(self, context: CtaContext):
        print('IronCondor Started')
        print('******************************************************************')        
        print()        
        self.opts = context.stra_get_codes_by_underlying(f'{self.exchg}.{self.underlying}')        
        print(self.opts)
        # context.stra_get_bars('SSE.ETF.510050', 'm1', 1)
        # return
        for opt in self.opts:
            exchg, code = opt.split('.')
            # opt_bars = context.stra_get_bars(opt, 'm5', 10)
            opt_bars = context.stra_get_bars(f'{exchg}.ETFO.{code}', 'm5', 30)
            if opt_bars is None:
                print(f'{opt} has no bars.')
            else:
                print(len(opt_bars))
        
        return None

    def on_session_begin(self, context: CtaContext, curDate: int):
        return None
    
    def on_calculate(self, context: CtaContext):                
        for opt in self.opts:
            opt_bars = context.stra_get_bars(self.exchg + '.' + opt, self.period, 10)
            print(len(opt_bars))            
        return None

    def on_session_end(self, context: CtaContext, curDate: int):
        return None
