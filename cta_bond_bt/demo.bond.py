from wtpy import BaseCtaStrategy
from wtpy import CtaContext
from Strategies.DualThrust import StraDualThrustStk
import random
import pandas as pd

class ConvertibleBond(BaseCtaStrategy):
    def __init__(self, name:str, codes:list, capital:float, barCnt:int, period:str):
        BaseCtaStrategy.__init__(self, name)
        self.__capital__ = capital # 起始资金
        self.__period__ = period   # 交易k线的时间级，如5分钟，1天
        self.__bar_cnt__ = barCnt  # 拉取的bar的次数

    def on_init(self, context:CtaContext):
        context.stra_log_text("Convertible Bond inited !")
        


    def on_session_begin(self, context: CtaContext, curDate: int):
        # 读取每天可交易的代码
        df = pd.read_parquet('./all_instruments.parquet')
        date = pd.to_datetime(str(curDate))
        date_list = df.loc[df['date'] <= date,'date'].unique()
        date_list.sort()
        latest_date = date_list[-1]
        
        df = df[(df.bond_type=='cb')&(df.date==latest_date)]
        cb_codes = df['order_book_id'].unique()
        stk_codes = df['stock_code'].unique()
        self.__cb_codes__ = cb_codes
        self.__stk_codes__ = stk_codes 
        
        # 获取每个标的前一天收盘价存储的基本面信息， 例如转股价   使用前一天的数据，因此要shift(1)
        df2 = pd.read_parquet('./universe.parq')
        self.fundamental_data = df2
        #建立字典将合约代码和基本面信息
        self.convertible_all_intruments={}
        for cb_code in cb_codes:
            bond_stk_code = df[df['order_book_id']==cb_code]['stock_code'].values[0]

            filtered_df2 = df2[df2['ticker']==cb_code]
            bond_ticker=filtered_df2['ticker'].values[0]
            bond_date=filtered_df2['date'].values[0]
            bond_prev_close=filtered_df2['prev_close'].values[0]
            bond_num_trades=filtered_df2['num_trades'].values[0]
            bond_low=filtered_df2['low'].values[0]
            bond_open=filtered_df2['open'].values[0]
            bond_total_turnover=filtered_df2['total_turnover'].values[0]
            bond_volume=filtered_df2['volume'].values[0]
            bond_close=filtered_df2['close'].values[0]
            bond_high=filtered_df2['high'].values[0]
            bond_pure_bond_value_1=filtered_df2['pure_bond_value_1'].values[0]
            bond_yield_to_maturity_pretax=filtered_df2['yield_to_maturity_pretax'].values[0]
            bond_pure_bond_value_pretax=filtered_df2['pure_bond_value_pretax'].values[0]
            bond_yield_to_put=filtered_df2['yield_to_put'].values[0]
            bond_conversion_premium=filtered_df2['conversion_premium'].values[0]
            bond_pure_bond_value=filtered_df2['pure_bond_value'].values[0]
            bond_put_trigger_price=filtered_df2['put_trigger_price'].values[0]
            bond_conversion_value=filtered_df2['conversion_value'].values[0]
            bond_put_status=filtered_df2['put_status'].values[0]
            bond_put_qualified_days=filtered_df2['put_qualified_days'].values[0]
            bond_call_status=filtered_df2['call_status'].values[0]
            bond_pure_bond_value_premium=filtered_df2['pure_bond_value_premium'].values[0]
            bond_yield_to_put_pretax=filtered_df2['yield_to_put_pretax'].values[0]
            bond_remaining_size=filtered_df2['remaining_size'].values[0]
            bond_call_qualified_days=filtered_df2['call_qualified_days'].values[0]
            bond_conversion_coefficient=filtered_df2['conversion_coefficient'].values[0]
            bond_call_trigger_price=filtered_df2['call_trigger_price'].values[0]
            bond_iv=filtered_df2['iv'].values[0]
            bond_turnover_rate=filtered_df2['turnover_rate'].values[0]
            bond_conversion_price_reset_qualified_days=filtered_df2['conversion_price_reset_qualified_days'].values[0]       
            bond_conversion_price_reset_status=filtered_df2['conversion_price_reset_status'].values[0]
            bond_conversion_price_reset_trigger_price=filtered_df2['conversion_price_reset_trigger_price'].values[0]
            bond_yield_to_maturity=filtered_df2['yield_to_maturity'].values[0]
            bond_pure_bond_value_premium_pretax=filtered_df2['pure_bond_value_premium_pretax'].values[0]
            bond_double_low_factor=filtered_df2['double_low_factor'].values[0]
            bond_pb_ratio=filtered_df2['pb_ratio'].values[0]
            bond_pure_bond_value_premium_1=filtered_df2['pure_bond_value_premium_1'].values[0]
            bond_convertible_market_cap_ratio=filtered_df2['convertible_market_cap_ratio'].values[0]
            bond_is_suspended=filtered_df2['is_suspended'].values[0]
            bond_industry=filtered_df2['industry'].values[0]
            bond_listed_date=filtered_df2['listed_date'].values[0]
            bond_stop_trading_date=filtered_df2['stop_trading_date'].values[0]
            bond_days_from_listed=filtered_df2['days_from_listed'].values[0]
            bond_days_to_stoptrading=filtered_df2['days_to_stoptrading'].values[0]        

               # Create a dictionary with trading data
            trading_info = {
            'order_book_id': cb_code,
            'stock_code': bond_stk_code,
            'ticker': bond_ticker,
            'date': bond_date,
            'prev_close': bond_prev_close,
            'num_trades': bond_num_trades,
            'low': bond_low,
            'open': bond_open,
            'total_turnover': bond_total_turnover,
            'volume': bond_volume,
            'close': bond_close,
            'high': bond_high,
            'pure_bond_value_1': bond_pure_bond_value_1,
            'yield_to_maturity_pretax': bond_yield_to_maturity_pretax,
            'pure_bond_value_pretax': bond_pure_bond_value_pretax,
            'yield_to_put': bond_yield_to_put,
            'conversion_premium': bond_conversion_premium,
            'pure_bond_value': bond_pure_bond_value,
            'put_trigger_price': bond_put_trigger_price,
            'conversion_value': bond_conversion_value,
            'put_status': bond_put_status,
            'put_qualified_days': bond_put_qualified_days,
            'call_status': bond_call_status,
            'pure_bond_value_premium': bond_pure_bond_value_premium,
            'yield_to_put_pretax': bond_yield_to_put_pretax,
            'remaining_size': bond_remaining_size,
            'call_qualified_days': bond_call_qualified_days,
            'conversion_coefficient': bond_conversion_coefficient,
            'call_trigger_price': bond_call_trigger_price,
            'iv': bond_iv,
            'turnover_rate': bond_turnover_rate,
            'conversion_price_reset_qualified_days': bond_conversion_price_reset_qualified_days,
            'conversion_price_reset_status': bond_conversion_price_reset_status,
            'conversion_price_reset_trigger_price': bond_conversion_price_reset_trigger_price,
            'yield_to_maturity': bond_yield_to_maturity,
            'pure_bond_value_premium_pretax': bond_pure_bond_value_premium_pretax,
            'double_low_factor': bond_double_low_factor,
            'pb_ratio': bond_pb_ratio,
            'pure_bond_value_premium_1': bond_pure_bond_value_premium_1,
            'convertible_market_cap_ratio': bond_convertible_market_cap_ratio,
            'is_suspended': bond_is_suspended,
            'industry': bond_industry,
            'listed_date': bond_listed_date,
            'stop_trading_date': bond_stop_trading_date,
            'days_from_listed': bond_days_from_listed,
            'days_to_stoptrading': bond_days_to_stoptrading,
             }
    
             # Add trading data to the dictionary using cb_codes as the key
            self.convertible_all_intruments[cb_code] = trading_info

        # 每天设置一个主品种
        # for i in range(0,len(codes)):
        #     if i == 0:
        #         context.stra_get_bars(codes[i], self.__period__, self.__bar_cnt__, isMain=True) # 设置第一支股票为主要品种
        #     else:
        #         context.stra_get_bars(codes[i], self.__period__, self.__bar_cnt__, isMain=False)      
        
    def on_session_end(self, context: CtaContext, curTDate: int):
        # 对当日绩效进行结算
        pass
    
    def on_calculate(self, context:CtaContext):
        stk_codes = self.__stk_codes__                              
        cb_codes = self.__cb_codes__                                
        capital = self.__capital__                                  
        date = context.stra_get_date()
        time = context.stra_get_time()
        print(111111111111111111111111111111111111111)
        #code = '115001'
        price = context.stra_get_price(stk_codes)
        #print(f"code={code}, price={price}")


        
if __name__ == "__main__":
    from wtpy import WtBtEngine,EngineType
    from wtpy.apps import WtBtAnalyst
    import os
    #path = "./"
    #datanames = os.listdir(path)
    #files = [string.replace('.csv','') for string in datanames]
    #files = [string.replace('_d','') for string in files]

    #创建一个运行环境，并加入策略

    engine = WtBtEngine(EngineType.ET_CTA)

    engine.init(folder='./', cfgfile="configbt.yaml", commfile="commodities_stk.json", contractfile="contracts_stk.json")
    engine.configBacktest(201010090930,201010211500)
    
    # import pdb;pdb.set_trace()
    engine.configBTStorage(mode="wtp", path="./storage/")
    engine.commitBTConfig()
  
    straInfo =StraDualThrustStk(name='bond', code="SSE.BOND.115001", barCnt=5, period="m1", days=30, k1=0.1, k2=0.1)
    engine.set_cta_strategy(straInfo)

    engine.run_backtest() 

    #绩效分析
    #analyst = WtBtAnalyst()
    #analyst.add_strategy("Hushen", folder="./outputs_bt", init_capital=10000000, rf=0.02, annual_trading_days=240)
    #analyst.run()

    kw = input('press any key to exit\n')
    engine.release_backtest()