from wtpy import WtEngine, EngineType
from strategies.hftSpreadTradeBase import hftSpreadTradeBase
from strategies.HftStraDemo import HftStraDemo
from SpreadConstant import Direction
from SpreadBase import TradeData,SpreadData, LegData
from SpreadFunction import round_to
from wtpy.ProductMgr import ProductMgr, ProductInfo
from SpreadTemplate import SpreadAlgoTemplate
import datetime
from wtpy import HftContext,BaseHftStrategy
from collections import defaultdict
from typing import Dict, List, Set, Callable, TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .SpreadEngine import SpreadAlgoEngine
from wtpy.ContractMgr import ContractMgr, ContractInfo
from SpreadConstant import Direction, Status, Offset,OrderData
from SpreadFunction import  floor_to, ceil_to, round_to


    #热卷螺纹套利
sp = SpreadData(                        
    name='hc_rb_05',
    legs=[
        LegData(stdCode='SHFE.hc.2405'),   #主动腿（不活跃的合约）
        LegData(stdCode='SHFE.rb.2405')
    ],
    variable_symbols={'A': 'SHFE.hc.2405', 'B': 'SHFE.rb.2405'},
    variable_signal_symbols={},
    variable_directions=  {'hc_rb_05':1},
    variable_direction_buy={'A': Direction.LONG, 'B': Direction.SHORT},
    variable_direction_sell={'A': Direction.SHORT, 'B': Direction.LONG},
    price_formula='A*1-B*1',
    trading_multipliers={'SHFE.hc.2405': 1, 'SHFE.rb.2405': 1},
    active_symbol='SHFE.hc.2405',
    min_volume=1
)


if __name__ == "__main__":
    #创建一个运行环境，并加入策略
    engine = WtEngine(EngineType.ET_HFT)

    #初始化执行环境，传入
    engine.init(folder = '../common/', cfgfile = "config.yaml")

    engine.commitConfig()
    
    #添加Python版本的策略
    straInfo  = hftSpreadTradeBase(name="hc_rb_05",activeleg='SHFE.hc.2405',spread=sp,leg= sp.variable_symbols,expsecs=15, offset=0, freq=30)
    #straInfo = HftStraDemo(name='pyhft_y', code="DCE.y.2403", expsecs=15, offset=0, freq=30)
    engine.add_hft_strategy(straInfo, trader="zhongxinFUT")
    
    #开始运行
    engine.run()

    kw = input('press any key to exit\n')