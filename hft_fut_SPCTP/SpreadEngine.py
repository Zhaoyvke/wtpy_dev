from wtpy import WtEngine, EngineType
#from strategies.HftStraDemo import HftStraDemo
import pandas as pd
import SpreadFunction as MF
from SpreadBase import LegData ,SpreadData,PositionData
from wtpy.ProductMgr import ProductMgr, ProductInfo
from SpreadConstant import load_json,save_json
from wtpy.ContractMgr import ContractInfo
from wtpy import HftContext,BaseHftStrategy
import traceback
import importlib
import os
from types import ModuleType
from typing import List, Dict, Set, Callable, Any, Optional
from collections import defaultdict
from copy import copy
from pathlib import Path
from datetime import datetime, timedelta
from SpreadConstant import Direction, Exchange, Interval, Offset, Status, Product, OptionType, OrderType,TradeData
from wtpy.WtCoreDefs import WTSBarStruct, WTSTickStruct
#from wtpy.WtDataDefs import WtTickRecords
from wtpy.wrapper import WtWrapper
from SpreadAlgo import SpreadTakerAlgo
from SpreadTemplate import SpreadAlgoTemplate
from event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_QUOTE
)

APP_NAME = "SpreadTrading"

class SpreadEngine(BaseHftStrategy):
    """"construcctor"""
    def _init_(self,main_engine:WtEngine,context:HftContext)->None:
        super().__init__(main_engine)

        self.active: bool = False
        self.main_engine :WtEngine(self)
        self.data_engine: SpreadDataEngine = SpreadDataEngine(self)
        self.algo_engine: SpreadAlgoEngine = SpreadAlgoEngine(self)
        self.strategy_engine:main_engine(self)

        self.add_spread = self.data_engine.add_spread
        self.remove_spread = self.data_engine.remove_spread
        self.get_spread = self.data_engine.get_spread
        self.get_all_spread = self.data_engine.get_all_spreads

        self.start_algo = self.algo_engine.start_algo
        self.stop_algo = self.algo_engine.stop_algo
        self.write_log = context.stra_log_text

    def start(self)->None:
        """"""
        if self.active:
            return
        self.active = True

        self.data_engine.start()
        self.algo_engine.start()
        #self.strategy_engine.start()
    
class SpreadDataEngine(HftContext):
    """
    负责价差的创建、删除、价差配置的维护
    负责价差bar的更新调度
    """
    setting_filename: str = "spread_trading_setting.json"
    pos_filename: str = "spread_trading_pos.json"
    var_filename = "spread_trading_var.json"
    def __init__(self,spread_engine:SpreadEngine,context:HftContext)->None:

        self.spread_engine:SpreadEngine = spread_engine
        self.main_engine:WtEngine = spread_engine.main_engine
        #self.event_engine: EventEngine = spread_engine.event_engine
        self.write_log = context.stra_log_text

        self.legs: Dict[str,LegData] = {}   # stdCode: leg
        self.all_legs: Dict[str, List[LegData]] = defaultdict(list)
        self.spreads: Dict[str, SpreadData] = {}    # name: spread
        #defaultdict默认字典的作用是在首次访问字典中不存在的键时，自动创建一个与该键关联的空列表。
        self.symbol_spread_map:Dict[str,List[SpreadData]] = defaultdict(list)
        self.order_spread_map:Dict[str,SpreadData] ={}

        #self.ticks: Dict[str, WTSTickStruct] = {}
        self.ticks: Dict[str, dict] = {}
        self.trades: Dict[str, TradeData] = {}
        self.positions: Dict[str, PositionData] = {}
        self.contracts: Dict[str, ContractInfo] = {}

        self.tradeid_history:Set[str] = set()
        self.persistent_var = {}
        self.var_loaded = False
    def start(self)->None:
        
        self.load_setting()
        self.load_pos()

        self.write_log("价差数据引擎启动成功！")

    #仅作为价差模块用，后续待优化至SpreadEngine模块
    def load_setting(self)->None:
        setting: dict = load_json(self.setting_filename)

        for spread_setting in setting:
            for leg_setting in spread_setting["leg_settings"]:
                leg_setting['buy_direction'] = Direction(leg_setting['buy_direction'])
                leg_setting['sell_direction'] = Direction(leg_setting['sell_direction'])
            self.add_spread(
                name=spread_setting["name"],
                leg_settings=spread_setting["leg_settings"],
                price_formula=spread_setting["price_formula"],
                signal_formula=spread_setting.get("signal_formula", None),
                active_symbol=spread_setting["active_symbol"],
                min_volume=spread_setting.get("min_volume", 1),
                basket_leg_name=spread_setting.get("basket_leg_name", None),
                price_level=spread_setting.get("price_level", 1),
                save=False
            )

    def save_setting(self)->None:

        setting:List = []

        for spread in self.spreads.values():
            leg_settings:List =[]
            for variable,stdCode in spread.variable_symbols.items():
                trading_direction:int = spread.variable_direction[variable]
                trading_multiplier:int = spread.trading_multipliers[stdCode]

                leg_setting:dict ={
                    "variable":variable,
                    "stdCode":stdCode,
                    "trading_direction":trading_direction,
                    "trading_multiplier":trading_multiplier
                }
                leg_settings.append(leg_setting)

            Spread_setting:dict = {
                "name":spread.name,
                "leg_settings":leg_settings,
                "price_formula":spread.price_formula,
                "active_symbol":spread.active_leg.stdCode,
                "min_volume":spread.min_volume
            }

            setting.append(Spread_setting)

        save_json(self.setting_filename,setting)

    def save_pos(self)->None:
        #保存价差持仓;待同步到spread_engine
        pos_data:dict = {}

        for spread in self.spreads.values():
            pos_data[spread.name] = spread.leg_pos

        save_json(self.pos_filename,pos_data)

    def save_var(self):
        """
        保存策略在循行过程中的中间变量，如成本价, 每秒主动保存一次
        :return:
        """
        if not self.var_loaded:
            return
        all_var = {}
        for k, strategy in self.spread_engine.strategy_engine.strategies.items():
            v = strategy.get_var()
            all_var[k] = v

        if all_var != self.persistent_var:
            save_json(self.var_filename, all_var)
            self.persistent_var = all_var

    def load_pos(self)->None:
        #加载价差持仓；待同步到spread_engine
        pos_data:dict = load_json(self.pos_filename)

        for name,leg_pos in pos_data.items():
            spread:SpreadData = self.spreads.get(name,None)
            if spread:
                spread.leg_pos.update(leg_pos)
                spread.calculate_pos()
    """
    #注册事件部分需要再更正 参考WtSpradEngine 
    # #注册EVENT_TICK、EVENT_TRADE、EVENT_POSITION三个事件，该函数在类StDataEngine的__init__中调用。
    #def register_event(self) -> None: 

    # #发出价差行情更新事件： 根据函数processTickEvent中的vtSymbol的tick变化，调用该函数发出价差行情更新事件。
    def process_tick_event(self)  ->None:
        """"""
        tick:WtTickRecords = self.ticks.values
        leg: LegData = self.legs.get(tick["code"],None)
        if not leg:
            return
        leg.update_tick(tick)
        
        for spread in self.symbol_spread_map[tick["code"]]:
            #只有能计算出价差盘后时，才推送事件
            if spread.calculate_price():
                pass
                #self.put_data_event(spread) ###################################################################
   #处理持仓推送： 如果成交的标的vtSymbol在腿字典legDict中，则根据vtSymbol的成交方向更新腿的longPos、shortPos、netPos；调用价差对象的函数spread.calculatePos更新价差持仓，推送价差持仓更新。
    def process_position_event(self)->None:
        position:PositionData =self.positions

        leg:LegData = self.legs.get(position.stdCode,None)
        if not leg:
            return 
        leg.update_positon(position)

        for spread in self.symbol_spread_map[position.stdCode]:
            spread.calculate_pos()
            #self.put_pos_event(spread) ###################################################################

    #处理成交推送： 如果成交的标的vtSymbol在腿字典legDict中，则根据vtSymbol的实际成交数量、成交方向计算腿的实际成交，更新腿持仓的longPos、shortPos和netPos；
    # 调用价差对象的函数spread.calculatePos更新价差持仓，推送价差持仓更新。
    def process_trade_event(self) -> None:
        """"""
        trade:TradeData = self.trades
        if trade.vt_tradeid in self.tradeid_history:
            return 
        self.tradeid_history.add(trade.vt_tradeid)
        #查询该笔成交，对应差价，并更新计算价差持仓
        spread:SpreadData = self.order_spread_map.get(trade.vt_orderid,None)
        if spread:
            spread.update_trade(trade)
            spread.calculate_pos()
            self.put_pos_event(spread)

            self.save_pos()

    def process_contract_event(self) -> None:

        contract :ContractInfo = self.contracts
        leg:LegData = self.legs.get(contract.code,None)

        if leg:
            #更新合约数据
            leg.update_contract(contract)
            
            #req: SubscribeRequest = SubscribeRequest(
            #    contract.symbol, contract.exchange
            #)
            #self.main_engine.subscribe(req, contract.gateway_name)
            
        
    def put_data_event(self, spread: SpreadData) -> None:
        pass
        ###################################################################
    def put_pos_event(self, spread: SpreadData) -> None:
        pass
    """
    def get_leg(self,stdCode:str,context:HftContext)->LegData:

        leg:LegData = self.legs.get(stdCode,None)

        if not leg:
            leg = LegData(stdCode)
            self.legs[stdCode] = leg

            legs = self.all_legs[stdCode]
            if leg not in legs:
                legs.append(leg)
            #订阅Leg部分
            contract:Optional[ProductInfo] = context.stra_get_comminfo(stdCode) 
            if contract:
                leg.update_contract(contract)
                context.stra_sub_ticks(stdCode)
            #初始化 leg posotion
            positions :List[PositionData] =self.get_all_positions()
            for positon in positions:
                if positon.stdCode == stdCode:
                    leg.update_position(positon) ####################################
        
        return leg


    def get_all_positions(self,stdCode) -> List[PositionData]:
        
        position:PositionData =self.positions
        #Get all position data.
        PositionData.stdCode = position.stdCode
        #PositionData.exchange = position.exchange
        PositionData.exchange =position.stdCode[:stdCode.index('.')] #获取stdCode的. 前的内容，即交易所代码
        PositionData.volume =HftContext.stra_get_position(stdCode)
        
        return list(self.positions.values())

    def add_spread(
            self,
            name:str,
            leg_settings:List[Dict],
            price_formula:str,
            active_symbol:str,
            min_volume:float,
            save:bool=True
    )->None:
        """
        添加价差
        :param name:
        :param leg_settings: [{'variable': 'A', 'vt_symbol': 'IH2112.CFFEX', 'trading_direction': '空‘,
        'trading_multiplier': ’1‘， ‘signal_symbol’: 'aaaa.index'},
        {'variable': 'B', 'vt_symbol': 'IH2203.CFFEX', 'trading_direction': -1, 'trading_multiplier': -1,
        signal_symbol’: 'bbbb.index'}}]
        :param price_formula: 价差公式 如 A - 5*B - 0.5*C
        :param signal_formula: 信号价格计算公式 如 A-B
        :param active_symbol: 主动腿symbol
        :param min_volume:
        :param basket_leg_name:
        :param save: 是否保存到json文件
        :return:
        """
        if name in self.spreads:
            self.write_log("价差创建失败，名称重复：{}".format(name))
            return 
        
        legs:List[LegData] = []
        variable_symbols:Dict[str,str] = {}
        variable_directions:Dict[str,int] = {}
        variable_directions_buy: Dict[str, Direction] = {}
        variable_directions_sell: Dict[str, Direction] = {}
        trading_multipliers :Dict[str,int] = {}

        for leg_setting in leg_settings:
            stdCode: str = leg_setting["stdCode"]
            variable:str = leg_setting["variable"]
            leg: LegData = self.get_leg(stdCode) #补充

            legs.append(leg)
            variable_symbols[variable] = stdCode
            variable_directions[variable] = leg_setting["trading_directions"]
            variable_directions_buy[variable] = Direction(leg_setting["buy_direction"])
            variable_directions_sell[variable] = Direction(leg_setting["sell_direction"])
            trading_multipliers[stdCode] = leg_setting["trading_multipliers"]

        spread: SpreadData = SpreadData(
            engine=self,
            name=name,
            legs=legs,
            variable_symbols=variable_symbols,
            #variable_signal_symbols=variable_signal_symbol,
            variable_directions_buy=variable_directions_buy,
            variable_directions_sell=variable_directions_sell,
            price_formula=price_formula,
            #signal_formula=signal_formula,
            trading_multipliers=trading_multipliers,
            active_symbol=active_symbol,
            min_volume=min_volume
        )
        self.spreads[name] = spread
        self.symbol_spread_map: Dict[str, List[SpreadData]] = defaultdict(list) 
        for leg in spread.legs.values():
            self.symbol_spread_map[leg.stdCode].append(spread)########################################################

        if save:
            self.save_setting()     #待补充

        self.write_log("价差创建成功：{}".format(name))
        self.put_data_event(spread)   #待补充 
        self.spread_engine.algo_engine.spreads[spread.name] = spread
    def remove_spread(self,name:str)->None:

        if name not in self.spreads:
            return 
        
        spread:SpreadData = self.spreads.pop(name)
        
        for leg in spread.legs.values():
            self.symbol_spread_map[leg.stdCode].remove(spread)

        self.save_setting()
        self.write_log("价差移除成功：{}，重启后生效".format(name))

    def get_spread(self,name:str)->Optional[SpreadData]:
        
        spread:SpreadData =self.spreads
        return spread
    
    def get_all_spreads(self)->List[SpreadData]:

        return list(self.spreads.values())

    def update_order_spread_map(self, vt_orderid: str, spread: SpreadData) -> None:
        """更新委托号对应的价差映射关系"""
        self.order_spread_map[vt_orderid] = spread
    

class SpreadAlgoEngine:
    """
    下单算法引擎
    """    
    algo_class:SpreadTakerAlgo = SpreadTakerAlgo

    def __init__(self,Spread_engine:SpreadEngine) -> None:

        self.spread_engine:SpreadEngine = Spread_engine
        self.data_engine: SpreadDataEngine = Spread_engine.data_engine
        self.main_engine :WtEngine = Spread_engine.main_engine
        #self.event_engine: EventEngine = spread_engine.event_engine######################

        self.write_log = Spread_engine.write_log

        self.spreads:Dict[str,SpreadData] = {}
        self.algos:Dict[str,SpreadAlgoTemplate] = {}

        self.order_algo_map:Dict[str,SpreadAlgoTemplate] = {}
        self.symbol_algo_map:Dict[str,List[SpreadAlgoTemplate]] = defaultdict(list)

        self.algo_count:int = 0
        self.vt_tradeids:set = set() 



    def start(self) -> None:
        """"""
        self.register_event()

        self.write_log("价差算法引擎启动成功")

    def stop(self) -> None:
        """"""
        for algo in self.algos.values():
            self.stop_algo(algo)

    def call_strategy_func(
            self,strategy:BaseHftStrategy,func:Callable,params:Any = None
    )->None:
        """
        Call function of a strategy and catch any exception raised.
        """
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            strategy.trading = False
            strategy.inited = False

            msg: str = f"触发异常已停止\n{traceback.format_exc()}"
            self.write_log(strategy, msg)

    def register_event(self) -> None:
        """
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)
        self.event_engine.register(
            EVENT_SPREAD_DATA, self.process_spread_event
        )
        """
        """
    def process_spread_event(self, event: Event) -> None:
        """"""
        spread: SpreadData = event.data
        self.spreads[spread.name] = spread

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data
        algos: List[SpreadAlgoTemplate] = self.symbol_algo_map[tick.stdCode]
        if not algos:
            return

        buf: List[SpreadAlgoTemplate] = copy(algos)
        for algo in buf:
            if not algo.is_active():
                algos.remove(algo)
            else:
                algo.update_tick(tick)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data

        algo: SpreadAlgoTemplate = self.order_algo_map.get(order.vt_orderid, None)
        if algo and algo.is_active():
            algo.update_order(order)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data

        # Filter duplicate trade push
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        algo: SpreadAlgoTemplate = self.order_algo_map.get(trade.vt_orderid, None)
        if algo and algo.is_active():
            algo.update_trade(trade)

    def process_timer_event(self, event: Event) -> None:
        """"""
        buf: List[SpreadAlgoTemplate] = list(self.algos.values())

        for algo in buf:
            if not algo.is_active():
                self.algos.pop(algo.algoid)
            else:
                algo.update_timer()



    #考虑放弃该函数
    def send_order(
            self,
            algo:SpreadAlgoTemplate,
            stdCode:str,
            price:float,
            volume:float,
            direction:Direction,
            lock:bool,
            fak:bool,
            fok:bool
    )->List[str]:
        #创建原始委托请求
        contract:Optional[ContractInfo] = self.get_contract(stdCode)

        if fak:
            order_type:OrderType = OrderType.FAK
        else:
            order_type:OrderType = OrderType.LIMIT

        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=Offset.OPEN,
            type=order_type,
            price=price,
            volume=volume,
        )
        # 判断使用净仓还是锁仓模式
        net: bool = not lock

        # 执行委托转换
        req_list: List[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            lock,
            net
        )
        # Send Orders
        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(
                req, contract.gateway_name)

            # Check if sending order successful
            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            # Save relationship between orderid and algo.
            self.order_algo_map[vt_orderid] = algo

            # 将委托号和价差的关系缓存下来
            self.data_engine.update_order_spread_map(vt_orderid, algo.spread)

        return vt_orderids

    #考虑放弃
    def cancel_order(self, algo: SpreadAlgoTemplate, vt_orderid: str) -> None:
        """"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_algo_log(algo, "撤单失败，找不到委托{}".format(vt_orderid))
            return
        """
    def get_tick(self,stdCode:str,context:HftContext) :
        """"""
        return context.stra_get_ticks(stdCode)
    
    def get_contract(self, stdCode: str,context:HftContext) -> Optional[ContractInfo]:
        """"""
        return context.stra_get_comminfo(stdCode)
        


#放弃
    """
class OrderData():
    symbol: str
    exchange: Exchange
    orderid: str

    type: OrderType = OrderType.LIMIT
    direction: Direction = None
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        #self.stdCode: str = f"{self.symbol}.{self.exchange.value}"
        #self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
    """
    """
    def is_active(self) -> bool:
       
        Check if the order is active.
        
        return self.status in ACTIVE_STATUSES
    """
    """
    def create_cancel_request(self) -> "CancelRequest":
        
        req: CancelRequest = CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )
        return req
        """
    """
#考虑放弃该函数
class OrderRequest:
    symbol: str
    exchange: Exchange
    direction: Direction
    type: OrderType
    volume: float
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:

        self.stdCode: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:

        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return order
    """