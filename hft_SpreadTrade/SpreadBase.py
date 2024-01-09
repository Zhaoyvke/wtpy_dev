from wtpy import WtEngine, EngineType
#from strategies.HftStraDemo import HftStraDemo
import pandas as pd
#import SpreadFunction as MF
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from wtpy import HftContext,BaseHftStrategy
from datetime import datetime
from SpreadConstant import Direction, Exchange, Interval, Offset, Status, Product, OptionType, OrderType,TradeData
#from wtpy.WtDataDefs import WtTickRecords
from wtpy.ContractMgr import (ContractInfo,ContractMgr)
from wtpy.ProductMgr import ProductMgr, ProductInfo
from SpreadFunction import floor_to,round_to,ceil_to
#from SpreadEngine import SpreadDataEngine
from dataclasses import dataclass
from wtpy.WtCoreDefs import WTSBarStruct
from zoneinfo import ZoneInfo, available_timezones  
#from tzlocal import get_localzone_name
EVENT_SPREAD_DATA = "eSpreadData"
EVENT_SPREAD_POS = "eSpreadPos"
EVENT_SPREAD_LOG = "eSpreadLog"
EVENT_SPREAD_ALGO = "eSpreadAlgo"
EVENT_SPREAD_STRATEGY = "eSpreadStrategy"



def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)
@dataclass
class PositionData():
    """
    Position data is used for tracking each individual position holding.
    """

    stdCode: str
    exchange: Exchange
    direction: Direction

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0
    yd_volume: float = 0

    #def __post_init__(self) -> None:

        #self.stdCode: str = f"{self.symbol}.{self.exchange.value}"
        #self.vt_positionid: str = f"{self.gateway_name}.{self.stdCode}.{self.direction.value}"

MAX_SPREAD_TIMEOUT = 30 # 有滞后三十秒的腿，不计算价差

#LegData中增加last_dt, 保存leg最后更新时间  （待增加）
class LegData(HftContext):
    """"""
    def __init__(self,stdCode:str,singal_symbol =None) -> None:
        self.context :HftContext
        self.stdCode : str = stdCode                           #用于真实买卖申赎
        #价格及其仓位设置
        self.signal_symbol:str = singal_symbol or self.stdCode #信号标的，只用于计算价格信号
        self.bid_price:float = 0
        self.ask_price:float = 0
        self.bid_volume:float = 0
        self.ask_volume:float = 0
        self.signal_price: float = 0
        self.long_pos:float = 0  #多仓
        self.short_pos:float = 0 #空仓
        self.net_pos:float = 0   #净持仓

        self.last_price: float = 0
        self.net_pos_price:float = 0 #净头寸平均入场价格
        self.last_dt=datetime(year=1900, month=1, day=1, hour=1, minute=0, second=0)
        self.tick :dict = None
        #Contract data 合约data
        self.size: float = 0            #lot_size, 最小成交量，A股100，期货1
        self.net_position: bool = False #净仓位
        self.min_volume : float = 0     #最小仓位
        self.pricetick:   float = 0     #价格跳动

    def update_contract(self,folder:str,stdCode:str,product:ProductInfo,contract:ContractInfo,context:HftContext)->None:
        #Contract相关需要改动    可以加个 isInit(false),在价差模块启动时候读入ProductInfo 
        #self.contractInit = WtEngine.init(folder)
        product=context.stra_get_comminfo(stdCode)
        
        #contract =WtEngine.getContractInfo(stdCode)
        self.min_volume = product.minlots
        self.pricetick = product.pricetick
        #self.size = product.size                  #size 需修改
        #self.net_position = contract.net_position #净头寸  需修改##############################################################
        
    def update_tick(self,tick:dict):

        #if tick["code"] ==self.stdCode:
            self.bid_price =tick["bid_price_0"]
            self.ask_price = tick["ask_price_0"]
            self.bid_volume = tick["bid_qty_0"]
            self.ask_volume = tick["ask_qty_0"]
            self.last_price = tick["price"]

            self.last_dt = tick["time"]
            self.tick = tick
        #    if self.signal_symbol is None or self.signal_symbol == self.stdCode:
        #        self.signal_price = tick["price"]
        #if tick["code"] == self.signal_symbol:
        #    self.signal_price = tick["price"]
       
    def update_position(self,position:PositionData)->None:

        if position.stdCode == self.stdCode:
            position.direction = Direction.NET
            self.net_pos=self.context.stra_get_position(self.stdCode)
            self.net_pos_price = self.context.stra_get_position_avgpx(self.stdCode)#净头寸平均入场价格
            if position.direction == Direction.LONG:
            #if  self.net_pos >0: #多头
                self.long_pos = position.volume
            elif position.direction == Direction.SHORT:
            #elif self.net_pos <0: #空头
                self.short_pos = position.volume
            self.net_pos = self.long_pos - self.short_pos
        else:
            return
        
    def update_trade(self,trade:TradeData)->None:
            #净头寸不为0时候
        if self.net_position:
            trade_cost:float = trade.volume * trade.price
            old_cost:float = self.net_pos * self.net_pos_price #净头寸* 净头寸平均入场价格
            #多头处理
            if trade.direction ==Direction.LONG:
                new_pos:float = self.net_pos + trade.volume 

                if self.net_pos >=0:
                    new_cost = old_cost +trade_cost
                    self.net_pos_price = new_cost / new_pos
                else:
                    #如果之前的空头头寸都已经平仓
                    if not new_pos:
                        self.net_pos_price = 0   #new_pos == 0 执行
                    #如果只有部分空头头寸平仓
                    elif new_pos >0:
                        self.net_pos_price = trade.price
            else:
                new_pos:float = self.net_pos -trade.volume
                
                if self.net_pos <= 0:
                    new_cost = old_cost - trade_cost
                    self.net_pos_price = new_cost /new_pos
                else:
                    #如果之前的多头全已平仓
                    if not new_pos:
                        self.net_pos_price= 0
                    #如果部分多头被平仓
                    elif new_pos < 0:
                        self.net_pos_price = trade.price

            self.net_pos = new_pos
        else:
            #初始态净头寸为0
            if trade.direction ==Direction.LONG:
                if trade.offset == Offset.OPEN:
                    self.long_pos +=trade.volume
                else:
                    self.short_pos -=trade.volume #此处，由于净头寸为0,当不为开仓方向只能为平仓。即向空头-trade.volume
            else:#空头处理
                if trade.offset ==Offset.OPEN:
                    self.short_pos +=trade.volume
                else:
                    self.long_pos -= trade.volume
                
            self.net_pos = self.long_pos - self.short_pos

#在价差的多条腿中，如果有其中一条腿的价格最后跟新时间与当前时间差大于30s, 不再更新价差，等待腿tick更新追平（待增加）
class SpreadData(HftContext):
    
    def __init__(
            self,
            #engine: "SpreadDataEngine",
            name:str,
            legs:List[LegData],
            variable_symbols:Dict[str,str],#[(A: CFFEX.IF.2311), (B: CFFEX.IF.2402)] 变量与stdCode的对应关系
            variable_directions:Dict[str,int],
            variable_signal_symbols: Dict[str, str],#[(A: aaaa.index), (B: bbbb.index)] 变量与价格合成 的对应关系
            variable_direction_buy: Dict[str, Direction],
            variable_direction_sell: Dict[str, Direction],
            price_formula:str,#价格公式
            trading_multipliers:Dict[str,int],#交易乘数
            active_symbol:str,
            min_volume:float,
            compile_formula:bool = True,
            signal_formula:str = None,

    )->None:
        self.name:str = name
        self.compile_formula:bool = compile_formula
        #self.engine = engine#######################
        self.legs:Dict[str,LegData] = {}
        self.active_leg:LegData = None
        self.passive_legs:List[LegData] = []
        self.__tick__ = dict()
        self.min_volume: float = min_volume
        self.pricetick :float = 0
        self.variable_symbols = variable_symbols#待考虑
        self.symbols_variables = {v: k for k, v in variable_symbols.items()}
        #计算价差仓位和分发订单
        self.trading_multipliers:Dict[str,int] = trading_multipliers
        #价格公式和交易公式
        self.price_formula:str = ""
        self.trading_formula:str = ""

        for leg in legs:
            self.legs[leg.stdCode] = leg
            if leg.stdCode ==active_symbol:
                self.active_leg = leg
            else:
                self.passive_legs.append(leg)
            #stdCode作为合约代码Key,获取其value 传trading_multipliers
            trading_multipliers :int  = self.trading_multipliers[leg.stdCode]
           
            """"""""""""""""""""
            if trading_multipliers > 0 :
                self.trading_formula += f"+{trading_multipliers}*{leg.stdCode}"
            else:
                self.trading_formula += f"{trading_multipliers}*{leg.stdCode}"
            """"""""""""""""""""    
                #价格跳动为0
            if not self.pricetick:
                self.pricetick = leg.pricetick
            else:
                self.pricetick = min(self.pricetick,leg.pricetick)

        #Spread data
        self.bid_price :float = 0
        self.ask_price :float = 0
        self.bid_volume : float = 0
        self.ask_volume : float = 0
        self.last_price: float = 0

        # 增加信号价格，根据对应的价格合成
        self.singal_price:float = 0
        self.bid_price_algo:float = 0
        self.ask_price_algo :float = 0

        self.long_pos :int = 0
        self.short_pos :int = 0
        self.net_pos :float = 0 #折合仓位
        
        #是否平衡
        self.is_balance = True          #‘5*CFFEX.IF.2311+2*CFFEX.IF.2402’
        self.price_formula:str = ""
        
        self.datetime:datetime = None
        self.leg_pos:defaultdict = defaultdict(int)#默认leg_pos键值都为整数

        #价差计算公式相关
        self.variable_symbols:dict = variable_symbols
        self.variable_signal_symbols =variable_signal_symbols
        self.variable_direction_buy = variable_direction_buy
        self.variable_direction_sell = variable_direction_sell
        self.variable_direction: dict = variable_directions
        self.price_formula = price_formula
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # 实盘时编译公式，加速计算
        if compile_formula:
            self.price_code: str = compile(price_formula, __name__, "eval")
        # 回测时不编译公式，从而支持多进程优化
        else:
            self.price_code: str = price_formula
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""                              ###########################################

        for leg in legs:
            self.legs[leg.stdCode] = leg
            if leg.signal_symbol != leg.stdCode:
                self.legs[leg.signal_symbol] = leg
            if leg.stdCode == active_symbol:
                self.active_leg = leg
            else:
                self.passive_legs.append(leg)
            
            trading_multiplier = self.trading_multipliers[leg.stdCode]

            leg_var = self.symbols_variables[leg.stdCode]
            if self.variable_direction_buy[leg_var]  == Direction.LONG:
                self.trading_formula +=f"+{trading_multiplier}*{leg.stdCode}"
            else:
                self.trading_formula +=f"-{trading_multiplier}*{leg.stdCode}"
            
            if not self.pricetick:
                self.pricetick = leg.pricetick
            else:
                self.pricetick = min(self.pricetick,leg.pricetick)

        self.price_formula = price_formula ########################################################待取消
        self.price_code = price_formula

        self.signal_formula = signal_formula or price_formula
        self.signal_code = self.signal_formula
        self.variable_legs:Dict[str,LegData] = {} 
        for variable, stdcode in variable_symbols.items():
            leg = self.legs[stdcode]
            self.variable_legs[variable] = leg
        """
        self.variable_signal_legs = {}
        if signal_formula:
            for variable, stdCode in self.variable_signal_symbols.items():
                if stdCode:
                    leg = self.legs.get(stdCode)
                else:
                    leg = self.variable_legs[variable]
                self.variable_signal_legs[variable] = leg
        """
    def get_traded_near_pos(self,stdCode:str):
        #计算某条腿的(折合)持仓量   
        #近似计算
        active_traded = self.leg_pos[stdCode]
        spread_volume = self.calculate_spread_volume(stdCode,active_traded)   
        return spread_volume

    def get_traded_pos(self, stdCode:str):
        #计算某条腿的(折合)持仓量。
        #严格计算
        traded = self.leg_pos[stdCode]
        trading_multiplier = self.trading_multipliers[stdCode]
        spread_volume = traded / trading_multiplier

        return spread_volume

    def calculate_price(self,context:HftContext) ->bool:
        """
        计算价差盘口

        1. 如果各条腿价格均有效，则计算成功，返回True
        2. 反之只要有一条腿的价格无效，则计算失败，返回False

        px：修改，每条腿与当前间相差30s以内，才计算价格
        """
        self.clear_price()

        #遍历所有腿计算价格
        bid_data:dict = {}
        ask_data:dict = {}
        last_price = {}
        signal_price = {}
        bid_data_algo = {}
        ask_data_algo = {}
        volume_inited:bool = False
        
        current_dt = makeTime(self.stra_get_date(), self.stra_get_time(), self.stra_get_secs())

        for variable,leg in self.variable_legs.items():
            if abs((current_dt - leg.last_dt).total_seconds()) > 120:
                context.stra_log_text(f'leg {leg.stdCode} 长时间无行情更新，不更新价差： {leg.last_dt}')
                self.clear_price()
                return False
            #过滤未收到所有腿价格数据Filter not all leg price data has been received
            if not leg.bid_volume or not leg.ask_volume:
                self.clear_price()
                return False
            
            #生成用于计算价差买入价/卖出价的价格字典
            variable_direction: int =self.variable_direction[variable]
            if variable_direction >0:
                bid_data[variable] = leg.bid_price
                ask_data[variable] = leg.ask_price
            else:
                bid_data[variable] = leg.ask_price
                ask_data[variable] = leg.bid_price

            #获取每条腿最新价
            last_price[variable] = leg.last_price
            signal_price[variable] = leg.signal_price

            #计算交易量
            trading_multiplier: int = self.trading_multipliers[leg.stdCode]
            #if 交易乘数==0
            if not trading_multiplier:
                continue

            leg_bid_volume: float = leg.bid_volume
            leg_ask_volume: float = leg.ask_volume

            if trading_multiplier >0:
                adjusted_bid_volume: float = floor_to(leg_bid_volume / trading_multiplier,self.min_volume)
                adjusted_ask_volume: float = floor_to(leg_ask_volume / trading_multiplier,self.min_volume)
            else:
                adjusted_bid_volume: float = floor_to(leg_bid_volume / abs(trading_multiplier),self.min_volume)
                adjusted_ask_volume: float = floor_to(leg_ask_volume / abs(trading_multiplier),self.min_volume)
            
            #对于第一条腿  仅做初始化
            if not volume_inited: #对没初始化做初始化
                self.bid_volume = adjusted_bid_volume
                self.ask_volume = adjusted_ask_volume
                volume_inited = True
            else:
                #For following legs, use min value of each leg quoting volume
                #初始化的腿，用最小量代替
                self.bid_volume = min(self.bid_volume,adjusted_bid_volume)
                self.ask_volume = min(self.ask_volume,adjusted_ask_volume)

        #计算价差价格
        self.bid_price = self.parse_formula(self.price_code,bid_data)
        self.ask_price = self.parse_formula(self.price_code,ask_data)

        if self.pricetick:
            self.bid_price = round_to(self.bid_price,self.pricetick)
            self.ask_price = round_to(self.ask_price,self.pricetick)

        #更新计算时间
        self.datetime = makeTime(self.stra_get_date(), self.stra_get_time(), self.stra_get_secs())
        return True

    def update_trade(self,trade:TradeData)->None:
        #更新委托成交
        if trade.direction ==Direction.LONG:
            self.leg_pos[trade.stdCode] += trade.volume
        else:
            self.leg_pos[trade.stdCode] -=trade.volume

    def calculate_pos(self) ->None:
        long_pos = 0
        short_pos = 0
       
        for n,leg in enumerate(self.legs.values()): #enumerate遍历一个可迭代对象，并同时返回元素的索引和元素本身
            leg_long_pos = 0
            leg_short_pos = 0

            trading_multiplier: int =  self.trading_multipliers[leg.stdCode]
            if not trading_multiplier:#交易乘数为0，跳过
                continue

            net_pos = self.leg_pos[leg.stdCode]
            adjusted_net_pos = net_pos / trading_multiplier #调整净头寸
            #调整后的净头寸大于 0，则将其向上取整到最小交易量
            if adjusted_net_pos >0:
                adjusted_net_pos = floor_to(adjusted_net_pos,self.min_volume)
                leg_long_pos = adjusted_net_pos
            else :
                adjusted_net_pos = ceil_to(adjusted_net_pos,self.min_volume)
                leg_short_pos = abs(adjusted_net_pos)
            
            if not n:# n==0时，进行第一次初始化
                long_pos = leg_long_pos
                short_pos = leg_short_pos
            else:
                long_pos = min(long_pos,leg_long_pos)
                short_pos = min(short_pos,leg_short_pos)
    
        self.long_pos = long_pos
        self.short_pos = short_pos
        self.net_pos = long_pos - short_pos
        #价差的净仓位 = 价差多仓 – 价差空仓。价差的多仓、空仓值在平仓时使用到，即只能平成交实际持有的仓位，不能多平
        
    def clear_price(self)->None: ##############################此处考虑不对价格处理，不能对价格突变

        self.bid_price = 0 #待测试对价格不进行突变 
        self.ask_price = 0
        self.bid_volume = 0
        self.ask_volume = 0
    
    def calculate_leg_volume(self,stdCode:str, spread_volume:float) ->float:
        #计算标的交易volume = 此标的预设交易乘数 * 价差交易量
        leg:LegData = self.legs[stdCode]
        trading_multiplier:int = self.trading_multipliers[leg.stdCode]
        leg_volume:float = spread_volume * trading_multiplier

        return leg_volume

    def calculate_spread_volume(self,stdCode:str,leg_volume:float) ->float:
        """
        近似计算，以min_volume取整
        :param stdCode:
        :param leg_volume:  已经交易的量
        :return: 换算成这条腿已经交易的量
        """
        leg:LegData = self.legs[stdCode]
        trading_multipliter:int = self.trading_multipliers[leg.stdCode]
        spread_volume :float =  leg_volume /trading_multipliter

        # 1.2, 取1， -1.2 取-2
        if spread_volume >0:
            spread_volume = floor_to(spread_volume,self.min_volume)
        else:
            spread_volume = ceil_to(spread_volume,self.min_volume)

        return spread_volume
    
    def to_tick(self,stdCode:str,tick:dict):#待修改
        self.__tick__[stdCode] = tick 
        self.__tick__[self.name] = tick(
        code = self.name,
        exchg = Exchange.LOCAL,
        action_date=self.datetime,
        last_price=round_to((self.bid_price + self.ask_price) / 2,  self.pricetick),
        bid_price_0= self.bid_price,
        ask_price_0= self.ask_price,
        bid_volume_0= self.bid_volume,
        ask_volume_0= self.ask_volume,
)
        return self.__tick__[stdCode]
    

    def to_signal_tick(self,stdCode:str,tick:dict):
        self.__tick__[stdCode] = tick(size=4) 
        self.__tick__[self.name] = tick(
            code = self.name,
            exchg = Exchange.LOCAL,
            action_date=self.datetime,
            #name = ,
            last_price=self.signal_price,
        )
        return self.__tick__[stdCode]

    
    def get_leg_size(self,stdCode:str) ->float: #获取最小成交量
           
        leg:LegData = self.legs[stdCode]
        return leg.size


    def parse_formula(self, formula: str, data: Dict[str, float]) -> Any:
        ##解析一个公式（formula），并使用给定的数据（data）来计算公式的值。该函数使用了eval()函数来执行给定的公式，并将计算结果存储在变量value中。然后，函数返回value的值。在执行eval()函数时，Python 会将公式中的变量和数据字典中的值进行替换，从而计算出公式.此外，该函数还使用了locals()函数来更新当前作用域中的变量。这意味着，在执行公式时，公式中使用的变量可以直接访问数据字典中的值，而无需在公式中显式地引用这些变量。
    
        locals().update(data)
        value = eval(formula)
       
        return value      


    """
    def __post_init__(self) -> None:
        """"""
        self.stdCode: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
    
        #Check if the order is active.
        
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        
        #Create cancel request object from order.
        
        req: CancelRequest = CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )
        return req
     """

#回测部分未添加



            

