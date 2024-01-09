from collections import defaultdict
from typing import Dict, List, Set, Callable, TYPE_CHECKING, Optional
from copy import copy
from wtpy import WtEngine, EngineType
from wtpy.wrapper import WtWrapper
from wtpy import HftContext,BaseHftStrategy
from SpreadBase import (
     TradeData,SpreadData, LegData
)
import datetime
from wtpy.ProductMgr import ProductMgr, ProductInfo
from SpreadConstant import Direction, Status, Offset,OrderData
from SpreadFunction import  floor_to, ceil_to, round_to
from wtpy.ContractMgr import ContractMgr, ContractInfo
if TYPE_CHECKING:
    from .SpreadEngine import SpreadAlgoEngine

#待补充锁仓委托转换
#在有今仓的情况下，如果想平仓，则会先平掉所有的昨仓，然后剩下的部分都进行反向开仓来代替平今仓，以避免平今的手续费惩罚）

class SpreadAlgoTemplate(HftContext,BaseHftStrategy):
    """
    Template for implementing spread trading algos.
    算法
    • 负责价差交易的执行
    • 一条主动腿，一条/多条被动腿
    """
    algo_name :str = "AlgoTemplate"

    def __init__(
            self,
            algo_engine:"SpreadAlgoEngine",
            algoid:str,
            spread:SpreadData,
            direction:Direction,
            price:float,
            volume:float,
            active_payup: int,
            passive_payup: int,
            payup:int,
            interval:int,
            lock:bool,
            extra:dict         
            ) -> None:
        """
        :param algo_engine:
        :param algoid:
        :param spread:
        :param direction: 方向
        :param price:   价格
        :param volume:  数量
        :param active_payup: 主动腿超价， +更容易成交，-更困难
        :param passive_payup: 被动腿超价， +更容易成交，-更困难
        :param interval:    撤单间隔
        :param lock:        模式
        :param extra:
        """
        self.algo_engine:"SpreadAlgoEngine" = algo_engine
        self.algo_start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.algo_engine: "SpreadAlgoEngine" = algo_engine
        self.algoid: str = algoid

        self.spread: SpreadData = spread
        self.spread_name: str = spread.name

        self.direction: Direction = direction
        self.price: float = price
        self.volume: float = volume
        self.active_payup: int = active_payup
        self.passive_payup: int = passive_payup
        self.interval = interval
        if not passive_interval:
            passive_interval = interval
        self.passive_interval = passive_interval
        self.lock = lock

        if direction ==Direction.LONG:
            self.target = volume
            self.side = 'add_pos'
        else:
            self.target = -volume
            self.side = 'sub_pos'
        # 表示当前的order订单是主动腿还是被动腿
        #print('加仓减仓判断：',self.target,self.side)
        self.current_order_is_active = True
        self.status:Status = Status.NOTTRADED #算法状态
        self.count: int = 0                   #读秒计数
        self.traded: float =0                 #成交数量
        self.traded_volume:float =0           #成交数量（绝对值）
        self.traded_price:float =0            #成交价格
        self.stopped:bool =False              #是否已被用户停止算法
        self.count: int = 0                     # 读秒计数
        self.traded: float = 0                  # 成交数量

        self.leg_traded:defaultdict =defaultdict(float)
        self.leg_cost: defaultdict =defaultdict(float)  #if not find,返回0.0
        self.leg_orders:defaultdict = defaultdict(list) #if not find,返回空列表

        self.order_trade_volume: defaultdict = defaultdict(int)
        self.oreders:Dict[str,OrderData] = {}
        self.write_log = HftContext.stra_log_text
        self.get_contract = HftContext.stra_get_comminfo
        self.is_balance = True
        self.calculate_traded_volume()

        self.write_log("算法已经启动")

    def is_active(self)->bool:
        """判断算法是否处于运行中"""
        if self.status not in [Status.CANCELLED, Status.ALLTRADED]:
            return True
        else:
            return False
    """ 
    def is_order_finished(self)->bool:
        finished:bool = True

        for leg in self.spread.legs.values():
            vt_orderids:list = self.leg_orders[leg.stdCode] #采用下单函数返回后的List

            if vt_orderids:
                finished = False
                break
        return finished
    """
    """
    def is_hedge_finished(self)->bool:
        #return: (是否已经平衡， 主动腿已经成交的单位量)
        active_symbol:str = self.spread.active_leg.stdCode
        active_traded:float = self.leg_traded[active_symbol]

        spread_volume :float = self.spread.calculate_spread_volume(
            active_symbol,active_traded
        )

        finished:bool =True

        for leg in self.spread.passive_legs:
            passive_symbol:str = leg.stdCode

            leg_target:float = self.spread.calculate_leg_volume(
                passive_symbol,spread_volume
            )
            leg_traded:float = self.leg_traded[passive_symbol]

            if leg_target >0 and leg_traded <leg_target:
                finished = False
            elif leg_target <0 and leg_traded >leg_target:
                finished = False
            
            if not finished:
                break
        
        return finished
    """
    def check_algo_cancelled(self)->None:
        """检查算法是否已停止"""
         # 检查algo是否可以停止之前，要计算一次是否平衡，避免因为trade刚回来，还没有新的tick回来导致的平衡未更新
        if(self.stopped
           and self.is_order_finished()
           and self.is_hedge_finished()
           ):
            self.status = Status.CANCELLED
            self.write_log("算法已经停止...")
            self.putevent()############################################################################

    def stop(self)->None:
        if not self.is_active(): #先判断是否在执行中
            return
        
        self.write_log("算法停止中...")
        self.cancel_all_order()####################

        self.check_algo_cancelled()

    def cancel_all_order(self)->None:

        for stdCode in self.leg_orders.keys():
            undone=self.stra_get_undone(stdCode)
            isbuy= (undone>0)
            self.stra_cancel_all(stdCode,isbuy)

    def update_tick(self,stdCode:str,tick:dict)->None:

        self.on_tick(stdCode,tick)
    """
    def update_trade(self,trade:TradeData)->None: ##############################需更替
 
        trade_volume:float = trade.volume

        if trade.direction ==Direction.LONG:
            self.leg_traded[trade.stdCode]+=trade_volume
            self.leg_cost[trade.stdCode]+=trade_volume * trade.price
        else:
            self.leg_traded[trade.stdCode]-=trade_volume
            self.leg_cost[trade.stdCode]-= trade_volume *trade.price
        
        self.calculate_traded_volume()
        self.calculate_traded_price()

        #汇总每个订单的所有交易量
        self.order_trade_volume[trade.vt_orderid] += trade.volume

        #如果订单都交易了，移除主动腿List
        order:OrderData = self.oreders[trade.vt_orderid]
        contract:ProductInfo = self.stra_get_comminfo(trade.stdCode)
        #product:Optional[ProductInfo] = self.stra_get_comminfo(trade.stdCode)
        trade_volume = round_to(
            self.order_trade_volume[order.vt_orderid],contract.minlots #最小交易数量
        )

        if trade_volume ==order.volume:
            vt_orderids:list  =self.leg_orders[order.stdCode]
            if order.vt_orderid in vt_orderids:
                vt_orderids.remove(order.vt_orderid)

        msg: str = "委托成交[{}]，{}，{}，{}@{}".format(
            trade.vt_orderid,
            trade.stdCode,
            trade.direction.value,
            trade.volume,
            trade.price
        )
        self.write_log(msg)
        
        self.put_event()                          ##############################
        self.on_trade(trade)
        """
    """
    def calculate_traded_volume(self)->None:#计算已成交价差数量

        self.traded =0
        spread:SpreadData = self.spread

        n:int =0
        for leg in spread.legs.values():
            leg_traded:float = self.leg_traded[leg.stdCode]
            trading_multiplier:int = spread.trading_multipliers[leg.stdCode]
            if not trading_multiplier:
                continue

            adjusted_leg_traded:float = leg_traded / trading_multiplier
            adjusted_leg_traded = round_to(adjusted_leg_traded,spread.min_volume)

            if adjusted_leg_traded > 0:
                adjusted_leg_traded = floor_to(adjusted_leg_traded,spread.min_volume)
            else:
                adjusted_leg_traded = ceil_to(adjusted_leg_traded,spread.min_volume)

            if not n :
                self.traded = adjusted_leg_traded
            else:
                if adjusted_leg_traded >0:
                    self.traded = min(self.traded,adjusted_leg_traded)
                elif adjusted_leg_traded <0:
                    self.traded = max(self,adjusted_leg_traded)
                else:
                    self.traded = 0
            n+=1
        
        self.traded_volume = abs(self.traded)

        if self.target > 0 and self.traded >=self.target:
            self.status = Status.ALLTRADED
        elif self.target < 0 and self.traded <= self.target:
            self.status = Status.ALLTRADED
        elif not self.traded:
            self.status = Status.NOTTRADED
        else:
            self.status = Status.PARTTRADED
    """        
    """
    #计算已成交价差均价
    def calculate_traded_price(self)->None:

        self.traded_price = 0
        spread:SpreadData =self.spread

        data:dict = {}

        for variable,stdCode in spread.variable_symbols.items():
            leg:LegData = spread.legs[stdCode]
            trading_multiplier:int = spread.trading_multipliers[leg.stdCode]

            # Use last price for non-trading leg (trading multiplier is 0)

            if not trading_multiplier:
                data[variable] = leg.tick["price"]
            else:
            # If any leg is not traded yet, clear data dict to set traded price to 0
                leg_traded:float = self.leg_traded[leg.stdCode]
                if not leg_traded:
                    data.clear()
                    break
            
                leg_cost:float = self.leg_cost[leg.stdCode]
                data[variable] = leg_cost / leg_traded

            if data:
                 self.traded_price = spread.parse_formula(spread.price_code,data)
                 self.traded_price = round_to(self.traded_price,spread.pricetick)
            else:
                self.traded_price = 0
    """           
    #用于hft中，用发单逻辑代替执行器
    def send_order(
        self,
        stdCode:str,
        price:float,
        volume:float,
        direction:Direction,
        fak:bool = False,
        fok:bool = False,
    )->None:
        #如果进入停止任务，禁止主动腿发单      多加校验；当前发单是否为主动腿代码单
        if self.stopped and stdCode ==self.spread.active_leg.stdCode:
            return
        #四舍五入下单量和最小变动价格
        leg :LegData = self.spread.legs[stdCode]
        volume:float = round_to(volume,leg.min_volume)
        price:float = round_to(price,leg.pricetick)

        #检查价格是否超过涨跌板
        tick = self.stra_get_ticks(stdCode) 

        if direction ==Direction.LONG and tick['upper_limit']:
            price == min(price,tick['upper_limit'])
        elif direction == Direction.SHORT and tick['lower_limit']:
            price == max (price,tick['lower_limit'])

        # Otherwise send order
        vt_orderids:list = self.algo_engine.send_order( #处理下单不用该函数
            self,
            stdCode,
            price,
            volume,
            direction,
            self.lock,
            fak
        )
        self.leg_orders[stdCode].extend(vt_orderids)
        msg: str = "发出委托[{}]，{}，{}，{}@{}".format(
            "|".join(vt_orderids),
            stdCode,
            direction.value,
            volume,
            price
        )
        self.write_log(msg)
"""
SpreadTakerAlgo（见价下单算法）¶
算法原理¶
见价下单：主动腿以对价先行下单；被动腿以对价立即对冲

算法收到tick数据推送：先检查委托是否结束，再检查对冲是否结束，若未结束则发起被动腿对冲。最后检查盘口是否满足条件，满足条件后发出主动腿委托

算法收到委托回报：若收到主动腿已结束的委托，则发起被动腿对冲

超时限制：到达计时时间执行委托全撤

算法优势¶
灵活且不占用过多撤单次数

不足¶
所有腿均需要付出盘口买卖价差的滑点成本

等待主动腿对价满足条件，需要比Maker更长的时间

SpreadMakerAlgo（报价做市算法）¶
算法原理¶
报价做市：基于被动腿盘口，计算主动腿最差成交价

算法收到tick数据推送：先检查委托是否结束，再检查对冲是否结束，若未结束则发起被动腿对冲。然后检查新的挂单价格与当前已挂价格的差值是否超过设定的阈值，如果未超过则发出主动腿委托，如果超过则重挂

算法收到委托回报：遭遇拒单则停止策略；若收到主动腿已结束的委托，则清空挂单价格记录

算法收到成交回报：只关心主动腿委托，若对冲未结束则发起被动腿对冲

计时时间到达：到达计时时间执行委托全撤

算法优势¶
主动腿报价挂单做市，目标是赚得盘口价差的同时提高成交概率

不足¶
虽然有设定挂单阈值限制，但撤单行为比Taker更频繁，需要仔细监控委托流量费成本

SpreadExchangeAlgo（交易所价差算法）¶
算法原理¶
基于交易所提供的价差组合来创建价差，采用各条腿的行情盘口计算价差盘口，最终采用交易所的价差合约进行交易执行

算法收到tick数据推送：检查是否已发出了委托，若已发出则返回。再检查价差套利合约是否生成成功，并查询价差套利合约的合约信息，然后发出交易所价差委托，缓存委托号和价差关系，最后输出日志并记录委托已经发出的状态

算法优势¶
体验上类似单合约，且免去了主动腿撤单

不足¶
缺乏灵活性，可选择合约范围有限
"""

   
