from typing import TYPE_CHECKING, Optional
from wtpy import WtEngine, EngineType
from SpreadConstant import Direction
from SpreadBase import   TradeData,SpreadData, LegData
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
class SpreadAlgo(SpreadAlgoTemplate,HftContext,BaseHftStrategy):

    def __init__(
        self,
        algo_engine: "SpreadAlgoEngine",
        algoid: str,
        spread: SpreadData,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        active_payup: int,
        passive_payup: int,
        interval: int,
        lock: bool,#考虑不需要
        extra: dict
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
        '''内部数据'''
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪

        self.algo_engine:"SpreadAlgoEngine" = algo_engine
        self.algo_start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.algoid: str = algoid
        self.legs: Dict[str, LegData] = {}          # vt_symbol: leg
        self.spreads: Dict[str, SpreadData] = {}    # name: spread
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
        self.is_balance = True
        self.calculate_traded_volume()

    def get_spread_tick(self):

        return self.spread.to_tick()
    
    #执行买卖价差合约
    def do_calc(self,newTick:dict)->None:

        # Return if there are any existing orders 
        if not self.is_order_finished():
            return
        
        self.__last_tick__ = newTick
        # Hedge if active leg is not fully hedged
        if not self.is_hedge_finished():
            self.hedge_passive_legs()
            return 
        
        # return if tick not inited 
        #价差买量卖量其中之一未初始化
        if not self.spread.bid_volume or not self.spread.ask_volume:
            return  
        
        #   Otherwise check if should take active leg
        if self.direction == Direction.LONG :
            if self.spread.ask_price <= self.price:  #做多价差时候，价差卖价 <=price
                self.take_active_leg()
        else:
            if self.spread.bid_price >=self.price: #做空价差时候，价差买价 >=price
                self.take_active_leg()
    #在on_order回调后处理   
    def by_order(self,order:OrderData,newTick:dict)->None:
        
        #Only care active leg order update
        if order.stdCode !=self.spread.active_leg.stdCode:
            return
        
        # Do nothing if still any existing orders       
        if not self.is_order_finished():
            return
        
        #Hedge passive legs if necessary
        if not self.is_hedge_finished(): #未对冲完
            self.hedge_passive_legs(newTick)
    #在on_trade回调后处理
    def  by_trade(self,trade:TradeData)->None:

        pass
       
    def by_interval(self)->None:
        """"""#判断超时 撤单操作
        if not self.is_order_finished():
            self.cancel_all_order()

    #处理主动腿
    def take_active_leg(self,newTick:dict)->None:
        active_symbol:str = self.spread.active_leg.stdCode

        #Calculate spread order volume of new round trade
        spread_volume_left:float = self.traget - self.traded #价差剩余量 = 目标量-交易量

        if self.direction ==Direction.long:   #做多价差
            spread_order_volume :float = self.spread.ask_volume #价差单量=价差卖量
            spread_order_volume = min(spread_order_volume,spread_volume_left)
        else:                                #做空价差
            spread_order_volume : float = -self.spread.bid_volume #价差单量  = 价差买量
            spread_order_volume = max(spread_order_volume,spread_volume_left)
        
        #calculate active leg order volume
        leg_order_volume:float = self.spread.calculate_leg_volume(
            active_symbol,
            spread_order_volume
        )

        #check active leg volume left
        active_volume_target:float = self.spread.calculate_leg_volume(
           active_symbol,
           self.traget  
        )
        active_volume_traded: float = self.leg_traded[active_symbol]
        active_volume_left: float = active_volume_target - active_volume_traded

        # Limit order volume to total volume left of the active leg
        #将订单交易量限制为活动支腿剩余的总交易量
        if active_volume_left >0:
            leg_order_volume:float = min(leg_order_volume,active_volume_left)
        else:
            leg_order_volume:float = max(leg_order_volume,active_volume_left)

            #Send active leg order
        self.send_leg_order(
            active_symbol,
            leg_order_volume
        )

    def hedge_passive_legs(self)->None:
        #对冲剩下的被动腿
        #计算价差量去对冲
        active_leg: LegData = self.spread.active_leg
        active_traded:float = self.leg_traded[active_leg.stdCode]
        active_traded:float = round_to(active_traded,self.spread.min_volume)

        hedge_volume:float = self.spread.calculate_spread_volume(
            active_leg.stdCode,
            active_traded
        )

        #计算被动腿目标量和需要对冲量
        for leg in self.spread.passive_legs:
            passive_traded:float = self.leg_traded[leg.stdCode]
            passive_traded:float = round_to(passive_traded,self.spread.min_volume)

            passive_target:float = self.spread.calculate_leg_volume(
                leg.stdCode,
                hedge_volume
            )

            leg_order_volume:float = passive_target - passive_traded
            if leg_order_volume:
                self.send_leg_order(leg.stdCode,leg_order_volume)
            
    def send_leg_order(self,stdCode:str,leg_volume:float,context:HftContext)->None:
        num:int = 1
        leg:LegData = self.spread.legs[stdCode]
        #leg_tick= context.stra_get_ticks(stdCode,num)
        leg_tick = self.__last_tick__
        leg_product :ProductInfo= self.stra_get_comminfo(stdCode)

        if leg_volume >0 :
            price:float= leg_tick['ask_price_0'] + leg_product.pricetick * self.payup
            #self.send_order(leg.stdCode,price,abs(leg_volume),Direction.LONG)
            ids = context.stra_buy(leg.stdCode,price,abs(leg_volume),"enterlong")
            self.write_log("%s发出%s委托,下单量：%d" %(stdCode,"多头",abs(leg_volume)))
            for localid in ids:
                self.__orders__[localid] = localid

        elif leg_volume <0:
            price = leg_tick["bid_price_0"] - leg_product.pricetick * self.payup
            #self.send_order(leg.stdCode,price,abs(leg_volume),Direction.SHORT)
            ids = context.stra_sell(leg.stdCode,price,abs(leg_volume),"entershort")
            self.write_log("%s发出%s委托,下单量：%d" %(stdCode,"多头",abs(leg_volume)))
            for localid in ids:
                self.__orders__[localid] = localid

#""""""""""""""""Template""""""""""""""
    def is_order_finished(self)->bool:
        """检查委托是否全部结束"""
        finished:bool = True

        for leg in self.spread.legs.values():
            vt_orderids:list = self.leg_orders[leg.stdCode] #采用下单函数返回后的List

            if vt_orderids:
                finished = False
                break
        return finished

    #计算已成交价差均价
    def calculate_traded_price(self)->None:
        """"""
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
        
    def calculate_traded_volume(self)->None:
        """
        计算已成交价差数量
        """
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

    def is_hedge_finished(self)->bool:
        """检查当前各条腿是否平衡"""
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
        
        self.on_trade(trade)

    def calculate_traded_volume(self)->None: ##############################需修改
        """
        计算已成交价差数量
        """
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

    #计算已成交价差均价
    def calculate_traded_price(self)->None:
        """"""
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

    def insert_tick_data(
        self,
        spread:SpreadData,
        leg:LegData,
        newTick:dict,
        stdCode:str,
        pricetick:float=0,
    ):
        """"""
        spread_ticks:List[dict] =[]
        #更新数据于每个腿中
        #for stdCode in spread.legs.keys():
        #    if newTick["code"] == stdCode:
        leg:LegData = self.legs.get(newTick["code"],None)
        print("on_tick",stdCode,"price:",newTick["price"],"A1price:",newTick["ask_price_0"],"A1qty:",newTick["ask_qty_0"],"B1price:",newTick["bid_price_0"],"B1qty:",newTick["bid_qty_0"],"tick:",newTick["action_time"])
        leg.update_tick(newTick)