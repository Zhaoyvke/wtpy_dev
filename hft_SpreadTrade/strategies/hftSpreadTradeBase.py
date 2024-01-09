from wtpy import BaseHftStrategy
from wtpy import HftContext
from SpreadBase import LegData,SpreadData
from SpreadConstant import TradeData
from SpreadTemplate import SpreadAlgoTemplate 
from datetime import datetime
from SpreadEngine import SpreadEngine,SpreadDataEngine
from SpreadAlgo import SpreadAlgo
def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)

class hftSpreadTradeBase(SpreadAlgo):
    def __init__(self, 
                 name:str, 
                 activeleg:str,
                 spread: SpreadData, 
                 leg:dict,
                 expsecs:int, 
                 offset:int, 
                 freq:int=30):
        BaseHftStrategy.__init__(self, name),
        SpreadAlgo.__init__(self,name)

        '''交易参数'''
        self.__activeleg__ = activeleg  #交易合约
        self.__expsecs__ = expsecs      #订单超时秒数
        self.__offset__ = offset        #指令价格偏移
        self.__freq__ = freq            #交易频率控制，指定时间内限制信号数，单位秒
        self.__legs__ = leg              
        self.__spread__ = spread
        '''内部数据'''
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪

        self.algo:SpreadAlgo
        '''初始化交易数据'''
        buy_price = 1.0     #买入开仓阈值
        sell_price = 5.0    #卖出平仓阈值
        cover_price = 5.0   #买入平仓阈值
        short_price = 1.0   #卖出开仓阈值
        max_pos = 1.0       #主动腿委托数量
        payup = 10          #超价的数值
        interval = 5        #时间间隔，即每隔一段时间，会发出委托
        start_time = "9:00:00"
        end_time = "15:00:00"

    def on_init(self, context:HftContext):
        '''
        策略初始化，启动的时候调用\n
        用于加载自定义数据\n
        @context    策略运行上下文
        '''
        #先订阅legs数据
        for leg in self.__legs__.values():
            context.stra_sub_ticks(leg)
            print(leg,"on_init")
            print(self.__spread__.name)
        self.__ctx__ = context

    def on_tick(self, context:HftContext,stdCode:str, newTick:dict,algo:SpreadAlgo):
        print("on_tick",stdCode,"price:",newTick["price"],"A1price:",newTick["ask_price_0"],"A1qty:",newTick["ask_qty_0"],"B1price:",newTick["bid_price_0"],"B1qty:",newTick["bid_qty_0"],"tick:",newTick["action_time"])

        #self.leg:LegData =  self.leg.update_tick(stdCode,newTick) 
        self.insert_tick_data()
        #self.spread:SpreadData = algo.get_spread_tick()
        

        #如果有未完成订单，则进入订单管理逻辑
        if len(self.__orders__.keys()) != 0:
            self.check_orders()
            return

        if not self.__channel_ready__:
            return

        self.__last_tick__ = newTick

        #如果已经入场，则做频率检查
        if self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            span = now - self.__last_entry_time__
            if span.total_seconds() <= 30:
                return

        #信号标志
        signal = 0
        #最新价作为基准价格
        price = newTick["price"]
        #计算理论价格
        pxInThry = (newTick["bid_price_0"]*newTick["ask_qty_0"] + newTick["ask_price_0"]*newTick["bid_qty_0"]) / (newTick["ask_qty_0"] + newTick["bid_qty_0"])

        context.stra_log_text("理论价格%f，最新价：%f" % (pxInThry, price))

        if pxInThry > price:    #理论价格大于最新价，正向信号
            signal = 1
            context.stra_log_text("出现正向信号")
        elif pxInThry < price:  #理论价格小于最新价，反向信号
            signal = -1
            context.stra_log_text("出现反向信号")

    def on_bar(self, context:HftContext, stdCode:str, period:str, newBar:dict):
        return

    def on_channel_ready(self, context:HftContext):
        print("on_channel_ready")
        undone = context.stra_get_undone(self.__activeleg__)
        if undone != 0 and len(self.__orders__.keys()) == 0:
            context.stra_log_text("%s存在不在管理中的未完成单%f手，全部撤销" % (self.__activeleg__, undone))
            isBuy = (undone > 0)
            ids = context.stra_cancel_all(self.__activeleg__, isBuy)
            for localid in ids:
                self.__orders__[localid] = localid
            self.__cancel_cnt__ += len(ids)
            context.stra_log_text("cancelcnt -> %d" % (self.__cancel_cnt__))
        self.__channel_ready__ = True

    def on_channel_lost(self, context:HftContext):
        context.stra_log_text("交易通道连接丢失")
        self.__channel_ready__ = False

    def on_entrust(self, context:HftContext, localid:int, stdCode:str, bSucc:bool, msg:str, userTag:str):
        if bSucc:
            context.stra_log_text("%s下单成功，本地单号：%d" % (stdCode, localid))
        else:
            context.stra_log_text("%s下单失败，本地单号：%d，错误信息：%s" % (stdCode, localid, msg))

    def on_order(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        if localid not in self.__orders__:
            return

        if isCanceled or leftQty == 0:
            self.__orders__.pop(localid)
            if self.__cancel_cnt__ > 0:
                self.__cancel_cnt__ -= 1
                self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))
        return

    def on_trade(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        return

    def check_orders(self):
        #如果未完成订单不为空
        if len(self.__orders__.keys()) > 0 and self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            span = now - self.__last_entry_time__
            if span.total_seconds() > self.__expsecs__: #如果订单超时，则需要撤单
                for localid in self.__orders__:
                    self.__ctx__.stra_cancel(localid)
                    self.__cancel_cnt__ += 1
                    self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))