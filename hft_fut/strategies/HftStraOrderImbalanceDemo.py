from wtpy import BaseHftStrategy
from wtpy import HftContext

from datetime import datetime



class HftStraOrderImbalance(BaseHftStrategy):
    
    def __init__(self, name:str, code:str, count:int, lots:int, beta_0:float, beta_r:float, threshold:float, 
            beta_oi:list, beta_rou:list, expsecs:int, offset:int, freq:int, active_secs:list, stoppl:dict, reserve:int=0):
        BaseHftStrategy.__init__(self, name)

        '''交易参数'''
        self.__code__ = code            #交易合约
        self.__expsecs__ = expsecs      #订单超时秒数，用于控制超时撤单
        self.__freq__ = freq            #交易频率控制，指定时间内限制信号数，单位秒

        self.__lots__ = lots            #单次交易手数

        self.count = count              #回溯tick条数
        self.beta_0 = beta_0            #常量系数+残差
        self.beta_r = beta_r            #中间价回归因子系数
        self.threshold = threshold      #中间价变动阈值
        self.beta_oi = beta_oi          #成交量不平衡因子系数序列
        self.beta_rou = beta_rou        #委比因子系数序列
        self.active_secs = active_secs  #交易时间区间
        self.stoppl = stoppl            #止盈止损配置

        '''内部数据'''
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪

        self.__comm_info__ = None
        self.__to_clear__ = False

        self._last_entry_price = 0.0
        self._max_dyn_prof = 0.0
        self._max_dyn_loss = 0.0

        self._last_atp__ = 0.0

    def is_active(self, curMin:int) -> bool:
        for sec in self.active_secs:
            if sec["start"] <= curMin and curMin <= sec["end"]:
                return True
        return False

    def on_init(self, context:HftContext):
        '''
        策略初始化，启动的时候调用\n
        用于加载自定义数据\n
        @context    策略运行上下文
        '''
        self.__comm_info__ = context.stra_get_comminfo(self.__code__)
        #先订阅实时数据
        context.stra_sub_ticks(self.__code__)

        self.__ctx__ = context

    def check_orders(self, ctx:HftContext):
        #如果未完成订单不为空
        ord_cnt = len(self.__orders__.keys())
        if ord_cnt > 0 and self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(ctx.stra_get_date(), ctx.stra_get_time(), ctx.stra_get_secs())
            span = now - self.__last_entry_time__
            total_secs = span.total_seconds()
            if total_secs >= self.__expsecs__: #如果订单超时，则需要撤单
                ctx.stra_log_text("%d条订单超时撤单" % (ord_cnt))
                for localid in self.__orders__:
                    ctx.stra_cancel(localid)
                    self.__cancel_cnt__ += 1
                    ctx.stra_log_text("在途撤单数 -> %d" % (self.__cancel_cnt__))

    def get_price(self, newTick, pricemode=0):
        if pricemode == 0:
            return newTick["price"]
        elif pricemode == 1:
            return newTick["askprice"][0] if len(newTick["askprice"])>0 else newTick["price"]
        elif pricemode == -1:
            return newTick["bidprice"][0] if len(newTick["bidprice"])>0 else newTick["price"]


    def on_tick(self, context:HftContext, stdCode:str, newTick:dict):
        if self.__code__ != stdCode:
            return

        #如果有未完成订单，则进入订单管理逻辑
        if len(self.__orders__.keys()) != 0:
            self.check_orders(context)
            return

        if not self.__channel_ready__:
            return

        curMin = context.stra_get_time()
        curPos = context.stra_get_position(stdCode)

        # 不在交易时间，则检查是否有持仓
	    # 如果有持仓，则需要清理
        if not self.is_active(curMin):
            self._last_atp__ = 0.0
            if curPos == 0:
                return
            self.__to_clear__ = True
        else:
            self.__to_clear__ = False

        # 如果需要清理持仓，且不在撤单过程中
        if self.__to_clear__ :
            if self.__cancel_cnt__ == 0:
                if curPos > 0:
                    targetPx = self.get_price(newTick, -1)
                    ids = context.stra_sell(self.__code__, targetPx, abs(curPos), "deadline")

                    #将订单号加入到管理中
                    for localid in ids:
                        self.__orders__[localid] = localid
                elif curPos < 0:
                    targetPx = self.get_price(newTick, 1)
                    ids = context.stra_buy(self.__code__, targetPx, abs(curPos), "deadline")

                    #将订单号加入到管理中
                    for localid in ids:
                        self.__orders__[localid] = localid
            
            return

        # 止盈止损逻辑
        if curPos != 0 and self.stoppl["active"]:
            isLong = (curPos > 0)
            price = 0
            if self.stoppl["calc_price"] == 0:
                price = self.get_price(newTick, -1) if isLong else self.get_price(newTick, 1)
            else:
                price = newTick["price"]
            diffTicks = (price - self._last_entry_price)*(1 if isLong else -1) / self.__comm_info__.pricetick
            if diffTicks > 0:
                self._max_dyn_prof = max(self._max_dyn_prof, diffTicks)
            else:
                self._max_dyn_loss = min(self._max_dyn_loss, diffTicks)

            bNeedExit = False
            usertag = ''
            stop_ticks = self.stoppl["stop_ticks"]
            track_threshold = self.stoppl["track_threshold"]
            fallback_boundary = self.stoppl["fallback_boundary"]
            if diffTicks <= stop_ticks:
                context.stra_log_text("浮亏%.0f超过%d跳，止损离场" % (diffTicks, stop_ticks))
                bNeedExit = True
                usertag = "stoploss"
            elif self._max_dyn_prof >= track_threshold and diffTicks <= fallback_boundary:
                context.stra_log_text("浮赢回撤%.0f->%.0f[阈值%.0f->%.0f]，止盈离场" % (self._max_dyn_prof, diffTicks, track_threshold, fallback_boundary))
                bNeedExit = True
                usertag = "stopprof"
            
            if bNeedExit:
                targetprice = self.get_price(newTick, -1) if isLong else self.get_price(newTick, 1)
                ids = context.stra_sell(self.__code__, targetprice, abs(curPos), usertag) if isLong else context.stra_buy(self.__code__, price, abs(curPos), usertag)
                for localid in ids:
                    self.__orders__[localid] = localid

                # 出场逻辑执行以后结束逻辑
                return

        now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
        
        # 成交量为0且上一个成交均价为0，则需要退出
        if newTick["volumn"] == 0 and self._last_atp__ == 0.0:
            return

        #如果已经入场，且有频率限制，则做频率检查
        if self.__last_entry_time__ is not None and self.__freq__ != 0:
            #当前时间，一定要从api获取，不然回测会有问题
            span = now - self.__last_entry_time__
            if span.total_seconds() <= self.__freq__:
                return

        hisTicks = context.stra_get_ticks(self.__code__, self.count + 1)
        if hisTicks.size != self.count+1:
            return

        if (len(newTick["askprice"]) == 0) or (len(newTick["bidprice"]) == 0):
            return

        spread = newTick["askprice"][0] - newTick["bidprice"][0]
        total_OIR = 0.0
        total_rou = 0.0
        for i in range(1, self.count + 1):
            prevTick = hisTicks.get_tick(i-1)
            curTick = hisTicks.get_tick(i)

            lastBidPx = self.get_price(prevTick, -1)
            lastAskPx = self.get_price(prevTick, 1)

            lastBidQty = prevTick["bidqty"][0] if len(prevTick["bidqty"]) > 0 else 0
            lastAskQty = prevTick["askqty"][0] if len(prevTick["askqty"]) > 0 else 0

            curBidPx = self.get_price(curTick, -1)
            curAskPx = self.get_price(curTick, 1)

            curBidQty = curTick["bidqty"][0] if len(curTick["bidqty"]) > 0 else 0
            curAskQty = curTick["askqty"][0] if len(curTick["askqty"]) > 0 else 0

            delta_vb = 0.0
            delta_va = 0.0
            if curBidPx < lastBidPx:
                delta_vb = 0.0
            elif curBidPx == lastBidPx:
                delta_vb = curBidQty - lastBidQty
            else:
                delta_vb = curBidQty

            if curAskPx < lastAskPx:
                delta_va = curAskQty
            elif curAskPx == lastAskPx:
                delta_va = curAskQty - lastAskQty
            else:
                delta_va = 0.0
            voi = delta_vb - delta_va
            total_OIR += self.beta_oi[i-1]*voi/spread

            #计算委比
            rou = (curBidQty - curAskQty)/(curBidQty + curAskQty)
            total_rou += self.beta_rou[i-1]*rou/spread


        prevTick = hisTicks.get_tick(-2)
        
        # t-1时刻的中间价
        prevMP = (self.get_price(prevTick, -1) + self.get_price(prevTick, 1))/2
        # 最新的中间价
        curMP = (newTick["askprice"][0] + newTick["bidprice"][0])/2
        # 两个快照之间的成交均价
        if newTick["volumn"] != 0:
            avgTrdPx = newTick["turn_over"]/newTick["volumn"]/self.__comm_info__.volscale
        elif self._last_atp__!= 0:
            avgTrdPx = self._last_atp__
        else:
            avgTrdPx = curMP

        self._last_atp__ = avgTrdPx

        # 计算中间价回归因子
        curR = avgTrdPx - (prevMP + curMP) / 2

        # 计算预期中间价变化量
        efpc = self.beta_0 + total_OIR + total_rou + self.beta_r * curR / spread

        if efpc >= self.threshold:
            targetPos = self.__lots__
            diffPos = targetPos - curPos
            if diffPos != 0.0:
                targetPx = newTick["askprice"][0]
                ids = context.stra_buy(self.__code__, targetPx, abs(diffPos), "enterlong")

                #将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                self.__last_entry_time__ = now
                self._max_dyn_prof = 0
                self._max_dyn_loss = 0
        elif efpc <= -self.threshold:
            targetPos = -self.__lots__
            diffPos = targetPos - curPos
            if diffPos != 0:
                targetPx = newTick["bidprice"][0]
                ids = context.stra_sell(self.__code__, targetPx, abs(diffPos), "entershort")

                #将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                self.__last_entry_time__ = now
                self._max_dyn_prof = 0.0
                self._max_dyn_loss = 0.0


    def on_bar(self, context:HftContext, stdCode:str, period:str, newBar:dict):
        return

    def on_channel_ready(self, context:HftContext):
        undone = context.stra_get_undone(self.__code__)
        if undone != 0 and len(self.__orders__.keys()) == 0:
            context.stra_log_text("%s存在不在管理中的未完成单%f手，全部撤销" % (self.__code__, undone))
            isBuy = (undone > 0)
            ids = context.stra_cancel_all(self.__code__, isBuy)
            for localid in ids:
                self.__orders__[localid] = localid
            self.__cancel_cnt__ += len(ids)
            context.stra_log_text("在途撤单数 -> %d" % (self.__cancel_cnt__))
        self.__channel_ready__ = True

    def on_channel_lost(self, context:HftContext):
        context.stra_log_text("交易通道连接丢失")
        self.__channel_ready__ = False

    def on_entrust(self, context:HftContext, localid:int, stdCode:str, bSucc:bool, msg:str, userTag:str):
        return

    def on_order(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        if localid not in self.__orders__:
            return

        if isCanceled or leftQty == 0:
            self.__orders__.pop(localid)
            if self.__cancel_cnt__ > 0:
                self.__cancel_cnt__ -= 1
                self.__ctx__.stra_log_text("在途撤单数 -> %d" % (self.__cancel_cnt__))
        return

    def on_trade(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        self._last_entry_price = price


def makeTime(date:int, time:int, secs:int):
        '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
        return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)