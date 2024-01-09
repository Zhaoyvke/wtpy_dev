"""
Event type string used in the trading platform.
"""



EVENT_TICK = "eTick."
EVENT_TRADE = "eTrade."
EVENT_ORDER = "eOrder."
EVENT_POSITION = "ePosition."
EVENT_ACCOUNT = "eAccount."
EVENT_QUOTE = "eQuote."
EVENT_CONTRACT = "eContract."
EVENT_LOG = "eLog"


@lru_cache(maxsize=999)
def load_tick_data(
    spread: SpreadData,
    start: datetime,
    end: datetime,
    pricetick: float = 0
):
    """"""
    # hxxjava debug spread_trading
    # 目前没有考虑反向合约的情况，以后解决
    spread_ticks: List[TickData] = []

    try:
        # 防止因为用户没有米筐tick数据权限而发生异常

        # Load tick data of each spread leg
        dt_legs: Dict[str, Dict] = {}   # datetime string : Dict[vt_symbol:tick]
        format_str = "%Y%m%d%H%M%S.%f"
        for vt_symbol in spread.legs.keys():
            symbol, exchange = extract_vt_symbol(vt_symbol)

            # hxxjava debug spread_trading
            tick_data = query_tick_from_rq(symbol=symbol, exchange=exchange,start=start,end=end)

            if tick_data:
                print(f"load from rqdatac {symbol}.{exchange} tick_data, len of = {len(tick_data)}")

            # save all the spread's legs tick into a dictionary by tick's datetime
            for tick in tick_data:
                dt_str = tick.datetime.strftime(format_str)
                if dt_str in dt_legs:
                    dt_legs[dt_str].update({vt_symbol:tick})
                else:
                    dt_legs[dt_str] = {vt_symbol:tick}

        # Calculate spread bar data
        # snapshot of all legs's ticks  
        snapshot:Dict[str,TickData] = {}
        spread_leg_count = len(spread.legs)

        for dt_str in sorted(dt_legs.keys()): 
            dt = datetime.strptime(dt_str,format_str).astimezone(LOCAL_TZ)
            # get each datetime  
            spread_price = 0
            spread_value = 0

            # get all legs's ticks dictionary at the datetime
            leg_ticks = dt_legs.get(dt_str)
            for vt_symbol,tick in leg_ticks.items():
                # save each tick into the snapshot
                snapshot.update({vt_symbol:tick})

            if len(snapshot) < spread_leg_count:
                # if not all legs tick saved in the snapshot
                continue

            # out_str = f"{dt_str} "
            # format_str1 = "%Y-%m-%d %H:%M:%S.%f "
            for vt_symbol,tick in snapshot.items():
                price_multiplier = spread.price_multipliers[vt_symbol]
                spread_price += price_multiplier * tick.last_price
                spread_value += abs(price_multiplier) * tick.last_price
                # out_str += f"[{vt_symbol} {tick.datetime.strftime(format_str1)} {tick.last_price}],"
            # print(out_str)

            if pricetick:
                spread_price = round_to(spread_price, pricetick)

            spread_tick = TickData(                
                symbol=spread.name,
                exchange=exchange.LOCAL,
                datetime=dt,
                open_price=spread_price,
                high_price=spread_price,
                low_price=spread_price,
                last_price=spread_price,
                gateway_name="SPREAD")

            spread_tick.value = spread_value
            spread_ticks.append(spread_tick)

        if spread_ticks:
            print(f"load {symbol}.{exchange}' ticks from rqdatac, len of = {len(tick_data)}")

    finally:
        if not spread_ticks:
            # 读取数据库中已经录制过的该价差的tick数据
            spread_ticks = database_manager.load_tick_data(spread.name, Exchange.LOCAL, start, end)

        return spread_ticks