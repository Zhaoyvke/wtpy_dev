from wtpy import WtBtEngine, EngineType
from strategies.HftStraOrderImbalanceDemo import HftStraOrderImbalance
#回测入口
def read_params_from_csv(filename) -> dict:
    params = {
        "beta_0":0.0,
        "beta_r":0.0,
        "beta_oi":[],
        "beta_rou":[]
    }
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    for row in range(1, len(lines)):
        curLine = lines[row]
        ay = curLine.split(",")
        if row == 1:
            params["beta_0"] = float(ay[1])
        elif row == 14:
            params["beta_r"] = float(ay[1])
        elif row > 1 and row <=7:
            params["beta_oi"].append(float(ay[1]))
        elif row > 7 and row <=13:
            params["beta_rou"].append(float(ay[1]))

    return params


if __name__ == "__main__":
    # 创建一个运行环境，并加入策略
    engine = WtBtEngine(EngineType.ET_HFT)
    engine.init('.\\Common\\', "configbt.json")
    engine.configBacktest(202101040900,202101181500)
    engine.configBTStorage(mode="csv", path="./storage/")
    engine.commitBTConfig()

    active_sections = [
        {
            "start": 931,       #9点31开始交易
            "end": 1457         #14点57分了结头寸离场
        }
    ]

    stop_params = {
        "active":True,          # 是否启用止盈止损
        "stop_ticks": -25,      # 止损跳数，如果浮亏达到该跳数，则直接止损
        "track_threshold": 15,  # 追踪止盈阈值跳数，超过该阈值则触发追踪止盈
        "fallback_boundary": 2, # 追踪止盈回撤边界跳数，即浮盈跳数回撤到该边界值以下，立即止盈
        "calc_price":0
    }

    params = read_params_from_csv('IF_10ticks_20201201_20201231.csv')
    straInfo = HftStraOrderImbalance(name='hft_IF',
                                     code="CFFEX.IF.HOT",
                                     count=6,
                                     lots=1,
                                     threshold=0.3,
                                     expsecs=5,
                                     offset=0,
                                     freq=0,
                                     active_secs=active_sections,
                                     stoppl=stop_params,
                                     **params)

    engine.set_hft_strategy(straInfo)
    engine.run_backtest()
    kw = input('press any key to exit\n')
    engine.release_backtest()