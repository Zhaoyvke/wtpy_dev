from wtpy import WtEngine, EngineType
from strategies.HftStraDemo import HftStraDemo

if __name__ == "__main__":
    #创建一个运行环境，并加入策略
    engine = WtEngine(EngineType.ET_HFT)

    #初始化执行环境，传入
    engine.init(folder = '../common/', cfgfile = "config.yaml")

    #设置数据存储目录
    # engine.configStorage(module="", path="D:\\WTP_Data\\")

    engine.commitConfig()
    
    #添加Python版本的策略
    straInfo = HftStraDemo(name='pyhft_y', code="DCE.i.2405", expsecs=15, offset=0, freq=30)
    engine.add_hft_strategy(straInfo, trader="simnow")
    
    #开始运行
    engine.run()

    kw = input('press any key to exit\n')