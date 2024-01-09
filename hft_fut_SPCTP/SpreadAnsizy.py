from wtpy import WtEngine, EngineType
from strategies.HftStraDemo import HftStraDemo

import pandas as pd
import pymysql
import matplotlib.pyplot as plt

import SpreadFunction as MF

# if __name__ == "__main__":
#     #创建一个运行环境，并加入策略
#     engine = WtEngine(EngineType.ET_HFT)

#     #初始化执行环境，传入
#     engine.init(folder = '../common/', cfgfile = "config.yaml")

#     #设置数据存储目录
#     # engine.configStorage(module="", path="D:\\WTP_Data\\")

#     #注册CTA策略工厂，即C++的CTA策略工厂模块所在的目录
#     # engine.regCtaStraFactories(factFolder = ".\\cta\\")
    
#     #添加外部CTA策略，即C++版本的CTA策略
#     '''
#     engine.addExternalHftStrategy(id = "cppxpa_rb", params = {
#         "name":"WtCtaStraFact.DualThrust",  #工厂名.策略名
#         "params":{  #这是策略所需要的参数
#             "code":"SHFE.rb.HOT",
#             "period":"m3",
#             "count": 50,
#             "days": 30,
#             "k1": 0.2,
#             "k2": 0.2
#         }
#     })
#     '''

#     engine.commitConfig()
    
#     #添加Python版本的策略
#     straInfo = HftStraDemo(name='pyhft_IF', code="CFFEX.IF.HOT", expsecs=15, offset=0, freq=30)
#     engine.add_hft_strategy(straInfo, trader="simnow")
    
#     #开始运行
#     engine.run()

#     kw = input('press any key to exit\n')


#初始化数据库连接
mysqlconn = pymysql.connect(host='localhost',user='',passwd='',database='',charset='utf8')

#读取合约信息数据
df_instrumentInfo = pd.read_sql('select * from futureinstrumentinfo table',mysqlconn)#全表选择

#只保留所需列
df_instrumentInfo = df_instrumentInfo.loc[:,['instrumentID','instrumentClass','listDate','expiryDate']]

#将date转换为datetime
df_instrumentInfo['listDate'] = pd.to_datetime(df_instrumentInfo['listDate'])
df_instrumentInfo['expiryDate'] = pd.to_datetime(df_instrumentInfo['expiryDate'])

#读取行情数据 (在此可更改回测日期)
df_marketData = pd.read_sql("select * from futuredaymarketdata table where updateDate > '2022-09-01' order by updateDate desc",mysqlconn)

#只保留所需列
df_marketData = df_marketData.loc[:,['instrument','updateDate','todayClosePrice']]

df_marketData['updateDate'] = pd.to_datetime(df_marketData['updateDate'])
 
#剔除上市天数较短或即将到期的合约
#计算上市天数
df_instrumentInfo['listDays'] = df_instrumentInfo['listDate'].apply(MF.GetListDays)

#计算剩余到期天数
df_instrumentInfo['expiryDays'] = df_instrumentInfo['expiryDays'].apply(MF.GetExpiryDays)

#剔除天数不符合条件的合约()
minListDays = 40
minExpiryDays = 80

df_instrumentInfo = df_instrumentInfo.drop(df_instrumentInfo[df_instrumentInfo['listDays']<minListDays].index)
df_instrumentInfo = df_instrumentInfo.drop(df_instrumentInfo[df_instrumentInfo['expiryDays']<minExpiryDays].index)

#根据合约品种分组，将同品种合约代码加入一个队列中并排序
#合约品种-list合约代码  ，对list添加键值对时需用setdefault
dict_instrumentClassToInstrumentID = {}

#遍历每行
for index,row in df_instrumentInfo.iterrows():

    dict_instrumentClassToInstrumentID.setdefault(row['instrumentClass'],[].append(row['instrumentID']))#ket-list(value) 所以添加键值对时候需要用setdefault
    dict_instrumentClassToInstrumentID[row['instrumentClass']].sort()

list = dict_instrumentClassToInstrumentID['rb']

for value in list:
    print(value)

#将总数据df根据合约代码拆分成子df
#合约代码-行情df
dict_instrumentMarkerData = {}
for value in dict_instrumentClassToInstrumentID.values():
    for instrumentID in value:
        #获取当前合约对应的df
        df_instrumentMarketData = df_marketData[df_marketData['instrumentID'] == instrumentID]
        dict_instrumentMarkerData[instrumentID] = df_instrumentMarketData

print(dict_instrumentMarkerData['rb2301'] )

#计算同品种合约价差 - 最新价还是买卖价 
#bid1 3500 ask 3550  last3540 (tick /s数据时候：用对手价：      或本方价计算 )
#买入价差 B ask1 - A bid1      卖出价差 A bid1 - B ask1
#matplotlib绘图

#计算价差
#(instrumentIDA - instrumentIDB) - 价差df   key:(instrumentIDA - instrumentIDB) value :价差df
dict_priceSpread = {}

#遍历合约品种
for value in dict_instrumentClassToInstrumentID.values(): #合约品种- 代码值队列
    listSize = len(value)

    #lsit  = ['rb2301','rb2302','rb2305']
    #两辆匹配 不重复
    for i in range(0,listSize - 1):
        for j in range(i+1,listSize):

            instrumentIDA = value[i]
            instrumentIDB = value[j]

            df_A = dict_instrumentMarkerData[instrumentIDA]
            df_B = dict_instrumentMarkerData[instrumentIDB]

            df_priceSpread = pd.merge(df_A,df_B,how= 'inner',left_on=['updateDate'],right_on=['updateDate'],suffixes=['A','B'])

            #计算价差
            #bid1 3500 ask 3550  last3540 (tick /s数据时候：用对手价：      或本方价计算 )
            #买入价差 B ask1 - A bid1      卖出价差 A bid1 - B ask1
            df_priceSpread['priceSpread'] = df_priceSpread['todayClosePriceA'] - df_priceSpread['todayClosePriceB']

            #只保留所需列
            df_priceSpread = df_priceSpread.loc[:,['instrumentIDA','instrumentIDB','updateDate','priceSpread']]

            #Key的格式为(instrumentIDA - instrumentIDB)
            dict_priceSpread[instrumentIDA + '-' + instrumentIDB] = df_priceSpread
        
print(dict_priceSpread['rb2301 - rb2302']) 

df_test = dict_priceSpread['MA302-MA306']

#plt将index 作为x轴
#绘图
df_test.set_index('updateDate',inplace = True)
plt.plot(df_test['priceSpread'])
plt.show()

#回归分析
#最小二乘法拟合斜率
#计算回归区间
#计算价差振幅（盈率区间）

#计算斜率、回归区间及价差振幅
df_result = pd.DataFrame()

#遍历所有价差df
for value in dict_priceSpread.values():
 
    #计算斜率
    value['slope'] = MF.CalSlope(value)

    #计算回归区间
    value['lowerLimit'] = value.groupby(['instrumentIDA','instrumentIDB'])['priceSpread'].transform('min')
    value['upperLimit'] = value.groupby(['instrumentIDA','instrumentIDB'])['priceSpread'].transform('max')

    #计算振幅
    value['deviation'] = value['upperLimit'] - value['lowerLimit']

    #根据合约代码去重
    value.drop_duplicates(subset = ['instrumentIDA','instrumentIDB'],keep = 'first',inplace = True)

    #只保留制定列
    value = value.loc[:['instrumentIDA','instrumentIDB','slope','lowerLimit','upperLimit','deviation']]

    df_result = df_result.append(value,ignore_index = True,sort = False)

    #剔除斜率绝对值过大的合约组
    maxAbsSlope = 0.5
    df_result = df_result.drop(df_result[abs(df_result['slope'])>maxAbsSlope].index)

    #剔除振幅过小的合约组
    minDeviation = 20
    df_result = df_result.drop(df_result[abs(df_result['deviation'])<minDeviation].index)

    df_result.to_csv(r'.\result.csv',index=False,encoding = 'utf_8_sig')

print(df_result)

 #判断 长期走势回归/不回归
 #参考分析报告调整交易参数
 #debug
 #价差触发不频繁 -- 太频繁
 #触发后未成交--滑点过大 - 交易延迟
 #成交后价差未回归 - 模型有效性\数据周期 