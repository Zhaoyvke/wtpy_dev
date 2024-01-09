import pandas as pd
import numpy as np
import datetime
from decimal import Decimal
from math import floor, ceil
#import matplotlib.pyplot as plt

# noqa

#获取当前日期
g_currentDate = datetime.datetime.now()
 
def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    rounded: float = float(int(round(value / target)) * target)
    return rounded

#将一个浮点数 value 向下取整到目标浮点数 target
def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(floor(value / target)) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(ceil(value / target)) * target)
    return result
 
#获取上市天数
def GetListDays(series_listDate):

    days = (g_currentDate - series_listDate).days

    return days

#获取距离到期天数
def GetExpiryDays(series_expiryDate):

    days = (series_expiryDate - g_currentDate).days

    return days

#计算斜率
def CalSlope(df):

    list_data = df['priceSpread'].tolist()

    #输出结果为[斜率,常数] y = a + bx
    slope = np.polyfit(df.index,list_data,1)

    return slope[0]
    


