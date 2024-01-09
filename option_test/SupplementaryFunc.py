# -*- coding: utf-8 -*-
from OptionInfo import *

import pandas as pd
import polars as pl
import numpy as np
import os
import calendar
from datetime import datetime
import re


def add_ETF(option):
    tmp = option.split('.')
    return tmp[0] + '.ETFO.' + tmp[-1]

def sub_ETF(option):
    tmp = option.split('.')
    return tmp[0] + '.' + tmp[-1]


def timeToMaturity(today, month, trading_dates):
    # thisYear = int(date[:4])
    trading_dates = np.array(sorted(trading_dates))

    thisYear = int(today[:4])
    thisMonth = int(today[4:6])
    targetMonth = int(month)

    if thisMonth > targetMonth:
        targetYear = thisYear + 1
    else:
        targetYear = thisYear

    if calendar.monthcalendar(targetYear, targetMonth)[0][2] == 0:
        day = calendar.monthcalendar(targetYear, targetMonth)[4][2]
    else:
        day = calendar.monthcalendar(targetYear, targetMonth)[3][2]

    date = datetime(targetYear, targetMonth, day).strftime('%Y%m%d')

    date_int = int(date)
    trading_dates_int = np.array([int(x) for x in trading_dates])
    if date not in trading_dates:
        date_int = np.min(np.where(trading_dates_int > date_int, trading_dates_int, 30000000))

    today_idx = np.where(trading_dates_int == int(today))[0][0]
    maturity_idx = np.where(trading_dates_int == date_int)[0][0]

    ttm = maturity_idx - today_idx

    if ttm % 1 != 0:
        raise ValueError('TTM not integer')

    return ttm

def get_previous_trading_date(date, trading_dates):
    date = str(date)
    # trading_dates_int = np.array([int(x) for x in sorted(trading_dates)])
    today_idx = np.where(np.array(trading_dates) == date)[0][0]

    prev_date = trading_dates[today_idx-1]

    return prev_date

def get_next_trading_date(date, trading_dates):
    date = str(date)
    today_idx = np.where(np.array(trading_dates) == date)[0][0]

    next_date = trading_dates[today_idx+1]

    return next_date


def month_in_name(name):
    name = name.split('月')[0]
    try:
        if '购' in name:
            month = int(name.split('购')[-1])
        else:
            month = int(name.split('沽')[-1])
    except:
        # import pdb
        # pdb.set_trace()
        # name = name.encode('utf-8').decode('GB2312')
        if '购' in name:
            month = int(name.split('购')[-1])
        else:
            month = int(name.split('沽')[-1])
        # month = int(re.sub(r'\D+', ' ' , name).split(' ')[1])
    return month


def singleDeposit(df):
    s0 = df['underlyingPrice']
    k = df['strike']
    optionType = df['type']
    c = df['close']
    multiplier = 10000

    if optionType in ['c', 'C', 'Call', 'call']:  # call
        return (c + max(0.12 * s0 - max(k - s0, 0), 0.07 * s0)) * multiplier
    elif optionType in ['p', 'P', 'Put', 'put']:
        return min(c + max(0.12 * s0 - max(s0 - k, 0), 0.07 * k), k) * multiplier
    else:
        raise TypeError('Illegal ValueError')


def data_to_option(data):
    options = []
    # data = pl.DataFrame(data)
    # data = OptionInfo(data).addRFIV(data)
    # data = data.to_pandas()
    for i in range(data.shape[0]):
        this_option = Option(data['underlyingPrice'][i], data['strike'][i], 0, data['timeToMaturity'][i], #data['IV'][i],
                             cpflag=data['type'][i], market=data['close'][i])

        options.append(this_option)

    return options


def get_trading_dates(path='./XSHG_Trading_Dates.txt'):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            res.append(line[:8])

    return res

def fill_time(date, time):
    time = str(time)
    if len(time) == 3:
        time = '0' + time + '00'
    elif len(time) == 4:
        time = time + '00'
    else:
        raise ValueError(f'{time} dont have length 3 or 4')

    return str(date) + time


def find_n_closest_value(vs, target, n):
    vs = np.array(vs)
    vs_abs = np.abs(vs - target)
    target_abs = sorted(set(vs_abs))[n - 1]

    idx = np.where(vs_abs == target_abs)[0][0]

    return vs[idx]








