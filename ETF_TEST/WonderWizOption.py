# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:18:09 2020

@author: kaizhuang
"""

import numpy as np
import pandas as pd
import urllib
from scipy.stats import norm # Gaussian
import datetime
# import trading_calendars
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import calendar

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import BFGS
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import pdb

import rqdatac
rqdatac.init('18616633529','wuzhi2020')



# Black-Scholes Framework
# scalar value version, should vectorize later
# S: Underlying price
# K: strike price
# r: risk free rate
# tau: time to maturity (measure in year)
# vol: volatility
# q: underlying dividend rate
# V: option price


def getTau(trading_date, month):
    today = datetime.datetime.strptime(trading_date,'%Y-%m-%d')
    maturity = fourthWednesday(2000+int(str(month)[:2]), int(str(month)[2:]))
    return daysBetween(today,maturity)/240

def d1(S, K, r, tau, vol, q=0):
    K = np.array(K, dtype=float)
    vol = np.array(vol, dtype=float)
    tau = np.array(tau, dtype=float)
    return (np.log(S/K)+(r-q+0.5*vol**2)*tau) / (vol*np.sqrt(tau))

def d2(S, K, r, tau, vol, q=0):
    K = np.array(K, dtype=float)
    vol = np.array(vol, dtype=float)
    tau = np.array(tau, dtype=float)
    return d1(S, K, r, tau, vol, q) - vol*np.sqrt(tau)

def blsPrice(S, K, r, tau, vol, q=0, cpflag='C'):
    if cpflag in {'C', 'c', 'Call', 'call'}:
        if tau == 0:
            return max(S-K,0)
        return S*np.exp(-q*tau)*norm.cdf(d1(S,K,r,tau,vol,q)) - K*np.exp(-r*tau)*norm.cdf(d2(S,K,r,tau,vol,q))
    elif cpflag in {'P', 'p', 'Put', 'put'}:
        if tau == 0:
            return max(K-S,0)
        return K*np.exp(-r*tau)*norm.cdf(-d2(S,K,r,tau,vol,q)) - S*np.exp(-q*tau)*norm.cdf(-d1(S,K,r,tau,vol,q))

def blsDelta(S, K, r, tau, vol, q=0, cpflag='C'):
    if cpflag in {'C', 'c', 'Call', 'call'}:
        return np.exp(-q*tau)*norm.cdf(d1(S,K,r,tau,vol,q))
    elif cpflag in {'P', 'p', 'Put', 'put'}:
        return np.exp(-q*tau)*(norm.cdf(d1(S,K,r,tau,vol,q))-1)

def blsGamma(S, K, r, tau, vol, q=0):
    return np.exp(-q*tau)*norm.pdf(d1(S,K,r,tau,vol,q)) / (S*vol*np.sqrt(tau))

def blsVega(S, K, r, tau, vol, q=0):
    return S*np.exp(-q*tau)*np.sqrt(tau)*norm.pdf(d1(S,K,r,tau,vol,q))

def blsVanna(S, K, r, tau, vol, q=0):
    # twice differential wrt vol and S
    return -np.exp(-q*tau)*norm.pdf(d1(S,K,r,tau,vol,q))*d2(S,K,r,tau,vol,q)/vol

def blsVolga(S, K, r, tau, vol, q=0):
    # twice differential wrt vol
    return S*np.exp(-q*tau)*norm.pdf(d1(S,K,r,tau,vol,q))*np.sqrt(tau)*d1(S,K,r,tau,vol,q)*d2(S,K,r,tau,vol,q)/vol

def blsRho1(S, K, r, tau, vol, q=0, cpflag='C'):
    if cpflag in {'C', 'c', 'Call', 'call'}:
        return K*tau*np.exp(-r*tau)*norm.cdf(d2(S,K,r,tau,vol,q))
    elif cpflag in {'P', 'p', 'Put', 'put'}:
        return -K*tau*np.exp(-r*tau)*norm.cdf(-d2(S,K,r,tau,vol,q))

def blsTheta(S, K, r, tau, vol, q=0, cpflag='C'):
    term1 = -0.5*np.exp(-q*tau)*vol*S*norm.pdf(d1(S,K,r,tau,vol,q))/np.sqrt(tau)
    if cpflag in {'C', 'c', 'Call', 'call'}:
        term2 = q*np.exp(-q*tau)*S*norm.cdf(d1(S,K,r,tau,vol,q))
        term3 = -r*np.exp(-r*tau)*K*norm.cdf(d2(S,K,r,tau,vol,q))
    elif cpflag in {'P', 'p', 'Put', 'put'}:
        term2 = -q*np.exp(-q*tau)*S*norm.cdf(-d1(S,K,r,tau,vol,q))
        term3 = r*np.exp(-r*tau)*K*norm.cdf(-d2(S,K,r,tau,vol,q))
    return term1+term2+term3

def blsTheta1(S, K, r, tau, vol, q=0, cpflag='C'):
    p0 = blsPrice(S, K, r, tau, vol, q, cpflag)
    p1 = blsPrice(S, K, r, tau-1/240, vol, q, cpflag)
    return p1-p0
    # if cpflag in {'C', 'c', 'Call', 'call'}:

    # elif cpflag in {'P', 'p', 'Put', 'put'}:

def blsTheta5(S, K, r, tau, vol, q=0, cpflag='C'):
    if tau > 5/240:
        p0 = blsPrice(S, K, r, tau, vol, q, cpflag)
        p5 = blsPrice(S, K, r, tau-5/240, vol, q, cpflag)
        return p5-p0
    else:
        print('Near Maturity')
        return -blsPrice(S, K, r, tau, vol, q, cpflag)

def blsImpvSingle(S, K, r, tau, V, q=0, cpflag='C', tol=1e-6):
    if blsPrice(S,K,r,tau,0.000001,q,cpflag=cpflag) > V:
        #print('Check settlement Price!!' + str(K) + cpflag + str(tau))
        return 0.000001
    # Newton's method
    iv = 1
    # count = 0
    while True:
        if abs(blsVega(S, K, r, tau, iv)) < 1e-12:
            break
        if abs(blsPrice(S, K, r, tau, iv, q, cpflag)-V) < 1e-10:
            break
        niv = iv - (blsPrice(S, K, r, tau, iv, q, cpflag)-V)/blsVega(S, K, r, tau, iv)
        # if count < 100:
            # print(abs(blsPrice(S, K, r, tau, iv, q, cpflag)-V),blsVega(S, K, r, tau, iv),niv,abs(iv-niv))
            # count = count + 1
        # else:
            # break
        if abs(iv-niv) < tol:
            break
        iv = niv
    return iv

def blsImpv(S, K, r, tau, V, q=0, cpflag='C', tol=1e-6):
    if blsPrice(S,K,r,tau,0.000001,q,cpflag=cpflag) > V:
        #print('Check settlement Price!!' + str(K) + cpflag + str(tau))
        return 0.000001
    # Newton's method
    iv = 1
    # count = 0
    while True:
        if abs(blsVega(S, K, r, tau, iv)) < 1e-12:
            break
        if abs(blsPrice(S, K, r, tau, iv, q, cpflag)-V) < 1e-10:
            break
        niv = iv - (blsPrice(S, K, r, tau, iv, q, cpflag)-V)/blsVega(S, K, r, tau, iv)
        # if count < 100:
            # print(abs(blsPrice(S, K, r, tau, iv, q, cpflag)-V),blsVega(S, K, r, tau, iv),niv,abs(iv-niv))
            # count = count + 1
        # else:
            # break
        if abs(iv-niv) < tol:
            break
        iv = niv
    return iv

def blsImpvB(S, K, r, tau, V, q=0, cpflag='C', upper=10.0, lower=0.00001, tol=1e-6):
    # binary search
    # first check option impv is in [lower, upper]
    assert blsPrice(S,K,r,tau,upper,q,cpflag) > V, 'Upper limit is too low!'
    assert blsPrice(S,K,r,tau,lower,q,cpflag) < V, 'Lower limit is too high!'
    # binary search
    while True:
        midvol = (upper+lower)/2.0
        midPrice = blsPrice(S,K,r,tau,midvol,q,cpflag)
        if upper-lower < tol:
            break
        if midPrice >= V:
            upper = midvol
        else:
            lower = midvol
    return (upper+lower)/2.0

def secondFriday(year, month):
    # for treasury bond futures
    # w = datetime.date(year, month, 1).weekday()
    # if w == 0:
    #     # Monday
    #     return datetime.datetime(year, month, 12)
    # elif w == 1:
    #     # Tuesday
    #     return datetime.datetime(year, month, 11)
    # elif w == 2:
    #     # Wednesday
    #     return datetime.datetime(year, month, 10)
    # elif w == 3:
    #     # Thursday
    #     return datetime.datetime(year, month, 9)
    # elif w == 4:
    #     # Friday
    #     return datetime.datetime(year, month, 8)
    # elif w == 5:
    #     # Saturday
    #     return datetime.datetime(year, month, 14)
    # elif w == 6:
    #     # Sunday
    #     return datetime.datetime(year, month, 13)
    #lxy
    Calendar = calendar.monthcalendar(year,month)
    if Calendar[0][4]>0:
        day = Calendar[1][4]
    else:
        day = Calendar[2][4]
    return datetime.datetime(year, month, day)

def thirdFriday(year, month):
    # for treasury bond futures
    # w = datetime.date(year, month, 1).weekday()
    # if w == 0:
    #     # Monday
    #     return datetime.datetime(year, month, 19)
    # elif w == 1:
    #     # Tuesday
    #     return datetime.datetime(year, month, 18)
    # elif w == 2:
    #     # Wednesday
    #     return datetime.datetime(year, month, 17)
    # elif w == 3:
    #     # Thursday
    #     return datetime.datetime(year, month, 16)
    # elif w == 4:
    #     # Friday
    #     return datetime.datetime(year, month, 15)
    # elif w == 5:
    #     # Saturday
    #     return datetime.datetime(year, month, 21)
    # elif w == 6:
    #     # Sunday
    #     return datetime.datetime(year, month, 20)
    #lxy
    Calendar = calendar.monthcalendar(year,month)
    if Calendar[0][4]>0:
        day = Calendar[2][4]
    else:
        day = Calendar[3][4]
    return datetime.datetime(year, month, day)

def fourthWednesday(year, month):
    # used for ETF option Expiration
    # w = datetime.date(year, month, 1).weekday()
    # if w == 0:
    #     return datetime.datetime(year, month, 24)
    # elif w == 1:
    #     return datetime.datetime(year, month, 23)
    # elif w == 2:
    #     return datetime.datetime(year, month, 22)
    # elif w == 3:
    #     return datetime.datetime(year, month, 28)
    # elif w == 4:
    #     return datetime.datetime(year, month, 27)
    # elif w == 5:
    #     return datetime.datetime(year, month, 26)
    # elif w == 6:
    #     return datetime.datetime(year, month, 25)
    #lxy
    Calendar = calendar.monthcalendar(year,month)
    if Calendar[0][2]>0:
        day = Calendar[3][2]
    else:
        day = Calendar[4][2]
    return datetime.datetime(year, month, day)


def daysBetween(day1, day2):
    # use datetime as input
    # 有考虑节假日的。
    # if type(day1) == datetime.date:
    #     day1 = datetime.datetime(day1.year, day1.month, day1.day)

    # if type(day2) == datetime.date:
    #     day2 = datetime.datetime(day2.year, day2.month, day2.day)
    # weeks = (day2-day1).days//7
    # daysRemain = (day2-day1).days-weeks*7
    # # calculate weekends
    # weekends = (day2-day1).days//7*2
    # for i in range(1,daysRemain+1):
    #     if day1.weekday() + i in [5,6]:
    #         weekends += 1
    # # calcuate holidays
    # # shh = trading_calendars.exchange_calendar_xshg.precomputed_shanghai_holidays
    # holidays = sum((shh>np.datetime64(day1)) * (shh<np.datetime64(day2)))
    # weekdays = (day2-day1).days - weekends - holidays
    # if datetime.datetime.now().hour >= 15:
    #     return weekdays+0.00001
    # else:
    #     return weekdays+0.00001

    trading_days = rqdatac.get_trading_dates(start_date=day1, end_date=day2)
    return len(trading_days)-1

def IFExpiration(year, month):
    # IF future expiration date
    third = datetime.date(year, month, 15)
    w = third.weekday()
    if w == 0:
        third = third.replace(day=19)
    elif w == 1:
        third = third.replace(day=18)
    elif w == 2:
        third = third.replace(day=17)
    elif w == 3:
        third = third.replace(day=16)
    elif w == 5:
        third = third.replace(day=21)
    elif w == 6:
        third = third.replace(day=20)
    return datetime.datetime(third.year,third.month,third.day)

def compute_smile_IF(date, maturity):
    path = 'C:/Database/CFFEX/Settlement/'
    date = str(date) # in case input an int
    maturity = str(maturity)
    with open(path + date + '_1.csv') as f:
        lines = f.read().split('\n')

    # time to maturity
    tau = daysBetween(datetime.datetime.strptime(date, '%Y%m%d').date(), IFExpiration(int('20'+maturity[:2]), int(maturity[2:4])))/240
    # interest rate and dividend rate
    r,q = 0.0,0
    # r = 0.0
    # underlying S
    tmp = lines[np.where(['IF'+maturity in line for line in lines])[0][0]].split(',')
    # 8 is close, 9 is settlement
    cs = 9
    underlying = float(tmp[cs])
    # print('Underlying prices: ' + str(underlying))
    # options K and V
    call_prices = []
    call_strikes = []
    put_prices = []
    put_strikes = []
    for line in lines:
        if 'IO' in line and maturity in line:
            tmp = line.split(',')
            if 'C' in line:
                call_prices.append(float(tmp[cs]))
                call_strikes.append(float(tmp[0].split('-')[-1]))
            elif 'P' in line:
                put_prices.append(float(tmp[cs]))
                put_strikes.append(float(tmp[0].split('-')[-1]))

    call_impv = [0]*len(call_prices)
    put_impv = [0]*len(put_prices)
    for idx in range(len(call_prices)):
        # if underlying <= call_strikes[idx]:
        if True:
            try:
                call_impv[idx] = blsImpv(underlying, call_strikes[idx], r, tau, call_prices[idx])
            except:
                pass
    for idx in range(len(put_prices)):
        # if underlying >= put_strikes[idx]:
        if True:
            try:
                put_impv[idx] = blsImpv(underlying, put_strikes[idx], r, tau, put_prices[idx],cpflag='P')
            except:
                pass
    impv = np.array(call_impv) + np.array(put_impv)
    # sns.scatterplot(x=call_strikes,y=impv)
    plt.figure()
    sns.scatterplot(x=call_strikes,y=call_impv)
    sns.scatterplot(x=call_strikes,y=put_impv)
    plt.title(date+'Impv')
    # -----------------------VEGA---------------------------------------------
    call_vega = [0]*len(call_prices)
    put_vega = [0]*len(put_prices)
    for idx in range(len(call_prices)):
        call_vega[idx] = blsVega(underlying, call_strikes[idx], r, tau, call_impv[idx])
        put_vega[idx] = blsVega(underlying, put_strikes[idx], r, tau, put_impv[idx])
    plt.figure()
    sns.scatterplot(x=call_strikes, y=call_vega)
    sns.scatterplot(x=put_strikes, y=put_vega)
    plt.title(date+'Vega')
    #---------------------Delta-----------------------------------------------
    call_delta = [0]*len(call_prices)
    put_delta = [0]*len(put_prices)
    for idx in range(len(call_prices)):
        call_delta[idx] = blsDelta(underlying, call_strikes[idx], r, tau, call_impv[idx])
        put_delta[idx] = 1+blsDelta(underlying, put_strikes[idx], r, tau, put_impv[idx], cpflag='P')
    plt.figure()
    sns.scatterplot(x=call_strikes, y=call_delta)
    sns.scatterplot(x=put_strikes, y=put_delta)
    plt.title(date+'Delta')
    #--------------------------Gamma------------------------------------------
    call_gamma = [0]*len(call_prices)
    put_gamma = [0]*len(put_prices)
    for idx in range(len(call_prices)):
        call_gamma[idx] = blsGamma(underlying, call_strikes[idx], r, tau, call_impv[idx])
        put_gamma[idx] = blsGamma(underlying, put_strikes[idx], r, tau, put_impv[idx])
    plt.figure()
    sns.scatterplot(x=call_strikes, y=call_gamma)
    sns.scatterplot(x=put_strikes, y=put_gamma)
    plt.title(date+'Gamma')
    #---------------------Theta---------------------------------------------
    call_theta = [0] * len(call_prices)
    put_theta = [0]*len(put_prices)
    for idx in range(len(call_prices)):
        call_theta[idx] = blsTheta(underlying, call_strikes[idx], r, tau, call_impv[idx])
        put_theta[idx] = blsTheta(underlying, put_strikes[idx], r, tau, put_impv[idx], cpflag='P')
    plt.figure()
    sns.scatterplot(x=call_strikes, y=call_theta)
    sns.scatterplot(x=put_strikes, y=put_theta)
    plt.title(date+'Theta')
    # plt.close('all')


# portfolio = dict()
# portfolio['IO2012-C-4700'] = 100
# portfolio['IO2012-C-4600'] = -100
# portfolio['IO2012-P-3100'] = 100
# portfolio['IO2012-P-3200'] = -100
# portfolio['IF2012'] = 3

def portfolioGreeks(date, portfolio):
    # portfolio example
    # 34 IO2009-C-4100
    path = 'C:/Database/CFFEX/Settlement/'
    date = str(date) # in case input an int
    with open(path + date + '_1.csv') as f:
        lines = f.read().split('\n')
    # use dateframe to record datas
    pg = pd.DataFrame(index=portfolio.keys(),columns=['quantity', 'price', 'impv', 'delta', 'gamma', 'vega', 'theta', 'ttm', 'change'])
    unders = dict() # save settlement price
    r,q = 0,0
    # go through each line
    for line in lines:
        tmp = line.split(',')
        tmp[0] = tmp[0].strip()
        # IF add to underlyings
        if 'IF' in tmp[0]:
            unders[tmp[0]] = float(tmp[9])
        # add to portfolio
        if tmp[0] in portfolio.keys():
            tau = daysBetween(datetime.datetime.strptime(date, '%Y%m%d').date(), IFExpiration(int('20'+tmp[0][2:4]), int(tmp[0][4:6])))/240
            # option or future
            if 'IF' in tmp[0]:
                pg.loc[tmp[0]]['quantity'] = portfolio[tmp[0]]
                pg.loc[tmp[0]]['price'] = float(tmp[9] )
                pg.loc[tmp[0]]['impv'] = 0
                pg.loc[tmp[0]]['delta'] = 1
                pg.loc[tmp[0]]['gamma'] = 0
                pg.loc[tmp[0]]['vega'] = 0
                pg.loc[tmp[0]]['theta'] = 0
                pg.loc[tmp[0]]['ttm'] = tau
                pg.loc[tmp[0]]['change'] = float(tmp[-2])
            elif 'IO' in tmp[0]:
                pg.loc[tmp[0]]['quantity'] = int(portfolio[tmp[0]])
                pg.loc[tmp[0]]['price'] = float(tmp[9])
                pg.loc[tmp[0]]['ttm'] = tau
                pg.loc[tmp[0]]['impv'] = blsImpv(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, float(tmp[9]), cpflag=tmp[0][7])
                pg.loc[tmp[0]]['delta'] = blsDelta(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, pg.loc[tmp[0]]['impv'],cpflag=tmp[0][7])
                pg.loc[tmp[0]]['gamma'] = blsGamma(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, pg.loc[tmp[0]]['impv'])
                pg.loc[tmp[0]]['vega'] = blsVega(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, pg.loc[tmp[0]]['impv'])
                pg.loc[tmp[0]]['theta'] = blsTheta(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, pg.loc[tmp[0]]['impv'],cpflag=tmp[0][7])/240
                pg.loc[tmp[0]]['change'] = float(tmp[-2])
    pg['totalValue'] = pg.price * pg.quantity
    pg['totalDelta'] = pg.delta * pg.quantity
    pg['totalGamma'] = pg.gamma * pg.quantity
    pg['totalVega'] = pg.vega * pg.quantity
    pg['totalTheta'] = pg.theta * pg.quantity
    pg['totalChange'] = pg.change * pg.quantity

    pg.loc['total'] = sum(pg.values)
    # pg.loc['total'] = sum(pg.values.astype(float))
    pg.loc['total'][['quantity','price','impv','delta','gamma','vega','theta','ttm','change']] = [0,0,0,0,0,0,0,0,0]
    pg.to_csv(date+'_pg.csv')
    return pg

def dailyGreeks(tradingDate):
    path = 'C:/Database/CFFEX/Settlement/'
    tradingDate = str(tradingDate) # in case input an int
    with open(path + tradingDate + '_1.csv') as f:
        lines = f.read().split('\n')
    # use dateframe to record datas
    greeks = pd.DataFrame(columns=['price', 'impv', 'delta', 'gamma', 'vega', 'theta', 'ttm', 'change'])
    unders = dict() # save settlement price
    r,q = 0.0,0
    # go through each line
    for line in lines:
        tmp = line.split(',')
        tmp[0] = tmp[0].strip()
        # if tmp[0] == 'IO2103-C-3200':
            # pdb.set_trace()
        # IF add to underlyings
        if 'IF' in tmp[0]:
            tau = daysBetween(datetime.datetime.strptime(tradingDate, '%Y%m%d').date(), IFExpiration(int('20'+tmp[0][2:4]), int(tmp[0][4:6])))/240
            unders[tmp[0]] = float(tmp[9])
            greeks.loc[tmp[0]] = 0
            greeks.loc[tmp[0]]['price'] = float(tmp[9] )
            greeks.loc[tmp[0]]['impv'] = 0
            greeks.loc[tmp[0]]['delta'] = 1
            greeks.loc[tmp[0]]['gamma'] = 0
            greeks.loc[tmp[0]]['vega'] = 0
            greeks.loc[tmp[0]]['theta'] = 0
            greeks.loc[tmp[0]]['ttm'] = tau
            greeks.loc[tmp[0]]['change'] = float(tmp[-2])
        elif 'IO' in tmp[0] and 'IF'+tmp[0][2:6] in unders:
            tau = daysBetween(datetime.datetime.strptime(tradingDate, '%Y%m%d').date(), IFExpiration(int('20'+tmp[0][2:4]), int(tmp[0][4:6])))/240
            greeks.loc[tmp[0]] = 0
            greeks.loc[tmp[0]]['price'] = float(tmp[9])
            greeks.loc[tmp[0]]['ttm'] = tau
            greeks.loc[tmp[0]]['impv'] = blsImpv(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, float(tmp[9]), cpflag=tmp[0][7])
            greeks.loc[tmp[0]]['delta'] = blsDelta(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, greeks.loc[tmp[0]]['impv'],cpflag=tmp[0][7])
            greeks.loc[tmp[0]]['gamma'] = blsGamma(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, greeks.loc[tmp[0]]['impv'])
            greeks.loc[tmp[0]]['vega'] = blsVega(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, greeks.loc[tmp[0]]['impv'])
            greeks.loc[tmp[0]]['theta'] = blsTheta(unders['IF'+tmp[0][2:6]], float(tmp[0][9:13]), r, tau, greeks.loc[tmp[0]]['impv'],cpflag=tmp[0][7])/240
            greeks.loc[tmp[0]]['change'] = float(tmp[-2])
    return greeks

def dailyImpVol(tradingDate):
    # if not os.path.isfile('C:/Database/CFFEX/Settlement/'+str(tradingDate)+'_1.csv'):
        # return [],[]
    path = 'C:/Users/Kai/Desktop/WonderWiz/Data/IF/'
    if not os.path.isfile(path+str(tradingDate)+'_1.csv'):
        return [],[],[],[]
    y = int(str(tradingDate)[:4])
    m = int(str(tradingDate)[4:6])
    d = int(str(tradingDate)[6:8])
    # record IF futures' settlement
    IFs = []
    # IF/IO maturity
    maturity = dict()
    # IF/IO time to maturity in years
    ttm = dict()
    # record futures and options daily settlement price
    futures = dict()
    options = dict()
    with open(path+str(tradingDate)+'_1.csv') as f:
        lines = f.read().split('\n')
        for line in lines:
            line_split = line.split(',')
            # dealing with futures
            if 'IF' in line_split[0]:
                code = line_split[0].strip()
                IFs.append(code)
                maturity[code] = IFExpiration(int('20'+code[2:4]), int(code[4:6]))
                ttm[code] = daysBetween(datetime.date(y,m,d), maturity[code])/240
                # today's settlement, and prior day's settlement
                futures[code] = [float(line_split[9]),float(line_split[10])]
                options['IO'+code[2:]] = pd.DataFrame(columns=['C','C0','P','P0'])
            elif 'IO' in line_split[0]:
                info = line_split[0].strip().split('-')
                if not 'IF'+info[0][2:] in futures.keys():
                    continue
                if int(info[2]) in options[info[0]].index:
                    if info[1] == 'C':
                        options[info[0]].loc[int(info[2])][['C','C0']] = [float(line_split[9]),float(line_split[10])]
                    elif info[1] == 'P':
                        options[info[0]].loc[int(info[2])][['P','P0']] = [float(line_split[9]),float(line_split[10])]
                else:
                    if info[1] == 'C':
                        options[info[0]].loc[int(info[2])] = [float(line_split[9]),float(line_split[10]),0,0]
                    elif info[1] == 'P':
                        options[info[0]].loc[int(info[2])] = [0,0,float(line_split[9]),float(line_split[10])]
    # first compute the implied volatility
    impv = dict()
    for code in futures.keys():
        month = code[2:]
        impv['IO'+month] = pd.DataFrame(index=options['IO'+month].index,columns=['civ','civ0','dciv','piv','piv0','dpiv'])
        for idx in options['IO'+month].index:
            civ = blsImpv(futures[code][0], idx, 0, ttm[code], options['IO'+month].loc[idx]['C'])
            civ0 = blsImpv(futures[code][1], idx, 0, ttm[code]+0.004, options['IO'+month].loc[idx]['C0'])
            dciv = civ - civ0
            piv = blsImpv(futures[code][0], idx, 0, ttm[code], options['IO'+month].loc[idx]['P'],cpflag='P')
            piv0 = blsImpv(futures[code][1], idx, 0, ttm[code]+0.004, options['IO'+month].loc[idx]['P0'],cpflag='P')
            dpiv = piv - piv0
            impv['IO'+month].loc[idx] = [civ, civ0, dciv, piv, piv0, dpiv]
    return impv,futures,options,ttm

def dailyATM(tradingDate):
    impv,futures,options,ttm = dailyImpVol(tradingDate)
    # print(impv)
    atm = pd.DataFrame(columns=['civ','piv','dS','dciv','dpiv','tau','K','underlying','rank'])
    rank = 0
    for future in futures:
        S = futures[future][0]
        if len(impv['IO'+future[2:]]) == 0:
            continue
        K = impv['IO'+future[2:]].index[np.argmin(abs(impv['IO'+future[2:]].index-S))]
        civ = impv['IO'+future[2:]].loc[impv['IO'+future[2:]].index[np.argmin(abs(impv['IO'+future[2:]].index-S))]].civ
        piv = impv['IO'+future[2:]].loc[impv['IO'+future[2:]].index[np.argmin(abs(impv['IO'+future[2:]].index-S))]].piv
        dciv = impv['IO'+future[2:]].loc[impv['IO'+future[2:]].index[np.argmin(abs(impv['IO'+future[2:]].index-S))]].dciv
        dpiv = impv['IO'+future[2:]].loc[impv['IO'+future[2:]].index[np.argmin(abs(impv['IO'+future[2:]].index-S))]].dpiv
        dS = futures[future][0] - futures[future][1]
        tau = ttm[future]
        atm.loc[rank] = [civ,piv,dS,dciv,dpiv,tau,K,future,rank]
        rank = rank + 1
    # print(atm)
    return atm


def etf_greeks(underlying, month, trading_date, am=0):
    from jqdatasdk import auth,is_auth,opt,get_price,query
    auth('18616633529','Wuzhi2020')
    is_auth = is_auth()
    # trading_date = '2020-10-21'
    # underlying = '159919.XSHE'
    # month = 2011
    underlying = str(underlying)
    options = opt.run_query(query(opt.OPT_CONTRACT_INFO).filter(opt.OPT_CONTRACT_INFO.underlying_symbol==underlying))
    codes = options.loc[[str(month)+'M' in x and x[-1]!='A' for x in options['trading_code']]].code.values
    trading_codes = options.loc[[str(month)+'M' in x and x[-1]!='A' for x in options['trading_code']]].trading_code.values
    Ks = options.loc[[str(month)+'M' in x and x[-1]!='A' for x in options['trading_code']]].exercise_price.values

    # calculation parameters
    r,q = 0,0
    #today = datetime.date.today()
    today = datetime.datetime.strptime(trading_date,'%Y-%m-%d')
    maturity = fourthWednesday(2000+int(str(month)[:2]), int(str(month)[2:]))
    tau = daysBetween(today,maturity)/240
    if am == 1:
        tau = tau + 0.002
    print(tau)
    # get underlying Price
    S = get_price(underlying, start_date=trading_date, end_date=trading_date, frequency='daily').close[0]

    # initial DataFrame
    df = pd.DataFrame(index=np.unique(Ks),columns=['Theta1_C','Theta5_C','Theta_C','Vega_C','Gamma_C','Delta_C','CIV','C','K','P','PIV','Delta_P','Gamma_P','Vega_P','Theta_P','Theta1_P','Theta5_P'])
    for idx,code in enumerate(codes):
        if 'P' in trading_codes[idx]:
            cpflag = 'P'
        if 'C' in trading_codes[idx]:
            cpflag = 'C'
        K = Ks[idx]
        V = get_price(code, start_date=trading_date, end_date=trading_date).close.values[0]
        if np.isnan(V):
            continue
        iv = blsImpv(S, K, r, tau, V, q=q, cpflag=cpflag)
        delta = blsDelta(S, K, r, tau, iv, q=q, cpflag=cpflag)
        gamma = blsGamma(S, K, r, tau, iv, q=q)
        vega  = blsVega(S, K, r, tau, iv, q=q)
        theta = blsTheta(S, K, r, tau, iv, q=q, cpflag=cpflag)
        theta1 = blsTheta1(S, K, r, tau, iv, q=q, cpflag=cpflag)
        theta5 = blsTheta5(S, K, r, tau, iv, q=q, cpflag=cpflag)
        if cpflag == 'C':
            df.loc[K]['Theta5_C','Theta1_C','Theta_C','Vega_C','Gamma_C','Delta_C','CIV','C','K'] = [theta5,theta1,theta,vega,gamma,delta,iv,V,K]
        if cpflag == 'P':
            df.loc[K]['Theta5_P','Theta1_P','Theta_P','Vega_P','Gamma_P','Delta_P','PIV','P','K'] = [theta5,theta1,theta,vega,gamma,delta,iv,V,K]
    df = df.astype(float)
    return df,S

def simple_iv(underlying, month, trading_date, am=0):
    df,S = etf_greeks(underlying, month, trading_date)
    tmp = df[['C','P','CIV','PIV']]
    tmp.to_csv('C:/Users/Kai/OneDrive/WonderWiz/'+underlying+trading_date+str(month)+'.csv')
    print(S)
    # print(df)

def nextExpiration(underlying, trading_date):
    if type(trading_date) == str:
        trading_date = datetime.datetime.strptime(trading_date, '%Y%m%d')
    elif type(trading_date) == pd._libs.tslibs.timestamps.Timestamp:
        trading_date = datetime.datetime(trading_date.year, trading_date.month, trading_date.day)

    if underlying in ['SH510300','SH510050','ZS159919','510300','510050','159919']:
        thisExpiration = fourthWednesday(trading_date.year, trading_date.month)
        if thisExpiration >= trading_date:
            return thisExpiration
        else:
            if trading_date.month < 12:
                return fourthWednesday(trading_date.year, trading_date.month+1)
            else:
                return fourthWednesday(trading_date.year + 1, 1)
    else:
        return None

def daysToNextExpiration(underlying, trading_date):
    if type(trading_date) == str:
        trading_date = datetime.datetime.strptime(trading_date, '%Y%m%d')
    elif type(trading_date) == pd._libs.tslibs.timestamps.Timestamp:
        trading_date = datetime.datetime(trading_date.year, trading_date.month, trading_date.day)
    return daysBetween(trading_date,nextExpiration(underlying, trading_date))/240

def volLevel(underlying, month, trading_date, tp=0):
    # type 0: vix type, use all option data to compute the vix index
    # type 1: lowest type, use simply the lowest implied
    # type 2: svi paramatrization, and the parameter m
    df,S = etf_greeks(underlying, month, trading_date)
    today = datetime.datetime.strptime(trading_date,'%Y-%m-%d')
    maturity = fourthWednesday(2000+int(str(month)[:2]), int(str(month)[2:]))
    tau = daysBetween(today,maturity)/240
    # here the prices are last price not MID!!! Need to modified further
    sigma2 = 0
    if tp == 0:
        jdx = np.where(S-df.K.values>0)[0][-1]
        # put options
        for idx in range(jdx+1):
            if idx == 0:
                sigma2 = sigma2 + 2*(df.iloc[idx+1].K-df.iloc[idx].K)*df.iloc[idx].P/(tau*df.iloc[idx].K**2)
            else:
                sigma2 = sigma2 + (df.iloc[idx+1].K-df.iloc[idx-1].K)*df.iloc[idx].P/(tau*df.iloc[idx].K**2)
        # last below S
        sigma2 = sigma2 - (S/df.iloc[idx].K-1)**2/tau
        # call option
        for idx in range(jdx+1,df.shape[0]):
            if idx == df.shape[0]-1:
                sigma2 = sigma2 + 2*(df.iloc[idx].K-df.iloc[idx-1].K)*df.iloc[idx].C/(tau*df.iloc[idx].K**2)
            else:
                sigma2 = sigma2 + (df.iloc[idx+1].K-df.iloc[idx-1].K)*df.iloc[idx].C/(tau*df.iloc[idx].K**2)
    elif tp == 1:
        pass
    elif tp == 2:
        pass
    sigma = np.sqrt(sigma2)
    return sigma

def otmVol(df,S):
    # input df from etf_greeks to select otm implied vol
    iv = df.CIV.values
    for i in range(df.shape[0]):
        if df.iloc[i].K < S:
            iv[i] = df.iloc[i].PIV
        else:
            break
    return iv

def impForwardLevel(df, tau=None, r=0):
    if tau == None:
        tau = df.tau[0]
    # C和P差别最小的点
    m = np.argmin(np.abs(df['C'].values-df['P'].values))
    F = df['K'][m] + np.exp(r*tau)*(df['C'][m]-df['P'][m])
    if df['K'][m] > F:
        m = m - 1
    return F,m

def impVIX_9(df, S):
    # vix style vol index with nearest 9 strikes
    pass

def impVIXOneSide(df, tau=None, r=0):
    # vix style vol index with nearest 9 strikes
    # (1) compute the forward price level
    if tau == None:
        tau = df.tau[0]
    F,m = impForwardLevel(df, tau, r)
    sigma2 = 0
    for idx in range(len(df)):
        if idx == 0:
            dK = df['K'][idx+1] - df['K'][idx]
        elif idx == (len(df)-1):
            dK = df['K'][idx] - df['K'][idx-1]
        else:
            dK = (df['K'][idx+1] - df['K'][idx-1]) / 2
        if idx < m:
            sigma2 = sigma2 + 2*dK*np.exp(r*tau)*df['P'][idx]/tau/(df['K'][idx]**2)
        elif idx == m:
            sigma2 = sigma2 + (F/df['K'][m]-1)**2/tau
        elif idx > m:
            sigma2 = sigma2 + 2*dK*np.exp(r*tau)*df['C'][idx]/tau/(df['K'][idx]**2)
    return np.sqrt(sigma2)*100

def impVIXTwoSide(df1, df2, tau1=None, tau2=None, r1=0, r2=0):
    if tau1 == None:
        tau1 = df1.tau[0]
    if tau2 == None:
        tau2 = df2.tau[0]
    v1 = impVIXOneSide(df1, tau1, r1)**2
    v2 = impVIXOneSide(df2, tau2, r2)**2
    # assume one month is 20 trading day
    # tau = 20/240
    # v = np.sqrt(12*(tau1*v1*(tau2-tau)/(tau2-tau1) + tau2*v2*(tau-tau1)/(tau2-tau1)))
    w1,w2 = get_weight(tau1,tau2)
    v = np.sqrt(12*(tau1*v1*w1 + tau2*v2*w2))
    return v

def impSkewOneSide(df, tau=None, r=0):
    if tau == None:
        tau = df.tau[0]
    F,m = impForwardLevel(df, tau, r)
    if m == -1:
        m = 1
    elif abs(df.K[m]-F) > abs(df.K[m+1]-F):
        m = m + 1
    K0 = df['K'][m]
    F0 = df['K'][m] + np.exp(r*tau)*(df['C'].iloc[m]-df['P'].iloc[m])
    eps1 = -(1+np.log(F0/K0)-F0/K0)
    eps2 = 2*np.log(K0/F0)*(F0/K0-1)+0.5*np.log(K0/F0)**2
    eps3 = 3*np.log(K0/F0)**2*(1/3*np.log(K0/F0)-1+F0/K0)
    P1,P2,P3 = 0,0,0
    # for idx in range(len(df)):
    #     # compute delta_K
    #     if idx == 0:
    #         dK = df['K'][idx+1] - df['K'][idx]
    #     elif idx == (len(df)-1):
    #         dK = df['K'][idx] - df['K'][idx-1]
    #     else:
    #         dK = (df['K'][idx+1] - df['K'][idx-1]) / 2
    #     if df.K[idx] < F0:
    #         P1 = P1 - np.exp(r*tau)*dK*df['P'][idx]/(df['K'][idx]**2)
    #         P2 = P2 + np.exp(r*tau)*2*dK*df['P'][idx]*(1-np.log(df['K'][idx]/F0))/(df['K'][idx]**2)
    #         P3 = P3 + np.exp(r*tau)*3*dK*df['P'][idx]*(2*np.log(df['K'][idx]/F0)-np.log(df['K'][idx]/F0)**2)/(df['K'][idx]**2)
    #     else:
    #         P1 = P1 - np.exp(r*tau)*dK*df['C'][idx]/(df['K'][idx]**2)
    #         P2 = P2 + np.exp(r*tau)*2*dK*df['C'][idx]*(1-np.log(df['K'][idx]/F0))/(df['K'][idx]**2)
    #         P3 = P3 + np.exp(r*tau)*3*dK*df['C'][idx]*(2*np.log(df['K'][idx]/F0)-np.log(df['K'][idx]/F0)**2)/(df['K'][idx]**2)
    for idx in range(len(df)):
        # compute delta_K
        if idx == 0:
            dK = df['K'][idx+1] - df['K'][idx]
        elif idx == (len(df)-1):
            dK = df['K'][idx] - df['K'][idx-1]
        else:
            dK = (df['K'][idx+1] - df['K'][idx-1]) / 2
        if idx <= m:
            P1 = P1 - np.exp(r*tau)*dK*df['P'][idx]/(df['K'][idx]**2)
            P2 = P2 + np.exp(r*tau)*2*dK*df['P'][idx]*(1-np.log(df['K'][idx]/F0))/(df['K'][idx]**2)
            P3 = P3 + np.exp(r*tau)*3*dK*df['P'][idx]*(2*np.log(df['K'][idx]/F0)-np.log(df['K'][idx]/F0)**2)/(df['K'][idx]**2)
        else:
            P1 = P1 - np.exp(r*tau)*dK*df['C'][idx]/(df['K'][idx]**2)
            P2 = P2 + np.exp(r*tau)*2*dK*df['C'][idx]*(1-np.log(df['K'][idx]/F0))/(df['K'][idx]**2)
            P3 = P3 + np.exp(r*tau)*3*dK*df['C'][idx]*(2*np.log(df['K'][idx]/F0)-np.log(df['K'][idx]/F0)**2)/(df['K'][idx]**2)
        # print(idx,P2)
    P1 = P1 + eps1
    P2 = P2 + eps2
    P3 = P3 + eps3
    S = (P3-3*P1*P2+2*np.power(P1,3))/np.power(P2-np.power(P1,2),1.5)
    return 100-10*S

def get_weight(tau1, tau2):
    d1,d2 = int(tau1*240),int(tau2*240)
    if d1 < 5.2:
        return [0,1]
    elif d1 < 20.2:
        c1,c2 = [d1-4,d2-4]
        return [c1/16, 1-c1/16]
    else:
        return [1,0]

def impSkewTwoSide(df1, df2, tau1=None, tau2=None, r1 = 0, r2 = 0):
    if tau1 == None:
        tau1 = df1.tau[0]
    if tau2 == None:
        tau2 = df2.tau[0]
    w1,w2 = get_weight(tau1,tau2)
    if w1 == 0:
        return impSkewOneSide(df2, tau2, r2)
    elif w2 == 0:
        return impSkewOneSide(df1, tau1, r1)
    else:
        s1 = impSkewOneSide(df1, tau1, r1)
        s2 = impSkewOneSide(df2, tau2, r2)
        s = w1*s1 + w2*s2
        return s
    # tau = 20/240
    # s  = s1*(tau2-tau)/(tau2-tau1) + s2*(tau-tau1)/(tau2-tau1)


def addOTMIVs(df, S, tau, r=0):
    F,m = impForwardLevel(df, tau, r)
    # calculate implied volatilities
    df['CIV'] = 0
    df['PIV'] = 0
    for idx in range(len(df)):
        df['CIV'].iloc[idx] = blsImpv(S, df['K'][idx], r, tau, df['C'][idx])
        df['PIV'].iloc[idx] = blsImpv(S, df['K'][idx], r, tau, df['P'][idx], cpflag='P')
    # pick out otm options
    df['otmIVs'] = 0
    for idx in range(len(df)):
        if df['K'][idx] < F:
            df['otmIVs'].iloc[idx] = df['PIV'][idx]
        else:
            df['otmIVs'].iloc[idx] = df['CIV'][idx]
    return df

#-----------------------Vanna Volga Methods-----------------------------------
def vanna_volga_x(ks, vegas, k0, vega0):
    # first one is target, the following 3 are the references
    x = [0]*3
    x[0] = (vega0/vegas[0]) * np.log(ks[1]/k0)*np.log(ks[2]/k0) / (np.log(ks[1]/ks[0])*np.log(ks[2]/ks[0]))
    x[1] = (vega0/vegas[1]) * np.log(k0/ks[1])*np.log(ks[2]/k0) / (np.log(ks[1]/ks[0])*np.log(ks[2]/ks[1]))
    x[2] = (vega0/vegas[2]) * np.log(k0/ks[0])*np.log(k0/ks[1]) / (np.log(ks[2]/ks[0])*np.log(ks[2]/ks[1]))
    return x

def vanna_volga_vol(ks, vols, k0):
    # return vanna volga implied volatility
    vol = 0
    vol = vol + vols[0]* np.log(ks[1]/k0)*np.log(ks[2]/k0)/(np.log(ks[1]/ks[0])*np.log(ks[2]/ks[0]))
    vol = vol + vols[1]* np.log(k0/ks[0])*np.log(ks[2]/k0)/(np.log(ks[1]/ks[0])*np.log(ks[2]/ks[1]))
    vol = vol + vols[2]* np.log(k0/ks[0])*np.log(k0/ks[1])/(np.log(ks[2]/ks[0])*np.log(ks[2]/ks[1]))
    return vol

def vanna_volga_price(ks, Cs_mkt, k0, S, vol_atm, tau, r=0):
    # more parameters need to be added.
    # ks and Cs_mkt are 3 by 1 vectors
    vegas = [0]*3
    for i in range(3):
        vegas[i] = blsVega(S, ks[i], r, tau, vol_atm)
    vega0 = blsVega(S, k0, r, tau, vol_atm)
    x = vanna_volga_x(ks, vegas, k0, vega0)
    C = blsPrice(S, k0, r, tau, vol_atm)
    C = C + x[0]*(Cs_mkt[0]-blsPrice(S, ks[0], r, tau, vol_atm))
    C = C + x[1]*(Cs_mkt[1]-blsPrice(S, ks[1], r, tau, vol_atm))
    C = C + x[1]*(Cs_mkt[1]-blsPrice(S, ks[2], r, tau, vol_atm))
    return C

#--------------------------SVI Parametrization--------------------------------
def svi_calibrate(k,w):
    # Zeliade Systems, Quasi-explicit calibration of Gatheral’s SVI model, Zeliade white paper, 2009.
    # k is the log strike price, k = log(K/F_t)
    # w is the total implied variance w
    def adc(m,s,k,w):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        A = np.array([[len(k),np.sum(y),np.sum(z)],
                      [np.sum(y),np.dot(y,y),np.dot(y,z)],
                      [np.sum(z),np.dot(y,z),np.dot(z,z)]])
        b = np.array([np.sum(w),np.dot(y,w),np.dot(z,w)])
        adc = np.linalg.solve(A,b)
        a,d,c = adc[0],adc[1],adc[2]
        # check constraints
        if a < 0:
            a = 0
        elif a > np.max(w):
            a = np.max(w)
        if c < 0:
            c = 0
        elif c > 4*s:
            c = 4*s
        bd = np.min([np.abs(c), np.abs(4*s-c)])
        if d < -bd:
            d = bd
        elif d > bd:
            d = bd
        return adc
    def svi(adc, m, s, k):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        return adc[0]+adc[1]*y+adc[2]*z
    def loss(x):
        m,s = x[0],x[1]
        # return np.sum(np.power(svi(adc(m,s,k,w),m,s,k)-w,2))
        return svi(adc(m,s,k,w),m,s,k)-w
    x0 = [0,1]
    result = least_squares(loss, x0, bounds=([-np.inf, 0], [np.inf, np.inf]))
    m,s = result.x[0],result.x[1]
    tmp = adc(m,s,k,w)
    a,d,c = tmp[0],tmp[1],tmp[2]
    b = c/s
    rho = d/c
    return [a,b,rho,m,s]

def svi_calibrate_weighted(k, w, weight):
    # Zeliade Systems, Quasi-explicit calibration of Gatheral’s SVI model, Zeliade white paper, 2009.
    # k is the log strike price, k = log(K/F_t)
    # w is the total implied variance w
    # weight = np.exp(-k**2*100)
    print(weight)
    def adc(m,s,k,w):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        A = np.array([[len(k),np.sum(y),np.sum(z)],
                      [np.sum(y),np.dot(y,y),np.dot(y,z)],
                      [np.sum(z),np.dot(y,z),np.dot(z,z)]])
        b = np.array([np.sum(w),np.dot(y,w),np.dot(z,w)])
        adc = np.linalg.solve(A,b)
        a,d,c = adc[0],adc[1],adc[2]
        # check constraints
        if a < 0:
            a = 0
        elif a > np.max(w):
            a = np.max(w)
        if c < 0:
            c = 0
        elif c > 4*s:
            c = 4*s
        bd = np.min([np.abs(c), np.abs(4*s-c)])
        if d < -bd:
            d = bd
        elif d > bd:
            d = bd
        return adc
    def svi(adc, m, s, k):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        return adc[0]+adc[1]*y+adc[2]*z
    def loss(x):
        m,s = x[0],x[1]
        # return sum(np.power(100*(svi(adc(m,s,k,w),m,s,k)-w),2) * (weight**2))
        return (svi(adc(m,s,k,w),m,s,k)-w) * weight
    x0 = [-0.01,0.3]
    result = least_squares(loss, x0, bounds=([-np.inf, 0], [np.inf, np.inf]))
    # result = minimize(loss, x0, bounds=([-np.inf, 0], [np.inf, np.inf]))
    m,s = result.x[0],result.x[1]
    tmp = adc(m,s,k,w)
    a,d,c = tmp[0],tmp[1],tmp[2]
    b = c/s
    rho = d/c
    return [a,b,rho,m,s]


def svi_recalibrate(k,w,theta0):
    # theta 0 is the give slice parameters
    def adc(m,s,k,w):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        A = np.array([[len(k),np.sum(y),np.sum(z)],
                      [np.sum(y),np.dot(y,y),np.dot(y,z)],
                      [np.sum(z),np.dot(y,z),np.dot(z,z)]])
        b = np.array([np.sum(w),np.dot(y,w),np.dot(z,w)])
        adc = np.linalg.solve(A,b)
        a,d,c = adc[0],adc[1],adc[2]
        # check constraints
        if a < 0:
            a = 0
        elif a > np.max(w):
            a = np.max(w)
        if c < 0:
            c = 0
        elif c > 4*s:
            c = 4*s
        bd = np.min([np.abs(c), np.abs(4*s-c)])
        if d < -bd:
            d = bd
        elif d > bd:
            d = bd
        return adc
    def svi(adc, m, s, k):
        y = (k-m)/s
        z = np.sqrt(1+np.power(y,2))
        return adc[0]+adc[1]*y+adc[2]*z
    def loss(x):
        m,s = x[0],x[1]
        tmp = adc(m,s,k,w)
        a,d,c = tmp[0],tmp[1],tmp[2]
        b = c/s
        rho = d/c
        theta1 = [a,b,rho,m,s]
        roots = svi_find_crossing(theta0, theta1)
        if len(roots) == 0:
            penalty = 0
        else:
            ks = [roots[0]-1] + [(roots[i]+roots[i+1])/2 for i in range(len(roots)-1)] + [roots[-1]+1]
            penalty = max([max(0, svi_calculate(theta0, k)-svi_calculate(theta1, k)) for k in ks])
        return np.sum(np.power(svi(adc(m,s,k,w),m,s,k)-w,2)) + 10000*penalty
    x0 = [0,1]
    result = least_squares(loss, x0, bounds=([-np.inf, 0], [np.inf, np.inf]))
    m,s = result.x[0],result.x[1]
    tmp = adc(m,s,k,w)
    a,d,c = tmp[0],tmp[1],tmp[2]
    b = c/s
    rho = d/c
    return [a,b,rho,m,s]

def svi_calculate(theta, k):
    # 返回值是total implied variance = implied variance * t = (implied volatility)^2 * t, 一般符号是v
    # v = w/t = s^2
    [a,b,rho,m,s] = theta
    c = s*b
    d = rho*c
    y = (k-m)/s
    z = np.sqrt(1+np.power(y,2))
    return a+d*y+c*z

def svi_raw_to_natural(theta_raw):
    # delta, mu, rho, omega, zeta
    [a,b,rho,m,s] = theta_raw
    # delta = a -
    # 以后再补

def svi_natural_to_raw(theta_natural):
    delta,mu,rho,omega,zeta = theta_natural
    # 以后再补

def svi_raw_to_JW(theta, t):
    # v,phi,p,c,v_tilda
    # v gives the ATM variance
    # phi gives the ATM skew
    # p gives the slope of the left (put) wing
    # c gives the slope of the right (call) wing
    # v_tilda is the minimum implied variance
    [a,b,r,m,s] = theta
    v = (a + b * (-r * m + np.sqrt(m*m + s*s)) )/ t
    w = v * t
    phi = b / 2 / np.sqrt(w) * (r - m / np.sqrt(m*m + s*s))
    p = b * (1 - r) / np.sqrt(w)
    c = b * (1 + r) / np.sqrt(w)
    v_tilda = (a + b * s * np.sqrt(1 - r * r)) / t
    return [v, phi, p, c, v_tilda]

def svi_JW_to_raw(theta, t):
    # method from Arbitrage-free SVI volatility surfaces by Jim Gatheral, Antonie Jacquier
    [v, phi, p, c, v_tilda] = theta
    w = v * t
    #-----------------------------------------
    b = np.sqrt(w) * (c + p) / 2
    r = 1 - p * np.sqrt(w) / b
    #-----------------------------------------
    # beta, alpha
    beta = r - 2 * phi * np.sqrt(w) / b
    alpha = np.sign(beta) * np.sqrt(1/np.power(beta, 2) - 1)
    #-----------------------------------------
    m = (v - v_tilda) * t / (b * (-r + np.sign(alpha) * np.sqrt(1 + alpha*alpha) - alpha * np.sqrt(1 - r*r)))
    if m == 0:
        # a = v_tilda * t - b * s * np.sqrt(1 - r*r)
        a = (v_tilda - v*(1-np.sqrt(1-r*r))) / (1-np.sqrt(1-r*r))
        s = (v * t - a) / b
    else:
        s = alpha * m
        a = v_tilda * t - b * s * np.sqrt(1 - r*r)
    return [a,b,r,m,s]

def eliminate_butterfly_JW(theta, t):
    # tuning c and v_tilda
    [v, phi, p, c, v_tilda] = theta
    c_prime = p + 2 * phi
    v_tilda_prime = v_tilda * 4 * p * c_prime / np.power(p + c_prime, 2)
    return [v, phi, p, c_prime, v_tilda_prime]

def svi_calibrate_no_arbitrage_single(k, w, t):
    '''
        k: log moneyness
        w: total variance
        t: time to maturity
    '''
    # single slice
    theta_raw = svi_calibrate(k,w)
    theta_JW_nb = eliminate_butterfly_JW(svi_raw_to_JW(theta_raw, t), t)
    theta_raw_nb = svi_JW_to_raw(theta_JW_nb, t)
    return theta_raw_nb

def svi_w(theta, k, t):
    # 假设theta是raw svi的参数
    w = svi_calculate(theta, k) * t
    return w

def svi_dw_dk(theta, k, t, h=0.001):
    dk = k * h
    wp = svi_w(theta, k+dk, t)
    wm = svi_w(theta, k-dk, t)
    return (wp - wm) / (2 * dk)

def svi_dw2_dk2(theta, k, t, h=0.001):
    dk = k * h
    wp = svi_w(theta, k+dk, t)
    w0 = svi_w(theta, k,    t)
    wm = svi_w(theta, k-dk, t)
    return (wp - 2 * w0 + wm) / (dk * dk)

def svi_g(theta, k, t, h=0.001):
    term1 = np.power(1 - k * svi_dw_dk(theta, k, t, h) / (2 * svi_w(theta, k, t)), 2)
    term2 = -(1/svi_w(theta, k, t)+0.25) * np.power(svi_dw_dk(theta, k, t, h),2) / 4
    term3 = svi_dw2_dk2(theta, k, t, h) / 2
    return term1 + term2 + term3

def svi_find_crossing(theta1, theta2):
    # 找两个slices之间交错的部分
    a1,b1,r1,m1,s1 = theta1
    a2,b2,r2,m2,s2 = theta2
    alpha = a2 - a1 + b1*r1*m1 - b2*r2*m2
    beta = b2*r2 - b1*r1
    # coefficients of quartic polynomials
    c4 = np.power(b1*b1-b2*b2-beta*beta,2) - 4*b2*b2*beta*beta # 4'th degree coefficient
    c3 = 8*b2*b2*(m2*beta*beta-alpha*beta) - 4*(b1*b1-b2*b2-beta*beta)*(alpha*beta+b1*b1*m1-b2*b2*m2)
    c2 = 4*(alpha*beta+b1*b1*m1-b2*b2*m2) + 2*(b1*b1-b2*b2-beta*beta)*(b1*b1*(m1*m1+s1*s1)-b2*b2*(m2*m2+s2*s2)-alpha*alpha) + 4*b2*b2*(4*m2*alpha*beta+(m2*m2+s2*s2)*beta*beta-alpha*alpha)
    c1 = 8*b2*b2*(alpha*alpha*m2-alpha*beta*(m2*m2+s2*s2)) - 4*(b1*b1*(m1*m1+s1*s1)-b2*b2*(m2*m2+s2*s2)-alpha*alpha)*(alpha*beta+b1*b1*m1-b2*b2*m2)
    c0 = np.power(b1*b1*(m1*m1+s1*s1)-b2*b2*(m2*m2+s2*s2)-alpha*alpha, 2) - 4*b2*b2*alpha*alpha*(m2*m2+s2*s2)
    roots = np.roots(np.array([c4,c3,c2,c1,c0])*10000000)
    # 实数根
    real_roots = roots[abs(np.imag(roots))<1e-10]
    def test(x, theta1, theta2):
        return svi_calculate(theta1, x) - svi_calculate(theta2, x)
    # def test(x, [c4,c3,c2,c1,c0]):
    # 真正的根
    n_real_roots = sum([abs(test(r, theta1, theta2))<1e-10 for r in real_roots])
    return [r for r in real_roots if abs(test(r, theta1, theta2))<1e-10]

def svi_interpolation_ATM_ITV(theta0,theta1, t0, t1, t):
    w0 = svi_calculate(theta0, 0)
    w1 = svi_calculate(theta1, 0)
    # use linear interpolation first, could be modified to any monotonic intepolation scheme
    wt = (t1-t)/(t1-t0) * w0 + (t-t0)/(t1-t0) * w1
    alpha = (np.sqrt(w1) - np.sqrt(wt)) / (np.sqrt(w1) - np.sqrt(w0))
    return alpha

def svi_interpolation_Call(theta0, theta1, t0, t1, t, k, S):
    # S is the underlying spot price
    # k is the log moneyness  k=log(K/S)
    alpha = svi_interpolation_ATM_ITV(theta0,theta1, t0, t1, t)
    K = S*np.exp(k)
    iv0 = np.sqrt(svi_calculate(theta0, k)/t0)
    iv1 = np.sqrt(svi_calculate(theta1, k)/t1)
    C0 = blsPrice(S, K, 0, t0, iv0)
    C1 = blsPrice(S, K, 0, t1, iv1)
    Ct = alpha*C0 + (1-alpha)*C1
    return Ct

def svi_interpolation_IV(theta0, theta1, t0, t1, t, k, S):
    Ct = svi_interpolation_Call(theta0, theta1, t0, t1, t, k, S)
    iv = blsImpv(S, S*np.exp(k), 0, t, Ct)
    return iv


#----------Stochastic Alpha Beta Rho (SABR) Start------------------------------
def sabr_atm(abrv, K, S, tau):
    # return the sabr atm implied volatility
    # alpha, beta, rho, vov
    a,b,r,v = abrv
    vol = a/np.power(S, 1-b)*(1+(np.power((1-b)*a,2)/24/np.power(S,2-2*b)+a*b*r*v/4/np.power(S,1-b)+np.power(v,2)*(2-3*np.power(r,2))/24)*tau)
    return vol

def sabr_calculate(abrv, K, S, tau):
    # return sabr implied volatility
    # alpha, beta, rho, vov
    a,b,r,v = abrv
    z = v/a*np.power(S*K,(1-b)/2)*np.log(S/K)
    xz = np.log((np.sqrt(1-2*r*z+np.power(z,2))+z-r)/(1-r))
    vol = a*z/xz/np.power(S*K,(1-b)/2)
    vol = vol / (1 + np.power(np.log(S/K),2)*np.power(1-b,2)/24+np.power(S/K,4)*np.power(1-b,4)/1920)
    vol = vol * (1 + (np.power(1-b,2)*np.power(a,2)/np.power(S*K,1-b)/24 + r*b*v*a/np.power(S*K,(1-b)/2)/4 + (2-3*np.power(r,2))*np.power(v,2)/24)*tau)
    return vol

def sabr_calibrate(ks, vols, S, tau, alpha=None, beta=1, rho=None, v=None):
    # should return sabr parameters: a,b,r,v
    # currently, assume is coming from {1,0.5,0}, leave out the aesthetic consideration method
    if np.array(tau).size == 1:
        if alpha:
            # need to fit r, v
            def fun(rv, a, b, ks, vols, S, tau):
                r,v = rv
                error = np.zeros(len(ks))
                for i,k in enumerate(ks):
                    error[i] = np.power(vols[i]-sabr_calculate([a,b,r,v], k, S, tau),2)
                return sum(error)
            loss = lambda rv: fun(rv, a=alpha, b=beta, ks=ks, vols=vols, S=S, tau=tau)
            bounds = Bounds([-1,0],[1, 1000])
            res = minimize(loss, [0,1], method='trust-constr', bounds=bounds)
            r,v = res.x
            return [alpha, beta, r, v]
        else:
            # need to fit a,r,v
            def fun(arv, b, ks, vols, S, tau):
                a,r,v = arv
                error = np.zeros(len(ks))
                for i,k in enumerate(ks):
                    error[i] = np.power(vols[i]-sabr_calculate([a,b,r,v], k, S, tau),2)
                return sum(error)
            loss = lambda arv: fun(arv, b=beta, ks=ks, vols=vols, S=S, tau=tau)
            bounds = Bounds([1e-10,-1,0],[2,1-1e-4, 10])
            # res = minimize(loss, [0.2,0,0.3], method='SLSQP', bounds=bounds)
            res = minimize(loss, [0.2,0,0.3], method='trust-constr', bounds=bounds)
            a,r,v = res.x
            return [a, beta, r, v]
    else:
        return sabr_calibrate_multiple_maturity(ks, vols, S, tau, alpha=alpha, beta=beta, rho=rho, v=v)

def sabr_calibrate2(ks, vols, S, tau, alpha=None, beta=1, rho=None, v=None):
    # should return sabr parameters: a,b,r,v
    # currently, assume is coming from {1,0.5,0}, leave out the aesthetic consideration method
    # add initial guess follow the paper by Floch and Kennedy
    # Explicit SABR Calibration through Simple Expansions 2014
    if np.array(tau).size == 1:
        if alpha:
            # need to fit r, v
            def fun(rv, a, b, ks, vols, S, tau):
                r,v = rv
                error = np.zeros(len(ks))
                for i,k in enumerate(ks):
                    error[i] = np.power(vols[i]-sabr_calculate([a,b,r,v], k, S, tau),2)
                return sum(error)
            loss = lambda rv: fun(rv, a=alpha, b=beta, ks=ks, vols=vols, S=S, tau=tau)
            bounds = Bounds([-1,0],[1, 1000])
            res = minimize(loss, [0,1], method='trust-constr', bounds=bounds)
            r,v = res.x
            return [alpha, beta, r, v]
        else:
            z = np.log(ks/S)
            rslt = np.linalg.lstsq(np.vstack((np.ones(z.shape),z,z**2)).T,vols,rcond=None)
            ss = rslt[0]
            a0 = ss[0]*np.power(S, 1-beta)
            if 3*ss[0]*ss[2]-0.5*np.power(ss[0]*(1-beta),2)+1.5*np.power(2*ss[1]+(1-beta)*ss[0],2) < 0:
                r0 = np.sign(2*ss[1]+(1-beta)*ss[0])
                v0 = (2*ss[1]+(1-beta)*ss[0]) / r0
            else:
                v0 = np.sqrt(3*ss[0]*ss[2]-0.5*np.power(ss[0]*(1-beta),2)+1.5*np.power(2*ss[1]+(1-beta)*ss[0],2))
                r0 = (2*ss[1]+(1-beta)*ss[0]) / v0
            # need to fit a,r,v
            def fun(arv, b, ks, vols, S, tau):
                a,r,v = arv
                error = np.zeros(len(ks))
                for i,k in enumerate(ks):
                    error[i] = np.power(vols[i]-sabr_calculate([a,b,r,v], k, S, tau),2)
                return sum(error)
            loss = lambda arv: fun(arv, b=beta, ks=ks, vols=vols, S=S, tau=tau)
            bounds = Bounds([1e-10,-1,0],[2,1-1e-4, 1000])
            # res = minimize(loss, [0.2,0,0.3], method='SLSQP', bounds=bounds)
            res = minimize(loss, [a0,r0,v0], method='trust-constr', bounds=bounds)
            a,r,v = res.x
            return [a, beta, r, v]
    else:
        return sabr_calibrate_multiple_maturity(ks, vols, S, tau, alpha=alpha, beta=beta, rho=rho, v=v)

def sabr_calibrate_multiple_maturity(ks, vols, S, taus, alpha=None, beta=1, rho=None, v=None):
    ks = np.array(ks, dtype=float)
    vols = np.array(vols, dtype=float)
    taus = np.array(taus, dtype=float)
    if alpha:
        def fun(rv, a, b, ks, vols, S, taus):
            r,v = rv
            return sum(np.power(vols-sabr_iv([a,b,r,v], ks, S, taus),2))
        loss = lambda rv: fun(rv, a=alpha, b=beta, ks=ks, vols=vols, S=S, taus=taus)
        bounds = Bounds([-1,0],[1, 1000])
        res = minimize(loss, [0,1], method='trust-constr', bounds=bounds)
        r,v = res.x
        return [alpha, beta, r, v]
    else:
        # need to fit a,r,v
        def fun(arv, b, ks, vols, S, taus):
            a,r,v = arv
            return sum(np.power(vols-sabr_iv([a,b,r,v], ks, S, taus),2))
        loss = lambda arv: fun(arv, b=beta, ks=ks, vols=vols, S=S, taus=taus)
        bounds = Bounds([1e-10,-1,0],[2,1-1e-4, 1000])
        # res = minimize(loss, [0.2,0,0.3], method='SLSQP', bounds=bounds)
        res = minimize(loss, [0.2,0,0.3], method='trust-constr', bounds=bounds)
        a,r,v = res.x
        return [a, beta, r, v]


def sabr_calibrate_from_df(df, S=None, alpha=None, beta=1, rho=None, v=None):
    F,pos = impForwardLevel(df, df.tau[0])
    if S:
        F = S
    if not 'vol' in df.columns:
        df['vol'] = 0
        for idx,i in enumerate(df.index):
            if idx <= pos:
                df.loc[i,'vol'] = blsImpv(F, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'P'],cpflag='P')
            else:
                df.loc[i,'vol'] = blsImpv(F, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'C'])
    ks = df.K.values
    vols = df.vol.values
    return sabr_calibrate(ks, vols, F, df.tau[0])

# numerical SABR greeks
# Delta, Vega, Vonna, Volga
# 但是如何处理Gamma呢？
# 一些辅组函数
def sabr_z(abrv, K, S):
    K = np.array(K, dtype=float)
    a,b,r,v = abrv
    return np.power(S*K, (1-b)/2) * np.log(S/K) * v / a

def sabr_x(abrv, K, S):
    a,b,r,v = abrv
    z = sabr_z(abrv, K, S)
    return np.log((np.sqrt(1-2*r*z+z*z)+z-r) / (1-r))

def sabr_iv(abrv, K, S, tau):
    # based on Hagan's paper
    a,b,r,v = abrv
    K = np.array(K, dtype=float)
    z = sabr_z(abrv, K, S)
    x = sabr_x(abrv, K, S)
    iv = a*z/x/np.power(S*K,(1-b)/2)
    iv = iv / (1 + np.power(np.log(S/K),2)*np.power(1-b,2)/24+np.power(np.log(S/K),4)*np.power(1-b,4)/1920)
    iv = iv * (1 + (np.power(1-b,2)*np.power(a,2)/np.power(S*K,1-b)/24 + r*b*v*a/np.power(S*K,(1-b)/2)/4 + (2-3*np.power(r,2))*np.power(v,2)/24)*tau)
    return iv

def sabr_z_Obloj(abrv, K, S):
    K = np.array(K, dtype=float)
    a,b,r,v = abrv
    return v*(np.power(S,1-b) - np.power(K,1-b))/(a * (1-b))

def sabr_x_Obloj(abrv, K, S):
    # same as original Hagen's x, except for z's expression
    a,b,r,v = abrv
    z = sabr_z_Obloj(abrv, K, S)
    return np.log((np.sqrt(1-2*r*z+z*z)+z-r) / (1-r))

def sabr_iv_Obloj(abrv, K, S, tau):
    a,b,r,v = abrv
    x = sabr_x(abrv, K, S)
    iv = v * np.log(S/K) / x
    iv = iv / (1 + np.power(1-b,2)/24 * np.power(np.log(S/K),2) + np.power(1-b,4)/1920*np.power(np.log(S/K),4))
    iv = iv * (1 + (np.power(1-b,2)*np.power(a,2)/np.power(S*K,1-b)/24 + r*b*v*a/np.power(S*K,(1-b)/2)/4 + (2-3*np.power(r,2))*np.power(v,2)/24)*tau)
    return iv

def sabr_iv_atm(abrv, K, S, tau):
    a,b,r,v = abrv
    iv_atm = a/np.power(S, 1-b)*(1+(np.power((1-b)*a,2)/24/np.power(S,2-2*b)+a*b*r*v/4/np.power(S,1-b)+np.power(v,2)*(2-3*np.power(r,2))/24)*tau)
    return iv_atm

def sabr_pSigmapS(abrv, K, S, tau, h=0.005):
    dS = S*h
    return (sabr_iv(abrv, K, S+dS, tau) - sabr_iv(abrv, K, S-dS, tau)) / (2*dS)

def sabr_pSigmapa(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    da = a * h
    return (sabr_iv([a+da,b,r,v], K, S, tau) - sabr_iv([a-da,b,r,v], K, S, tau)) / (2*da)

def sabr_pSigmapr(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    dr = r * h
    return (sabr_iv([a,b,r+dr,v], K, S, tau) - sabr_iv([a,b,r-dr,v], K, S, tau)) / (2*dr)

def sabr_pSigmapv(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    dv = v * h
    return (sabr_iv([a,b,r,v+dv], K, S, tau) - sabr_iv([a,b,r,v-dv], K, S, tau)) / (2*dv)

def sabr_pSigmapt(abrv, K, S, tau, h=0.05):
    a,b,r,v = abrv
    dt = tau * h
    return (sabr_iv(abrv, K, S, tau+dt) - sabr_iv(abrv, K, S, tau-dt)) / (2*dt)

def sabr_p2SigmapS2(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    dS = S * h
    return (sabr_pSigmapS(abrv, K, S+dS, tau, h) - sabr_pSigmapS(abrv, K, S+dS, tau, h)) / (2*dS)

def sabr_p2SigmapSpa(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    da = a * h
    return (sabr_pSigmapS([a+da,b,r,v], K, S, tau, h) - sabr_pSigmapS([a-da,b,r,v], K, S, tau, h)) / (2 * da)

def sabr_p2SigmapapS(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    dS = S * h
    return (sabr_pSigmapa(abrv, K, S+dS, tau, h) - sabr_pSigmapa(abrv, K, S-dS, tau, h)) / (2 * dS)

def sabr_p2Sigmapa2(abrv, K, S, tau, h=0.005):
    a,b,r,v = abrv
    da = a * h
    return (sabr_pSigmapa([a+da,b,r,v], K, S, tau, h) - sabr_pSigmapa([a-da,b,r,v], K, S, tau, h)) / (2 * da)

def sabr_delta_Original(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    pSigmapS = sabr_pSigmapS(abrv, K, S, tau, h)
    delta_Original = blsDelta(S, K, rf ,tau, sabr_iv(abrv, K, S, tau)) + blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau)) * pSigmapS
    return delta_Original

def sabr_vega_Original(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    pSigmapa = sabr_pSigmapa(abrv, K, S, tau, h)
    Original_Bartlett = blsVega(S, K, rf, tau, sabr_iv(abrv,K,S,tau)) * pSigmapa
    return Original_Bartlett

def sabr_delta_Bartlett(abrv, K, S, tau, h=0.005, rf=0):
    K = np.array(K, dtype=float)
    a,b,r,v = abrv
    pSigmapS = sabr_pSigmapS(abrv, K, S, tau, h)
    pSigmapa = sabr_pSigmapa(abrv, K, S, tau, h)
    delta_Bartlett = blsDelta(S, K, rf ,tau, sabr_iv(abrv, K, S, tau)) + blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau)) * (pSigmapS + pSigmapa*r*v/np.power(S,b))
    return delta_Bartlett

def sabr_vega_Bartlett(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    pSigmapS = sabr_pSigmapS(abrv, K, S, tau, h)
    pSigmapa = sabr_pSigmapa(abrv, K, S, tau, h)
    vega_Bartlett = blsVega(S, K, rf, tau, sabr_iv(abrv,K,S,tau)) * (pSigmapa + pSigmapS*r*np.power(S,b)/v)
    return vega_Bartlett

def sabr_delta_mod(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    delta = blsDelta(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return  delta + vega*(sabr_pSigmapS(abrv, K, S, tau, h) + sabr_pSigmapa(abrv, K, S, tau, h) * r * v / np.power(S, b))

def sabr_vega_mod(abrv, K, S, tau, h=0.005, rf=0):
    return sabr_vega_Original(abrv, K, S, tau, h, rf)

def sabr_theta(abrv, K, S, tau, h=0.005, rf=0):
    theta = blsTheta(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return theta + vega * sabr_pSigmapt(abrv, K, S, tau, h)

def sabr_gamma(abrv, K, S, tau, h=0.005, rf=0):
    gamma = blsGamma(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return gamma + vega * sabr_p2SigmapS2(abrv, K, S, tau, h)

def sabr_vanna(abrv, K, S, tau, h=0.005, rf=0):
    vanna = blsVanna(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return vanna + vega * sabr_p2SigmapSpa(abrv, K, S, tau, h)

def sabr_volga(abrv, K, S, tau, h=0.005, rf=0):
    volga = blsVolga(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return volga + vega * sabr_p2Sigmapa2(abrv, K, S, tau, h)

def sabr_calibrate_dynamic_piecewise_v(ks, vols, S, taus, alpha=None, beta=1, rho=None, v=None):
    # should return sabr parameters: a,b,r,v
    # currently, assume is coming from {1,0.5,0}, leave out the aesthetic consideration method
    # ks,vols,taus should be vector, S is scalar
    if alpha:
        # need to fit r, v
        n = len(set(taus))
        def fun(rv, a, b, ks, vols, S, taus):
            r,v = rv[0],rv[1:]
            tt = list(set(taus))
            vv = [v[tt.index(tau)] for tau in taus]
            return sum([np.power(vols[i]-sabr_calculate([a,b,r,vv[i]], ks[i], S, taus[i]),2) for i in range(len(ks))])
        loss = lambda rv: fun(rv, a=alpha, b=beta, ks=ks, vols=vols, S=S, taus=taus)
        bounds = Bounds([-1]+[0]*n,[1]+[1000]*n)
        res = minimize(loss, [0]+[0.3]*n, method='trust-constr', bounds=bounds)
        r,v = res.x[0],res.x[1:]
        return [alpha, beta, r, v]
    else:
        # need to fit a,r,v
        n = len(set(taus))
        def fun(arv, b, ks, vols, S, taus):
            a,r,v = arv[0],arv[1],arv[2:]
            tt = list(set(taus))
            vv = [v[tt.index(tau)] for tau in taus]
            return sum([np.power(vols[i]-sabr_calculate([a,b,r,vv[i]], ks[i], S, taus[i]),2) for i in range(len(ks))])
        loss = lambda arv: fun(arv, b=beta, ks=ks, vols=vols, S=S, taus=taus)
        bounds = Bounds([1e-10,-1]+[0]*n,[2,1-1e-4]+[1000]*n)
        # res = minimize(loss, [0.2,0,0.3], method='SLSQP', bounds=bounds)
        res = minimize(loss, [0.2,0]+[0.3]*n, method='trust-constr', bounds=bounds)
        a,r,v = res.x[0],res.x[1],res.x[2:]
        return [a, beta, r, v]

'''
def sabr_theta(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    dt = tau * h
    theta = blsTheta(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega  = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    return theta + vega * sabr_pSigmapt(abrv, K, S, tau, h)

def sabr_vanna(abrv, K, S, tau, h=0.005, rf=0):
    pSigmapr = sabr_pSigmapr(abrv, K, S, tau, h)
    vanna = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau)) * pSigmapr
    return vanna

def sabr_volga(abrv, K, S, tau, h=0.005, rf=0):
    pSigmapv = sabr_pSigmapv(abrv, K, S, tau, h)
    volga = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau)) * pSigmapv
    return volga

def sabr_gamma(abrv, K, S, tau, h=0.005, rf=0):
    dS = S * h
    return (sabr_delta_Bartlett(abrv, K, S+dS, tau)-sabr_delta_Bartlett(abrv, K, S-dS, tau)) / (2*dS)

def sabr_gamma2(abrv, K, S, tau, h=0.005, rf=0):
    a,b,r,v = abrv
    gamma = blsGamma(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vanna = blsVanna(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    vega  = blsVega(S, K, rf, tau, sabr_iv(abrv, K, S, tau))
    pSigmapS = sabr_pSigmapS(abrv, K, S, tau, h)
    psigmapa = sabr_pSigmapa(abrv, K, S, tau, h)
    p2SigmapS2 = (sabr_pSigmapS(abrv, K, S*(1+h), tau, h) - sabr_pSigmapS(abrv, K, S*(1-h), tau, h)) / (2*S*h)
    p2SigmapSpa = (sabr_pSigmapa(abrv, K, S*(1+h), tau, h) - sabr_pSigmapa(abrv, K, S*(1-h), tau, h)) / (2*S*h)
    ans = gamma + vanna*(pSigmapS + pSigmapa*r*v/np.power(S, b))
    ans = ans + vega*(p2SigmapS2 + 2*p2SigmapSpa*r*v/np.power(S,b) - pSigmapa*r*v*b/np.power(S,b+1))
    return ans
'''

#----------Stochastic Alpha Beta Rho (SABR) End  ------------------------------

def otm_impvs(df,S):
    F,pos = impForwardLevel(df, df.tau[0])
    df['vol'] = 0
    df['civ'] = 0
    df['piv'] = 0
    df['otm'] = 0
    for idx,i in enumerate(df.index):
        if idx <= pos:
            df.loc[i,'vol'] = blsImpv(F, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'P'],cpflag='P')
            df.loc[i,'otm'] = blsImpv(S, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'P'],cpflag='P')
        else:
            df.loc[i,'vol'] = blsImpv(F, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'C'])
            df.loc[i,'otm'] = blsImpv(S, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'C'])
        df.loc[i,'civ'] = blsImpv(S, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'C'])
        df.loc[i,'piv'] = blsImpv(S, df.loc[i,'K'], 0, df.tau[0], df.loc[i,'P'],cpflag='P')
    return df

def greekMonitor():
    # used for monitoring current market condition
    # input: current market mid prices (options and underlyings)
    # output: a table of implied vol and other greeks
    # similar as function etf_greeks
    pass

def riskMonitor():
    # used for monitoring current portfolio's risk, including delta,gamma,theta,vega, and so on.
    # input: market prices
    pass




# used for daily VIX and SKEW calculation

def daily_VIX_SKEW_update(tradingDate, underlying = '510300'):
    # tradingDate = '2021-04-07'
    import rqdatac
    rqdatac.init('18616633529','wuzhi2020')
    # download option code list
    options = rqdatac.all_instruments(type='Option', market='cn', date=None)
    trading_codes = []
    symbols = []
    product_names = []
    maturities = []
    for i in options.index:
        if underlying in options.loc[i,'underlying_symbol'] and options.loc[i,'maturity_date'] > tradingDate and options.loc[i,'listed_date'] <= tradingDate:
            trading_codes.append(options.loc[i,'trading_code'])
            symbols.append(options.loc[i,'symbol'])
            maturities.append(options.loc[i,'maturity_date'])
            product_names.append(options.loc[i,'product_name'])
    columns = ['low', 'contract_multiplier', 'strike_price', 'limit_up',
       'total_turnover', 'prev_settlement', 'close', 'open', 'limit_down',
       'volume', 'settlement', 'high', 'open_interest']
    df = pd.DataFrame(index=trading_codes, columns=columns, dtype = float)
    for trading_code in trading_codes:
        daily = rqdatac.get_price(trading_code, start_date=tradingDate, end_date=tradingDate, frequency='1d', adjust_type='none', skip_suspended =False, market='cn', expect_df=False)
        for col in daily.columns:
            df.loc[trading_code,col] = daily[col].values[0]
    df['symbol'] = ''
    df['maturity'] = ''
    df['product_name'] = ''
    for i,trading_code in enumerate(trading_codes):
        df.loc[trading_code,'symbol'] = symbols[i]
        df.loc[trading_code,'maturity'] = maturities[i]
        df.loc[trading_code,'product_name'] = product_names[i]
    # sort and combine
    df['tau'] = 0
    for i,idx in enumerate(df.index):
        df.loc[idx,'tau'] = daysBetween(datetime.datetime.strptime(tradingDate, '%Y-%m-%d'), datetime.datetime.strptime(df['maturity'][i], '%Y-%m-%d'))/240
    # taus = np.unique(df.tau)

    if underlying == '510050':
        RecordDates = ['2016-11-28','2017-11-27','2018-11-30','2019-11-29','2020-11-27'] # for 510050
    elif underlying == '510300':
        RecordDates = ['2021-01-15']
    df['M'] = 0
    for i,index in enumerate(df.index):
        idx = np.where(options.trading_code == str(index))[0][0]
        adjust = False
        for rd in RecordDates:
            if options.listed_date[idx] < rd < options.maturity_date[idx]:
                adjust = True
                break
        if not adjust:
            df.M[i] = 1
        elif tradingDate <= rd:
            df.M[i] = 1

    def selectFront(df):
        # front so select taus[0]
        taus = np.unique(df['tau'])
        idx = np.where(df['tau']==taus[0])[0]
        labels = np.unique([df['strike_price'][i]+float(df['product_name'][i][-5:]) for i in idx])
        Ks =[round(x-50*(x//50),3) for x in labels]
        K0s = [50*(x//50)/1000 for x in labels]
        df1 = pd.DataFrame(index=range(len(labels)),columns=['K','K0','C','P','tau'],dtype=float)
        for i in idx:
            if df['M'][i] == 0:
                continue
            j = np.where(labels==(df['strike_price'][i]+float(df['product_name'][i][-5:])))[0][0]
            df1.loc[j,'K'] = df['strike_price'][i]
            df1.loc[j,'K0'] = float(df['product_name'][i][-5:])/1000
            df1.loc[j,'tau'] = df['tau'][i]
            # choose settlement or close price?
            # for me, settlement seems more reasonable
            if 'C' in df['product_name'][i]:
                # df1.loc[j,'C'] = df['settlement'][i]
                df1.loc[j,'C'] = df['close'][i]
            if 'P'in df['product_name'][i]:
                # df1.loc[j,'P'] = df['settlement'][i]
                df1.loc[j,'P'] = df['close'][i]
        df1 = df1.dropna()
        df1.index = range(len(df1))
        return df1


    def selectSecond(df):
        # second so select taus[1]
        taus = np.unique(df['tau'])
        idx = np.where(df['tau']==taus[1])[0]
        labels = np.unique([df['strike_price'][i]+float(df['product_name'][i][-5:]) for i in idx])
        Ks =[round(x-50*(x//50),3) for x in labels]
        K0s = [50*(x//50)/1000 for x in labels]
        df2 = pd.DataFrame(index=range(len(labels)),columns=['K','K0','C','P','tau'],dtype=float)
        for i in idx:
            if df['M'][i] == 0:
                continue
            j = np.where(labels==(df['strike_price'][i]+float(df['product_name'][i][-5:])))[0][0]
            df2.loc[j,'K'] = df['strike_price'][i]
            df2.loc[j,'K0'] = float(df['product_name'][i][-5:])/1000
            df2.loc[j,'tau'] = df['tau'][i]
            # choose settlement or close price?
            # for me, settlement seems more reasonable
            if 'C' in df['product_name'][i]:
                # df2.loc[j,'C'] = df['settlement'][i]
                df2.loc[j,'C'] = df['close'][i]
            if 'P'in df['product_name'][i]:
                # df2.loc[j,'P'] = df['settlement'][i]
                df2.loc[j,'P'] = df['close'][i]
        df2 = df2.dropna()
        df2.index = range(len(df2))
        return df2
    df1 = selectFront(df)
    df2 = selectSecond(df)
    # calculate spot-forward difference
    F1,_ = impForwardLevel(df1, df1['tau'][0])
    F2,_ = impForwardLevel(df2, df2['tau'][0])
    if underlying == '510300':
        und = rqdatac.get_price(underlying+'.XSHG', start_date=tradingDate, end_date=tradingDate, frequency='1d', adjust_type='none', skip_suspended =False, market='cn', expect_df=False)
        S = und['close'].values[0]
    elif underlying == '510050':
        und = rqdatac.get_price(underlying+'.XSHG', start_date=tradingDate, end_date=tradingDate, frequency='1d', adjust_type='none', skip_suspended =False, market='cn', expect_df=False)
        S = und['close'].values[0]
    dFS1 = F1-S
    dFS2 = F2-S
    # VIX
    vix = impVIXTwoSide(df1, df2, df1['tau'][0], df2['tau'][0])
    # SKEW
    skew = impSkewTwoSide(df1, df2, df1['tau'][0], df2['tau'][0])
    del rqdatac
    return F1,vix,skew


#-------------------Front month option informations---------------------------
def front_month_table(tradingDate, underlying = '510300'):
    # tradingDate = '2021-04-07'
    import rqdatac
    rqdatac.init('18616633529','wuzhi2020')
    # download option code list
    options = rqdatac.all_instruments(type='Option', market='cn', date=None)
    trading_codes = []
    symbols = []
    product_names = []
    maturities = []
    for i in options.index:
        if underlying in options.loc[i,'underlying_symbol'] and options.loc[i,'maturity_date'] > tradingDate and options.loc[i,'listed_date'] <= tradingDate:
            trading_codes.append(options.loc[i,'trading_code'])
            symbols.append(options.loc[i,'symbol'])
            maturities.append(options.loc[i,'maturity_date'])
            product_names.append(options.loc[i,'product_name'])
    columns = ['low', 'contract_multiplier', 'strike_price', 'limit_up',
       'total_turnover', 'prev_settlement', 'close', 'open', 'limit_down',
       'volume', 'settlement', 'high', 'open_interest']
    df = pd.DataFrame(index=trading_codes, columns=columns)
    for trading_code in trading_codes:
        daily = rqdatac.get_price(trading_code, start_date=tradingDate, end_date=tradingDate, frequency='1d', adjust_type='none', skip_suspended =False, market='cn', expect_df=False)
        for col in daily.columns:
            df.loc[trading_code,col] = daily[col].values[0]
    df['symbol'] = ''
    df['maturity'] = ''
    df['product_name'] = ''
    for i,trading_code in enumerate(trading_codes):
        df.loc[trading_code,'symbol'] = symbols[i]
        df.loc[trading_code,'maturity'] = maturities[i]
        df.loc[trading_code,'product_name'] = product_names[i]
    # sort and combine
    df['tau'] = 0
    for i,idx in enumerate(df.index):
        df.loc[idx,'tau'] = daysBetween(datetime.datetime.strptime(tradingDate, '%Y-%m-%d'), datetime.datetime.strptime(df['maturity'][i], '%Y-%m-%d'))/240
    # taus = np.unique(df.tau)
    def selectFront(df):
        # front so select taus[0]
        taus = np.unique(df['tau'])
        idx = np.where(df['tau']==taus[0])[0]
        labels = np.unique([df['strike_price'][i]+float(df['product_name'][i][-5:]) for i in idx])
        Ks =[round(x-50*(x//50),3) for x in labels]
        K0s = [50*(x//50)/1000 for x in labels]
        df1 = pd.DataFrame(index=range(len(labels)),columns=['K','K0','C','P','tau'])
        for i in idx:
            j = np.where(labels==(df['strike_price'][i]+float(df['product_name'][i][-5:])))[0][0]
            df1.loc[j,'K'] = df['strike_price'][i]
            df1.loc[j,'K0'] = float(df['product_name'][i][-5:])/1000
            df1.loc[j,'tau'] = df['tau'][i]
            # choose settlement or close price?
            # for me, settlement seems more reasonable
            if 'C' in df['product_name'][i]:
                # df1.loc[j,'C'] = df['settlement'][i]
                df1.loc[j,'C'] = df['close'][i]
            if 'P'in df['product_name'][i]:
                # df1.loc[j,'P'] = df['settlement'][i]
                df1.loc[j,'P'] = df['close'][i]
        return df1
    df1 = selectFront(df)
    F,m = impForwardLevel(df1, df1.tau[0])
    if underlying == '510300':
        und = rqdatac.get_price(underlying+'.XSHG', start_date=tradingDate, end_date=tradingDate, frequency='1d', adjust_type='none', skip_suspended =False, market='cn', expect_df=False)
        S = und['close'].values[0]
    df1['CIV'] = 0
    df1['PIV'] = 0
    df1['IV'] = 0
    df1['DeltaC'] = 0
    df1['DeltaP'] = 0
    df1['Vega'] = 0
    df1['Theta5'] = 0   # measured in $
    df1['Gamma'] = 0
    # df1['Vanna'] = 0
    # df1['Volga'] = 0
    tau = df1.tau[0]
    r = 0
    for idx in df1.index:
        K = df1.loc[idx,'K']
        C = df1.loc[idx,'C']
        P = df1.loc[idx,'P']
        df1.loc[idx,'CIV'] = blsImpv(S, K, r, tau, C, cpflag='C')
        df1.loc[idx,'PIV'] = blsImpv(S, K, r, tau, P, cpflag='P')
        if idx <= m:
            df1.loc[idx,'IV'] = blsImpv(F, K, r, tau, P, cpflag='P')
        else:
            df1.loc[idx,'IV'] = blsImpv(F, K, r, tau, C, cpflag='C')
        df1.loc[idx,'DeltaC'] = blsDelta(F, K, r, tau, df1.loc[idx,'IV'], cpflag='C')
        df1.loc[idx,'DeltaP'] = blsDelta(F, K, r, tau, df1.loc[idx,'IV'], cpflag='P')
        df1.loc[idx,'Vega']   = blsVega(F, K, r, tau, df1.loc[idx,'IV'])
        df1.loc[idx,'Gamma']  = blsGamma(F, K, r, tau, df1.loc[idx,'IV'])
        df1.loc[idx,'Theta5']= blsTheta5(F, K, r, tau, df1.loc[idx,'IV'])
    return df1


#---------- Heston Stochastic Volatility Model Start---------------------------
def heston_alpha(i, u):
    # i == 0 or 1
    return -np.power(u,2)/2-1j*u/2+i*1j*u

def heston_beta(i, u, param):
    lmbd,v0,eta,rho = param
    # i == 0 or 1
    return lmbd-rho*eta*i-rho*eta*1j*u

def heston_d(i,u, param):
    lmbd,v0,eta,rho = param
    alpha = heston_alpha(i, u)
    beta = heston_beta(i, u, param)
    gamma = np.power(eta,2)/2
    return np.sqrt(np.power(beta,2)-4*alpha*gamma)

def heston_r(i, u, param):
    lmbd,v0,eta,rho = param
    alpha = heston_alpha(i, u)
    beta = heston_beta(i, u, param)
    gamma = np.power(eta,2)/2
    rp = (beta + np.sqrt(np.power(beta,2)-4*alpha*gamma))/(2*gamma)
    rm = (beta - np.sqrt(np.power(beta,2)-4*alpha*gamma))/(2*gamma)
    return rp,rm

def heston_g(i, u, param):
    rp,rm = heston_r(i, u, param)
    return rm/rp

def heston_C(i, u, tau, param):
    # i = 0 or 1
    lmbd,v0,eta,rho = param
    rp,rm = heston_r(i, u, param)
    d = heston_d(i, u, param)
    g = heston_g(i, u, param)
    return lmbd * (rm*tau - 2/np.power(eta,2) * np.log((1-g*np.exp(-d*tau))/(1-g)))

def heston_D(i, u, tau, param):
    # i = 0 or 1
    lmbd,v0,eta,rho = param
    rp,rm = heston_r(i, u, param)
    d = heston_d(i, u, param)
    g = heston_g(i, u, param)
    return rm*(1-np.exp(-d*tau))/(1-g*np.exp(-d*tau))

def heston_P(i, x, v, tau, param, du = 2**-4, ulim = 2**10):
    # should add partition and tolerance control
    lmbd,v0,eta,rho = param
    # numerical integration
    us = np.arange(du/2, ulim, du)
    # compute integrands
    Cs = heston_C(i, us, tau, param)
    Ds = heston_D(i, us, tau, param)
    return 0.5 + np.real(np.sum(np.exp(Cs*v0+Ds*v+1j*us*x)/(1j*us)*du))/np.pi

def heston_price(S, K, r, tau, params, q=0, cpflag='C'):
    F = S*np.exp((r-q)*tau)
    x = np.log(F/K)
    v,lmbd,v0,eta,rho = params
    param = [lmbd,v0,eta,rho]
    return F*heston_P(1,x,v,tau,param)-K*heston_P(0,x,v,tau,param)

def heston_calibrate(df):
    # df should contain the following informatin:
    # 1. S
    # 2. Ks
    # 3. tau
    # 4. Call option prices
    # 4. Put option prices
    def fun(params, S, Ks, r, tau, Vs):
        error = np.zeros(len(Ks))
        for i,k in enumerate(Ks):
            error[i] = np.power(Vs[i]-heston_price(S, K, r, tau, params),2)
        return sum(error)
    F,_ = impForwardLevel(df,df.tau[0])
    loss = lambda params: fun(params, S=F, Ks=df.K, r=0, tau=df.tau[0], Vs=df.C)
    bounds = Bounds([1e-10,1e-10,1e-3,0,-0.99],[2,2,100,10,0.99])
    # res = minimize(loss, [0.2,0,0.3], method='SLSQP', bounds=bounds)
    params0 = [0.10,0.10,10,1,-0.5]
    res = minimize(loss, params0, method='trust-constr', bounds=bounds)
    a,r,v = res.x
    return [a, beta, r, v]
#---------- Heston Stochastic Volatility Model End ----------------------------









#------------Some utility function for extracting data------------------------
def get_options_by_date(date, underlying):
    return pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+date+'.csv')

def get_option_by_contract(Id, underlying):
    return pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+Id+'.csv')

def get_front_options(date, underlying, major=True):
    # major means excluding adjusted options
    df = pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+date+'.csv')
    df['month'] = [x[7:11] for x in df['name']]
    months = np.unique(df['month'])
    front = df.loc[df['month']==months[0]]
    # 判断是否存在修改过的期权
    if not major:
        return front
    else:
        return front.loc[front.M==1]

def get_second_options(date, underlying, major=True):
    # major means excluding adjusted options
    df = pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+date+'.csv')
    df['month'] = [x[7:11] for x in df['name']]
    months = np.unique(df['month'])
    second = df.loc[df['month']==months[1]]
    # 判断是否存在修改过的期权
    if not major:
        return second
    else:
        return second.loc[second.M==1]

def get_third_options(date, underlying, major=True):
    # major means excluding adjusted options
    df = pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+date+'.csv')
    df['month'] = [x[7:11] for x in df['name']]
    months = np.unique(df['month'])
    third = df.loc[df['month']==months[2]]
    # 判断是否存在修改过的期权
    if not major:
        return third
    else:
        return third.loc[third.M==1]

def get_fourth_options(date, underlying, major=True):
    # major means excluding adjusted options
    df = pd.read_csv('F:/Kai/Data/Options/ByDate/'+underlying+'/'+date+'.csv')
    df['month'] = [x[7:11] for x in df['name']]
    months = np.unique(df['month'])
    fourth = df.loc[df['month']==months[3]]
    # 判断是否存在修改过的期权
    if not major:
        return fourth
    else:
        return fourth.loc[fourth.M==1]

def convert_to_brief_table(df):
    # Correcting Strike Price
    for idx in df.index:
        df.loc[idx,'strike_price'] = float(df.loc[idx, 'name'][12:])/1000

    K = np.unique(df.strike_price)
    tmp = pd.DataFrame(index=range(len(K)),columns=['K','C','P'])
    tmp['K'] = K
    today = datetime.datetime.strptime(df.date.values[0],'%Y-%m-%d')
    maturity = fourthWednesday(int(df.month.values[0][:2])+2000, int(df.month.values[0][2:]))
    tmp['tau'] = daysBetween(today,maturity)/240
    for idx in df.index:
        tmp.loc[K == df.loc[idx, 'strike_price'], df.loc[idx,'name'][6]] = df.loc[idx,'close']
        # print(tmp)
    return tmp

def simple_process(df):
    df['CIV'] = 0
    df['PIV'] = 0
    df['IV']  = 0
    F,i = impForwardLevel(df, df.tau.values[0])
    for idx in df.index:
        df.loc[idx, 'CIV'] = blsImpv(F, df.loc[idx, 'K'], 0, df.loc[idx, 'tau'], df.loc[idx, 'C'], cpflag='C')
        df.loc[idx, 'PIV'] = blsImpv(F, df.loc[idx, 'K'], 0, df.loc[idx, 'tau'], df.loc[idx, 'P'], cpflag='P')
        if df.loc[idx, 'K'] < F:
            df.loc[idx, 'IV'] = df.loc[idx, 'PIV']
        else:
            df.loc[idx, 'IV'] = df.loc[idx, 'CIV']
    return df


#%% 从中金场外报价估计隐含波动率和分红率
# def implied_iv_dividend_atm(c, p, tau):
#     S = 1
#     K = 1
#     r = 0
#     def loss(x):
#         iv,d = x[0],x[1]
#         return np.power(wwo.blsPrice(S,K,d,tau,iv)-c,2)+np.power(wwo.blsPrice(S,K,d,tau,iv,cpflag='P')-p,2)
#     x0 = [0.2,0]
#     res = minimize(loss, [0.2,0], method='trust-constr')
#     iv,d = res.x
#     return iv,d

#%% 20230427
# 策略归因分析
def _attribution_anaysis_daily(prev_pos, trades, prices):
    '''
    prev_pos:       昨仓
    trades:         今日的交易记录
    prices:             各品种最新价
    '''



    return



















