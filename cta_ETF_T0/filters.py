# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:41:54 2023

@author: wzer
"""

import numpy as np
import scipy as sp
import cvxpy as cp
import cvxopt as cvxopt

import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

import statsmodels.api as sm
# assume inputs are 1-dim numpy array or array-like object

import numpy as np
#%% (1) Linear filters
# Moving average filter
def moving_average(y, n=100):
    L = np.ones(n) / n
    x = np.convolve(y, L)[:len(y)]
    for i in range(len(y)):
        if i < n:
            x[i] = np.mean(y[:i+1])
    return x

def symmetric(y, n=100):
    L = 4/n**2*(n/2-np.abs(np.arange(n)-n/2))
    x = np.convolve(y, L)[:len(y)]
    return x

def asymmetric(y, n=100, m=20):
    L1 = 2*np.arange(m)/n/m
    L2 = 2*m/n *(n-np.arange(m,n)) / (n-m) / m
    L = np.hstack((L1,L2))
    x = np.convolve(y, L)[:len(y)]
    return x

def triangle(y, n=100):
    # triangle
    L = 2/ n**2 *(n-np.arange(n))
    x = np.convolve(y, L)[:len(y)]
    return x

def Lanczos(y, n=100):
    L = 6/n**3*np.arange(n)*(n-np.arange(n))
    x = np.convolve(y, L)[:len(y)]
    return x

def L2(y, lamb=100):
    cycle, trend = sm.tsa.filters.hpfilter(y, lamb)
    return trend

def L2_optimal_lambda(y, T=10):
    # Reference: A.4 in Trading Strategies with L1 Filtering 2011
    # T: investment horizon, 10 days
    l_star = 0.5*np.power(T/(2*np.pi), -4)
    return L2(y, 10.27*l_star)

def Kalman():
    pass


# L = np.ones(n) / n
# plt.plot(L, label='sma')
# L = 4/n**2*(n/2-np.abs(np.arange(n)-n/2))
# plt.plot(L, label='sym')
# L = 2/ n**2 *(n-np.arange(n))
# plt.plot(L, label='triangle')
# L1 = 2*np.arange(m)/n/m
# L2 = 2*m/n *(n-np.arange(m,n)) / (n-m) / m
# L = np.hstack((L1,L2))
# plt.plot(L, label='asym')
# L = 6/n**3*np.arange(n)*(n-np.arange(n))
# plt.plot(L, label='Lanczos')
# plt.grid()
# plt.legend()


#%% (2) Nonlinear filters
# def lowess(y):
#     LW = sm.nonparametric.lowess
#     z = LW(y, np.arange(len(y)))

def loess(y):
    pass

def kernel_regression(y):
    # statsmodels.nonparametric.kernel_regression.KernelReg
    pass

def spline_regression(y):
    pass

def Lp(y, lamb=100):
    pass

def L1_T(y, vlambda=100, verbose=False):
    n = len(y)
    e = np.ones((1, n))
    D = sp.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                      + vlambda * cp.norm(D@x, 1) )
    prob = cp.Problem(obj)
    # prob.solve(solver=cp.CVXOPT, verbose=True)
    prob.solve(solver=cp.CVXOPT, verbose=verbose)
    print('Solver status: {}'.format(prob.status))
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    print("optimal objective value: {}".format(obj.value))
    return x.value

def L1_C(y, vlambda=100):
    n = len(y)
    e = np.ones((1, n))
    D = sp.sparse.spdiags(np.vstack((-e, e)), range(2), n-2, n)
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                      + vlambda * cp.norm(D@x, 1) )
    prob = cp.Problem(obj)
    prob.solve(solver=cp.CVXOPT, verbose=True)
    print('Solver status: {}'.format(prob.status))
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    print("optimal objective value: {}".format(obj.value))
    return x.value

def L1_TC(y, vlambda1=1, vlambda2=2):
    n = len(y)
    e = np.ones((1, n))
    D1 = sp.sparse.spdiags(np.vstack((-e, e)), range(2), n-2, n)
    D2 = sp.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                      + vlambda1 * cp.norm(D1@x, 1)
                      + vlambda2 * cp.norm(D2@x, 1))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.CVXOPT, verbose=True)
    print('Solver status: {}'.format(prob.status))
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    print("optimal objective value: {}".format(obj.value))
    return x.value

def D_L1(n):
    e = np.ones((1, n))
    D = sp.sparse.spdiags(np.vstack((-e, e)), range(2), n-1, n)
    return D

def get_lambda_max(y):
    n = len(y)
    D = D_L1(n)
    return sp.linalg.norm(sp.sparse.linalg.spsolve(D@D.T, D@y), np.inf)

def cv_L1_T(y, T1=500, T2=50, n=10):
    # 生成sample pairs (train,test)
    stops = np.arange(len(y)-T2, T1, -T2)[::-1]
    # compute lambda_max for each T2
    lms = np.zeros(len(stops))
    for i, stop in enumerate(stops):
        _y = y[stop:stop+T2]
        lms[i] = get_lambda_max(_y)
    lm_m = np.mean(lms)
    lm_s = np.std(lms)
    lm1, lm2 = max(lm_m - 2*lm_s,lm_m/100), lm_m + 2*lm_s     # 文中做法无法保证lm1>0
    lms = [lm1*np.power(lm2/lm1, j/n) for j in range(n+1)]
    # m = np.mean(np.log(lms))
    # s = np.std(np.log(lms))
    # lms = np.exp(np.arange(m-1*s, m+1*s, s/n))
    errors = np.zeros(n+1)  # forecast errors
    for j in range(n+1):
        lm = lms[j]
        for i,stop in enumerate(stops):
            # print(j,i, lm)
            _y_train = y[stop-T1:stop]
            _y_test = y[stop:stop+T2]
            x = L1_T(_y_train, lm)
            mu = np.diff(x)[-1]
            _y_pred = _y_train[-1] + np.arange(1,T2+1)*mu
            errors[j] += np.sum(np.square(_y_pred - _y_test))
        # print(errors)
    # plt.plot(lms, errors,'.-')
    return lms[np.argmin(lms)]

def two_trend_prediction(y, T1=500, T2=50, T3=10):
    lmG = cv_L1_T(y, T1, T2)
    lmL = cv_L1_T(y, T1, T3)
    xG = L1_T(y, lmG)
    xL = L1_T(y, lmL)
    if np.abs(y[-1] - xG[-1]) < np.std(y-xG):
        return xL
    else:
        return xG

def wavelet(y):
    pass

def singular_spectrum(y):
    # Vautard 1992
    pass

def svm(y):
    pass

def empirical_mode_decomposition(y):
    # Flandrin 2004
    pass

def Ehlers(y):
    # Ehlers 2001
    # import ehlers
    pass

#%% (3) multivariate filtering
# assuming y is multivariate time series
# common trend
# (3.1) mean of individual trend
def L2_multivariate(y, lamb=100):
    _y = np.mean(y)
    cycle, trend = sm.tsa.filters.hpfilter(_y, lamb)
    return trend

# (3.2) ECM
# (3.3) PT permanent-transitory decomposition (y=P+T)
# (3.4) common stochastic trend model
# essentially, use Kalman



#%% (4) Calibration
# (4.1) MLE of LLK
# (4.2) cross-validation CV
# (4.3) based on benchmark estimator

#%% (5) Variance
# Measure efficiency by MSE (MSE = Bias + Variance)

# (5.2) Trend Detection
# Mann 1945
def mann(y, n=20):
    '''


    Parameters
    ----------
    y : TYPE
        time series.
    n : TYPE, optional
        window length. The default is 20.

    Returns
    -------
    None.

    '''
    T = len(y)
    S = np.zeros(T) # time series of S
    # SgnMtx =
    for t in range(n, T):
        # from t-n to t
        s = 0
        for i in np.arange(n-1):
            s += np.sum(np.sign(y[t-i]-y[t-n:t-i]))
        S[t] = s
    # normalized S -> SS \in [-1,1]
    # SS>0, positive trend
    # SS<0, negative trend
    # SS = 2*S/(n*(n+1))
    # Z-score
    Z = S / np.std(S[n:])
    # 加点别的判断？ Z ~ N(0,1) as n-> inf

    return S, Z

#%% (6) Forecasting
# (6.1) Naive classifier
# upper trend -> positive conditional return
# vice versa

#%% (7) signal -> weighting
# use normalized signal
def simple_sigmoid(z, l=1):
    # l is the scaling factor, approximately maximum/minimum position
    # 需要能更灵活的调整中间的斜率，加一个什么参数?
    c_l = 1/np.sqrt(2/np.pi*np.arctan(l**2/np.sqrt(1+2*l**2)))
    return c_l*(2*sp.stats.norm.cdf(l*z) - 1)

def reverting_sigmoid(z, l=1):
    c_l = np.power(1 + 2*l**2, 0.75)
    return c_l*z*np.exp(-l**2*z**2/2)

def double_step(z, eps=1):
    c_l = 1/np.sqrt(2*sp.stats.norm.cdf(-eps)) / 1.775242853114575
    return c_l * (1*(z>eps)-1*(z<-eps))




# #%% 读取510050试一下
# y = pd.read_csv('F:/Kai/pythonScripts/510050.XSHG.csv')['close'].values
# plt.figure()
# plt.plot(y, label='original')
# plt.plot(moving_average(y), label='MA')
# plt.plot(symmetric(y), label='sym')
# plt.plot(asymmetric(y), label='asym')
# plt.plot(triangle(y), label='triangle')
# plt.plot(Lanczos(y), label='Lanczos')
# plt.plot(L2(y), label='L2')
# plt.grid()
# plt.legend()

# #%% L1 filtering and strategy
# df = pd.read_csv('F:/Kai/pythonScripts/510050.XSHG.csv')
# ST = 5
# LT = 20
# T1 = 120
# last_update = 0
# weights = np.zeros(len(df)) # weight on equity
# wealth  = np.ones(len(df))
# a_min, a_max = -1, 1
# penalty = 1
# lmG,lmL = 0.2, 0.001
# start = 30*T1
# for i in range(start, len(df)-1):
#     _y = df['close'].values[i-2*T1:i]
#     # 间隔20天重新估计一次lambda
#     if i - last_update > 100:
#         lmG = cv_L1_T(_y, T1, LT)   # long term global trend
#         lmL = cv_L1_T(_y, T1, ST)   # short term trend
#         last_update = i
#         print(f'----------------------------------------------{i}------------------------------')
#     _xG = L1_T(_y, lmG)
#     _xL = L1_T(_y, lmL)
#     if np.abs(_y[-1] - _xG[-1]) < np.std(_y-_xG):
#         _x = _xL
#     else:
#         _x = _xG
#     _mu = _x[-1] - _x[-2]
#     _std = np.std(np.diff(np.log(_y)))
#     a_star = _mu/(_std**2*penalty*wealth[i-1])
#     weights[i] = a_min*(a_star<a_min) + a_star*((a_star>=a_min)*(a_star<=a_max)) + a_max*(a_star>a_max)
#     wealth[i+1] = wealth[i] +  wealth[i] * weights[i] * (df['close'].values[i]/_y[-1]-1)
#     print(i)

# plt.figure()
# # plt.subplot(2,1,1)
# plt.plot(df['close'].values[start:-1]/df['close'].values[start],label='underlying')
# # plt.subplot(2,1,2)
# plt.plot(wealth[start:-1], label='strategy')

# # 估计是有用的不过好慢啊 14：00
# # 感觉也没啥用啊，还很慢 16：50


# #%%
# ST = 5
# LT = 20
# T1 = 240
# df = pd.read_csv('F:/Kai/pythonScripts/510500.XSHG.csv')
# df['ST_trend'] = np.nan
# df['LT_trend'] = np.nan
# df['ret'] = df['close'].apply(np.log).diff().fillna(0)
# for i in range(T1, len(df)):
#     # df.loc[i,'ST_trend'] = Lanczos(df.loc[i-LT:i,'ret'], ST)[-1]
#     # df.loc[i,'LT_trend'] = Lanczos(df.loc[i-LT:i,'ret'], LT)[-1]
#     df.loc[i,'ST_trend'] = np.mean(np.diff(L2_optimal_lambda(df.loc[i-T1:i,'close'].values, ST))[-10:])
#     df.loc[i,'LT_trend'] = np.mean(np.diff(L2_optimal_lambda(df.loc[i-T1:i,'close'].values, LT))[-10:])
# #%%
# df['fret'] = df['close'].apply(np.log).diff().shift(-1)
# # 信号
# # MAE
# df['ST_signal'] = df['ST_trend'] / df['ST_trend'].abs().ewm(span=LT).mean()
# df['LT_signal'] = df['LT_trend'] / df['ST_trend'].abs().ewm(span=LT).mean()
# # df['ST_signal'] = df['ST_trend'] / df['ST_trend'].rolling(T1).std()
# # df['LT_signal'] = df['LT_trend'] / df['ST_trend'].rolling(T1).std()

# # 仓位
# # df['ST_pos'] = df['ST_signal'].apply(np.sign) * (df['ST_signal'].abs() > 1.0)
# # df['LT_pos'] = df['LT_signal'].apply(np.sign) * (df['LT_signal'].abs() > 1.0)
# # sigmoid
# # df['ST_pos'] = simple_sigmoid(df['ST_signal'])
# # df['LT_pos'] = simple_sigmoid(df['LT_signal'])
# # reverting不太行
# # df['ST_pos'] = reverting_sigmoid(df['ST_signal'])
# # df['LT_pos'] = reverting_sigmoid(df['LT_signal'])
# # double step
# df['ST_pos'] = double_step(df['ST_signal'],1.5)
# df['LT_pos'] = double_step(df['LT_signal'],1.5)



# df['ST_ret'] = (df['ST_pos'] * df['fret']).fillna(0)
# df['LT_ret'] = (df['LT_pos'] * df['fret']).fillna(0)
# df['same_ret'] = (df['ST_pos'] * df['fret']).fillna(0) * (df['ST_pos']==df['LT_pos'])

# plt.figure()
# plt.subplot(1,1,1)
# plt.plot(pd.to_datetime(df['date']), df['ST_ret'].fillna(0).cumsum(), label='ST')
# plt.plot(pd.to_datetime(df['date']), df['LT_ret'].fillna(0).cumsum(), label='LT')
# # plt.plot(df['same_ret'].fillna(0).cumsum(), label='same')   # 长短周期同向
# plt.plot(pd.to_datetime(df['date']), np.log(df['close']/df['close'].values[0]), '-.', label='long', alpha=0.5)
# plt.legend()
# plt.grid()
# plt.ylabel('cumulative return')
# plt.ylim([-1,3])
# # plt.subplot(2,1,2)
# # plt.plot(df['ST_ret'].fillna(0).cumsum()-np.log(df['close']/df['close'].values[0]), label='ST')
# # plt.plot(df['LT_ret'].fillna(0).cumsum()-np.log(df['close']/df['close'].values[0]), label='LT')
# # # plt.plot(df['same_ret'].fillna(0).cumsum()-np.log(df['close']/df['close'].values[0]), label='same')
# # plt.legend()
# # plt.grid()
# # plt.ylabel('execss return')

