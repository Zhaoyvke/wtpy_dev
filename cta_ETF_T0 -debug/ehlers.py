# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 15:53:43 2023

@author: Kai
"""

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
import os
import datetime
import matplotlib.pyplot as plt

#%% auxilary functions
def shift(p, n=1):
    if n == 0:
        return p
    elif n > 0:
        return np.hstack((np.zeros(n),p[:-n]))
    else:
        return np.hstack((p[-n:],np.zeros(np.abs(n))))

def lowest(s, n=10):
    # running window lowest for series s
    lows = np.zeros(len(s))
    for i in range(len(s)):
        lows[i] = np.min(s[max(i-n,0):i+1])
    return lows

def highest(s, n=10):
    # running window highest
    highs = np.zeros(len(s))
    for i in range(len(s)):
        highs[i] = np.max(s[max(i-n,0):i+1])
    return highs

def waverage(s, n=10):
    # triangle average
    w = (1 + np.arange(n)) / sum(1 + np.arange(n))
    return np.convolve(s, w)[:-(n-1)]

def wsmooth(s, n=4):
    oddEven = n%2
    peak    = (n//2)*(1-oddEven) + ((n+1)//2)*oddEven
    w       = np.hstack((np.arange(1,peak+1-oddEven),np.arange(peak,0,-1)))
    return np.convolve(s, w)[:-(n-1)]

#%% indicators
def fisher_transform(p, N=10):
    # p = (h+l) / 2
    maxH    = np.zeros(len(p))
    minL    = np.zeros(len(p))
    value   = np.zeros(len(p))
    fisher  = np.zeros(len(p))
    for i in range(1, len(p)):
        maxH[i]     = max(p[max(0,i-N):i+1])
        minL[i]     = min(p[max(0,i-N):i+1])
        value[i]    = max(min(((p[i]-minL[i])/(maxH[i]-minL[i])-0.5) + 0.5*value[i-1],1-1e-8),1e-8-1)
        fisher[i]   = 0.25*np.log((1+value[i])/(1-value[i])) + 0.5*fisher[i-1]
    return fisher

def i_trend(p, a=0.07):
    # smooth  = np.zeros(len(p))
    trend  = np.zeros(len(p))
    trigger = np.zeros(len(p))
    aa = a**2
    for i in range(2, len(p)):
        if i < 2:
            trend[i] = p[i]
        elif i < 7:
            trend[i] = (p[i] + 2*p[i-1] + p[i-2])/4
        else:
            trend[i] += (a-aa/4)*p[i] + 0.5*aa*p[i-1] -(a-0.75*aa)*p[i-2]
            trend[i] += 2*(1-a)*trend[i-1] - (1-a)**2*trend[i-2]
        trigger[i] = 2*trend[i] - trend[i-2]
    return trend, trigger

def i_cycle(p, a=0.07):
    # oscillator
    smooth  = np.zeros(len(p))
    cycle   = np.zeros(len(p))
    trigger = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 2:
            cycle[i] = 0
        elif i < 7:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
        else:
            smooth[i] = (p[i]+2*p[i-1]+2*p[i-2]+p[i-3])/6
            cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
            cycle[i] += 2*(1-a)*cycle[i-1] - (1-a)**2*cycle[i-2]
    trigger[1:] = cycle[:-1]
    return cycle, trigger

def cg_oscillator(p, N=10):
    # center of gravity oscillator
    num     = np.zeros(len(p))
    denom   = np.zeros(len(p))
    cg      = np.zeros(len(p))
    for count in range(N):
        for i in range(count, len(p)):
            num[i]      += (1+count)*p[i-count]
            denom[i]    += p[i-count]
    idx = np.where(denom!=0)
    cg[idx] = -num[idx]/denom[idx] + (N+1)/2
    cg[:N] = 0
    return cg

def relative_vigor(o, h, l, c, N=10):
    # relative vigor index oscillator
    rvi     = np.zeros(len(c))
    value1  = c-o + 2*shift(c-o,1) + 2*shift(c-o,2) + shift(c-o,3)
    value2  = h-l + 2*shift(h-l,1) + 2*shift(h-l,2) + shift(h-l,3)
    num     = value1
    denom   = value2
    for count in range(1, N):
        num += shift(value1, count)
        denom += shift(value2, count)
    idx = np.where(denom!=0)
    rvi[idx] = num[idx]/denom[idx]
    trigger  = shift(rvi, 1)
    return rvi, trigger

def rsi(p, N=10):
    # relative strength index
    change  = np.hstack((0,np.diff(p)))
    rs      = np.zeros(len(p))
    for i in range(N,len(p)):
        ccc     = change[i-N:i+1]
        if len(ccc[ccc<0]) == 0:
            # all positive
            rs[i] = np.inf
        elif len(ccc[ccc>0]) == 0:
            # all negative
            rs[i] = 0
        else:
            up    = np.mean(ccc[ccc>0])
            down  = -np.mean(ccc[ccc<0])
            rs[i] = up / down
    rsi     = 100 - 100 / (1 + rs)
    return rsi

def stochastic_rsi(c, N_rsi=8, N_stoc=8, N_wma=8):
    value1 = rsi(c, N_rsi) - lowest(rsi(c, N_rsi), N_stoc)
    value2 = highest(rsi(c, N_rsi), N_stoc) - lowest(rsi(c, N_rsi), N_stoc)
    value3 = np.zeros(len(c))
    idx = np.where(value2!=0)
    value3[idx] = value1[idx] / value2[idx]
    stocRsi = 2*(waverage(value3, N_wma)-0.5)
    trigger = shift(stocRsi, 1)
    return stocRsi, trigger

def stochastic_rvi(o, h, l, c, N=8):
    rvi,_ = relative_vigor(o, h, l, c, N)
    maxRvi  = highest(rvi, N)
    minRvi  = lowest(rvi, N)
    value3  = (rvi-minRvi) / (maxRvi-minRvi)
    value4  = waverage(value3, 4)
    stocRvi = 2*(value4-0.5)
    trigger = 0.96*(shift(stocRvi,1)+0.02)
    return stocRvi, trigger

def stochastic_cycle(p, a=0.07, N=8):
    cycle,_     = i_cycle(p, a)
    maxCycle    = highest(cycle, N)
    minCycle    = lowest(cycle, N)
    value1      = (cycle - minCycle) / (maxCycle - minCycle)
    value2      = waverage(value1, 4)
    stocCycle   = 2*(value2-0.5)
    trigger     = 0.96*(shift(stocCycle,1)+0.02)
    return stocCycle, trigger

def stochastic_cg(p, N=10, N_stoc=8):
    cg      = cg_oscillator(p, N)
    maxCg   = highest(cg, N_stoc)
    minCg   = lowest(cg, N_stoc)
    value1  = (cg-minCg) / (maxCg-minCg)
    value2  = waverage(value1, 4)
    stocCg  = 2*(value2-0.5)
    trigger = 0.96*(shift(stocCg, 1)+0.02)
    return stocCg, trigger

def fisher_cycle(p, a=0.07, N=8):
    cycle,_     = i_cycle(p, a)
    maxCycle    = highest(cycle, N)
    minCycle    = lowest(cycle, N)
    value1      = (cycle-minCycle)/(maxCycle-minCycle)
    value2      = waverage(value1, 4)
    fisherCycle = 0.5*np.log((1+1.98*(value2-0.5))/(1-1.98*(value2-0.5)))
    trigger     = shift(fisherCycle,1)
    return fisherCycle, trigger

def fisher_cg(p, N=8):
    cg          = cg_oscillator(p, N)
    maxCg       = highest(cg, N)
    minCg       = lowest(cg, N)
    value1      = (cg-minCg) / (maxCg-minCg)
    value2      = waverage(value1, 4)
    fisherCg    = 0.5*np.log((1+1.98*(value2-0.5))/(1-1.98*(value2-0.5)))
    trigger     = shift(fisherCg,1)
    return fisherCg, trigger

def fisher_rvi(o,h,l,c, N=8):
    rvi,_ = relative_vigor(o, h, l, c, N)
    maxRvi      = highest(rvi, N)
    minRvi      = lowest(rvi, N)
    value3      = (rvi-minRvi) / (maxRvi-minRvi)
    value4      = waverage(value3, 4)
    fisherRvi   = 0.5*np.log((1+1.98*(value4-0.5))/(1-1.98*(value4-0.5)))
    trigger     = shift(fisherRvi,1)
    return fisherRvi, trigger

def get_cycle_period(p, a=0.07):
    # smooth = wsmooth(p, 4)
    cycle,_ = i_cycle(p, a=0.07)
    Q1          = np.zeros(len(p))
    I1          = np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    Period      = np.zeros(len(p))
    DC          = np.zeros(len(p))
    for i in range(6, len(p)):
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if i == 6: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        Period[i] = 0.15*InstPeriod[i] + 0.85*Period[i-1]
    return Period

def adaptive_cycle(p, a=0.07):
    smooth      = waverage(p, 4)
    cycle       = np.zeros(len(p))
    Q1          = np.zeros(len(p))
    I1          = np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    Period      = np.zeros(len(p))
    DC          = np.zeros(len(p))
    alpha1      = np.zeros(len(p))
    AdaptCycle  = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 6:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
            continue
        cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
        cycle[i] += 2*(1-a)*cycle[i-1] -(1-a)**2*cycle[i-2]
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if i == 6: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        Period[i] = 0.15*InstPeriod[i] + 0.85*Period[i-1]
        alpha1[i] = 2/(Period[i]+1)
        AdaptCycle[i] += (1-0.5*alpha1[i])**2 * (smooth[i]-2*smooth[i-1]+smooth[i-2])
        AdaptCycle[i] += 2*(1-alpha1[i])*AdaptCycle[i-1]
        AdaptCycle[i] -= (1-alpha1[i])**2 * AdaptCycle[i-2]
    trigger = shift(AdaptCycle, 1)
    return AdaptCycle, trigger

def adaptive_cg(p, a=0.07):
    smooth      = waverage(p, 4)
    cycle       = np.zeros(len(p))
    Q1          = np.zeros(len(p))
    I1          = np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    DC          = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    value1      = np.zeros(len(p))
    IntPeriod   = np.zeros(len(p))
    CG          = np.zeros(len(p))
    num         = np.zeros(len(p))
    denom       = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 5:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
            continue
        cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
        cycle[i] += 2*(1-a)*cycle[i-1] -(1-a)**2*cycle[i-2]
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if Q1[i-1] == 0: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        value1[i] = 0.15*InstPeriod[i] + 0.85*value1[i-1]
        IntPeriod[i] = np.round(value1[i] / 2)
        for j in range(int(IntPeriod[i])):
            num[i] += (1+j)*p[i-j]
            denom[i] += p[i-j]
        if denom[i] == 0:     continue
        CG[i] = -num[i]/denom[i] + (IntPeriod[i]+1)/2
    trigger = shift(CG, 1)
    return CG, trigger

def adaptive_rvi(o, h, l, c, a=0.07):
    p           = (h+l) / 2
    smooth      = waverage(p, 4)
    cycle       = np.zeros(len(p))
    Q1          = np.zeros(len(p))
    I1          = np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    DC          = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    value1      = np.zeros(len(p))
    value2      = np.zeros(len(p))
    Period      = np.zeros(len(p))
    Length      = np.zeros(len(p))
    rvi         = np.zeros(len(p))
    num         = np.zeros(len(p))
    denom       = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 6:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
            continue
        cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
        cycle[i] += 2*(1-a)*cycle[i-1] -(1-a)**2*cycle[i-2]
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if i == 6: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        Period[i] = 0.15*InstPeriod[i] + 0.85*Period[i-1]
        # Length[i] = int(np.round(4*Period[i]+3*Period[i-1]+2*Period[i-3]+Period[i-4])/20) # in the book
        Length[i] = int(np.round(4*Period[i]+3*Period[i-1]+2*Period[i-2]+Period[i-3])/20)
        value1[i] = ((c[i]-o[i])+2*(c[i-1]-o[i-1])+ 2*(c[i-2]-o[i-2]) + (c[i-3]-o[i-3])) / 6
        value2[i] = ((h[i]-l[i])+2*(h[i-1]-l[i-1])+ 2*(h[i-2]-l[i-2]) + (h[i-3]-l[i-3])) / 6
        for j in range(int(Length[i])):
            num[i] += value1[i-j]
            denom[i] += value2[i-j]
        if denom[i] == 0: continue
        rvi[i] = num[i]/denom[i]
    trigger = shift(rvi, 1)
    return rvi, trigger

def sinewave_indicator(p, a=0.07):
    smooth      = waverage(p, 4)
    cycle       = np.zeros(len(p))
    I1,I2       = np.zeros(len(p)), np.zeros(len(p))
    Q1,Q2       = np.zeros(len(p)), np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    MaxAmp      = np.zeros(len(p))
    AmpFix      = np.zeros(len(p))
    RE,IM,DC    = np.zeros(len(p)), np.zeros(len(p)), np.zeros(len(p))
    alpha1      = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    DCPeriod    = np.zeros(len(p))
    SmoothCycle = np.zeros(len(p))
    RealPart    = np.zeros(len(p))
    ImagPart    = np.zeros(len(p))
    DCPhase     = np.zeros(len(p))
    Value1      = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 6:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
            continue
        cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
        cycle[i] += 2*(1-a)*cycle[i-1] -(1-a)**2*cycle[i-2]
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if i == 6: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        Value1[i] = 0.15*InstPeriod[i] + 0.85*Value1[i-1]
        # Compute Dominant Cycle Phase
        DCPeriod[i] = int(Value1[i])
        for j in range(int(DCPeriod[i])):
            RealPart[i] += np.sin(2*np.pi*j/DCPeriod[i]) * cycle[i-j]
            ImagPart[i] += np.cos(2*np.pi*j/DCPeriod[i]) * cycle[i-j]
        if np.abs(ImagPart[i]) > 0.001:
            DCPhase[i] = np.arctan(RealPart[i]/ImagPart[i])
        else:
            DCPhase[i] = 0.5*np.pi*np.sign(RealPart[i])
        DCPhase[i] += 0.5*np.pi
        if ImagPart[i] < 0:
            DCPhase[i] += np.pi
        if DCPhase[i] > 1.75*np.pi:
            DCPhase[i] -= 2*np.pi
    Sine = DCPhase
    LeadSine = DCPhase + 0.25*np.pi
    return Sine, LeadSine

def smoothed_adaptive_momentum(p, a=0.07, cutoff=8):
    smooth      = waverage(p, 4)
    cycle       = np.zeros(len(p))
    I1,Q1       = np.zeros(len(p)), np.zeros(len(p))
    DeltaPhase  = np.zeros(len(p))
    MedianDelta = np.zeros(len(p))
    DC          = np.zeros(len(p))
    InstPeriod  = np.zeros(len(p))
    Period      = np.zeros(len(p))
    Value1      = np.zeros(len(p))
    a1          = np.exp(-np.pi/cutoff)
    # b1          = 2*a1*np.cos(1.738/2/cutoff)   # seems to be sqrt(3)=1.732
    b1          = 2*a1*np.cos(np.sqrt(3)/2/cutoff)
    c1          = a1**2
    coef2       = b1 + c1
    coef3       = -(c1+b1*c1)
    coef4       = c1**2
    coef1       = 1 - coef2 - coef3 - coef4
    Filt3       = np.zeros(len(p))
    for i in range(2, len(p)):
        if i < 6:
            cycle[i] = (p[i] - 2*p[i-1] + p[i-2]) / 4
            continue
        cycle[i] += (1-0.5*a)**2*(smooth[i]-2*smooth[i-1]+smooth[i-2])
        cycle[i] += 2*(1-a)*cycle[i-1] -(1-a)**2*cycle[i-2]
        Q1[i] = (0.0962*(cycle[i]-cycle[i-6])+0.5769*(cycle[i-2]-cycle[i-4])) * (0.5+0.08*InstPeriod[i-1])
        I1[i] = cycle[i-3]
        if i == 6: continue
        DeltaPhase[i] = (I1[i]/Q1[i]-I1[i-1]/Q1[i-1]) / (1 + I1[i]*I1[i-1]/(Q1[i]*Q1[i-1]))
        DeltaPhase[i] = max(0.1, DeltaPhase[i])
        DeltaPhase[i] = min(1.1, DeltaPhase[i])
        MedianDelta[i] = np.median(DeltaPhase[i-5:i+1])
        if MedianDelta[i] == 0:
            DC[i] = 15
        else:
            DC[i] = 6.28318 / MedianDelta[i] + 0.5
        InstPeriod[i] = 0.33 * DC[i] + 0.67 * InstPeriod[i-1]
        Period[i] = 0.15*InstPeriod[i] + 0.85*Period[i-1]
        Value1[i] = p[i] - p[i-int(Period[i])+1]
        Filt3[i] = coef1*Value1[i] + coef2*Value1[i-1] + coef3*Value1[i-2] + coef4*Value1[i-3]
    return Filt3

def butterworth_filter_2(p, period=15):
    a = np.exp(-np.sqrt(2)*np.pi/period)
    b = 2*a*np.cos(np.sqrt(2)*np.pi/period)
    coef2 = b
    coef3 = -a**2
    coef1 = (1 - coef2 - coef3) / 4
    Butter = np.zeros(len(p))
    for i in range(len(p)):
        if i < 3:
            Butter[i] = p[i]
        else:
            Butter[i] += coef1*(p[i]+2*p[i-1]+p[i-2])
            Butter[i] += coef2*Butter[i-1] + coef3*Butter[i-2]
    return Butter

def butterworth_filter_3(p, period=15):
    a = np.exp(-np.pi/period)
    b = 2*a*np.cos(1.738*np.pi/period)
    c = a*a
    coef2 = b+c
    coef3 = -(c+b*c)
    coef4 = c*c
    coef1 = (1-b+c)*(1-c)/8
    Butter = np.zeros(len(p))
    for i in range(len(p)):
        if i < 4:
            Butter[i] = p[i]
        else:
            Butter[i] += coef1*(p[i]+3*p[i-1]+3*p[i-2]+p[i-3])
            Butter[i] += coef2*Butter[i-1] + coef3*Butter[i-2] + coef4*Butter[i-3]
    return Butter

def super_smoother_2(p, period=15):
    a = np.exp(-np.sqrt(2)*np.pi/period)
    b = 2*a*np.cos(np.sqrt(2)*np.pi/period)
    coef2 = b
    coef3 = -a**2
    coef1 = (1 - coef2 - coef3)
    Filt2 = np.zeros(len(p))
    for i in range(len(p)):
        if i < 3:
            Filt2[i] = p[i]
        else:
            Filt2[i] += coef1*p[i] + coef2*Filt2[i-1] + coef3*Filt2[i-2]
    return Filt2

def super_smoother_3(p, period=15):
    a = np.exp(-np.pi/period)
    b = 2*a*np.cos(np.sqrt(3)*np.pi/period)
    c = a*a
    coef2 = b+c
    coef3 = -(c+b*c)
    coef4 = c*c
    coef1 = 1 - coef2 - coef3 - coef4
    Filt3 = np.zeros(len(p))
    for i in range(len(p)):
        if i < 4:
            Filt3[i] = p[i]
        else:
            Filt3[i] += coef1*p[i] + coef2*Filt3[i-1] + coef3*Filt3[i-2] + coef4*Filt3[i-3]
    return Filt3

def Laguerre_filter(p, gamma=0.8):
    L0 = np.zeros(len(p))
    L1 = np.zeros(len(p))
    L2 = np.zeros(len(p))
    L3 = np.zeros(len(p))
    Filt = np.zeros(len(p))
    FIR = waverage(p, 4)
    for i in range(1, len(p)):
        L0[i] = (1-gamma)*p[i] + gamma*L0[i-1]
        L1[i] = -gamma*L0[i] + L0[i-1] + gamma*L1[i-1]
        L2[i] = -gamma*L1[i] + L1[i-1] + gamma*L2[i-1]
        L3[i] = -gamma*L2[i] + L2[i-1] + gamma*L3[i-1]
    Filt = (L0+2*L1+2*L2+L3)/6
    return Filt, FIR

def Laguerre_RSI(p, gamma=0.5):
    L0 = np.zeros(len(p))
    L1 = np.zeros(len(p))
    L2 = np.zeros(len(p))
    L3 = np.zeros(len(p))
    CU = np.zeros(len(p))
    CD = np.zeros(len(p))
    lRSI = np.zeros(len(p))
    for i in range(1, len(p)):
        L0[i] = (1-gamma)*p[i] + gamma*L0[i-1]
        L1[i] = -gamma*L0[i] + L0[i-1] + gamma*L1[i-1]
        L2[i] = -gamma*L1[i] + L1[i-1] + gamma*L2[i-1]
        L3[i] = -gamma*L2[i] + L2[i-1] + gamma*L3[i-1]
        CU[i] += max(L0[i]-L1[i], 0) + max(L1[i]-L2[i], 0) + max(L2[i]-L3[i], 0)
        CD[i] += max(L1[i]-L0[i], 0) + max(L2[i]-L1[i], 0) + max(L3[i]-L2[i], 0)
        lRSI[i] = CU[i] / (CU[i] + CD[i])
    return lRSI

def leading_indicator(p, a1=0.25, a2=0.33):
    Lead = np.zeros(len(p))
    NetLead = np.zeros(len(p))
    EMA = np.zeros(len(p))
    for i in range(1, len(p)):
        Lead[i] = 2*p[i]+(a1-2)*p[i-1]+(1-a1)*Lead[i-1]
        NetLead[i] = a2*Lead[i] + (1-a2)*NetLead[i-1]
        EMA[i] = 0.5*p[i] + 0.5*EMA[i-1]
    return NetLead, EMA



#%% (-1) 试一试
# df = pd.read_csv('F:/Kai/pythonScripts/510050.XSHG.csv')
# o = df['open'].values
# h = df['high'].values
# l = df['low'].values
# c = df['close'].values
# p = (h+l) / 2
# fisher = fisher_transform(p, 40)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:])
# plt.grid()
# plt.subplot(2,1,2)
# plt.plot(fisher[-200:])
# plt.grid()

# trend,trigger = i_trend(p)
# plt.figure()
# plt.plot(p[-200:], label='price')
# plt.plot(trend[-200:], label='trend')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# cycle, trigger = i_cycle(p)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(cycle[-200:], label='cycle')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# cg = cg_oscillator(p)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(cg[-200:], label='cg')
# plt.grid()
# plt.legend()

# rvi,trigger = relative_vigor(o, h, l, c)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(rvi[-200:], label='rvi')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# #%%
# stocRsi, trigger = stochastic_rsi(p)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(stocRsi[-200:], label='stocRsi')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# #%%
# stocRvi, trigger = stochastic_rvi(o,h,l,c)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(stocRvi[-200:], label='stocRsi')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# #%%
# stocCycle, trigger = stochastic_cycle(p, a=0.07, N=8)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(stocCycle[-200:], label='stocCycle')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# #%%
# stocCg, trigger = stochastic_cg(p, N=8)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(stocCg[-200:], label='stocCg')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()
# #%%
# fcycle, trigger = fisher_cycle(p, a=0.07, N=8)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(fcycle[-200:], label='fcycle')
# plt.plot(trigger[-200:], label='trigger')
# plt.grid()
# plt.legend()

# #%%
# period = get_cycle_period(p)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-400:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(period[-400:], label='period')
# plt.grid()
# plt.legend()

# #%%
# cycle,trigger = i_cycle(p, a=0.07)
# AdaptCycle, trigger = adaptive_cycle(p, a=0.07)
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(AdaptCycle[-200:], label='AdaptCycle')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(cycle[-200:], label='cycle')
# plt.grid()
# plt.legend()

# #%%
# # cycle,trigger = i_cycle(p, a=0.07)
# AdaptCG, trigger = adaptive_cg(p, a=0.07)
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(AdaptCycle[-200:], label='AdaptCG')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(cycle[-200:], label='cycle')
# plt.grid()
# plt.legend()

# #%%
# AdaptRVI, trigger = adaptive_rvi(o,h,l,c, a=0.07)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(AdaptRVI[-200:], label='AdaptCG')
# plt.grid()
# plt.legend()

# #%%
# Sine, LeadSine = sinewave_indicator(p, a=0.07)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(np.sin(Sine[-200:]), label='Sine')
# plt.plot(np.sin(LeadSine[-200:]), label='LeadSine')
# plt.grid()
# plt.legend()

# #%%
# sam = smoothed_adaptive_momentum(p)
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(sam[-200:], label='SAM')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,3)
# r = np.hstack((0,np.diff(np.log(p))))*np.sign(sam)
# plt.plot(np.cumsum(r[-200:]), label='cumulative return')
# plt.grid()

# #%%
# bw2 = butterworth_filter_2(p)
# bw3 = butterworth_filter_3(p)
# sm2 = super_smoother_2(p)
# sm3 = super_smoother_3(p)

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(sm2[-200:], label='SM2')
# plt.plot(sm3[-200:], label='SM3')
# plt.grid()
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(bw2[-200:], label='BW2')
# plt.plot(bw3[-200:], label='BW3')
# plt.grid()
# plt.legend()


# #%%
# lRSI = Laguerre_RSI(p)
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p[-200:], label='price')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(lRSI[-200:], label='lRSI')
# plt.grid()
# plt.legend()

# #%%
# NL, EMA = leading_indicator(p)
# plt.figure()
# plt.subplot(1,1,1)
# plt.plot(p[-200:], label='price')
# plt.plot(NL[-200:], label='NetLead')
# plt.plot(EMA[-200:], label='EMA')
# plt.grid()
# plt.legend()


















