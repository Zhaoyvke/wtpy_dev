# -*- coding: utf-8 -*-
"""
Created on Thu Mar  14:17 2023

@author: wzer
"""
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import calendar


global must_columns
must_columns = ['date', 'code', 'underlying', 'underlyingPrice', 'strike', 'close', 'maturity', 'timeToMaturity',
                'exchg', 'type', 'multiplier', 'adjusted']


class Option:
    def __init__(self, s0, k0, r, days, sigma=0, cpflag='call', market=0, div=0):
        self.s = s0
        self.k = k0
        self.r = r
        self.interval = days / 240
        if self.interval == 0:
            self.interval = 0.1 / 240
        self.sigma = sigma
        self.type = cpflag
        self.price = market
        self.div = div

        self.delta, self.gamma, self.theta, self.vega, self.vanna, self.volga = 0, 0, 0, 0, 0, 0

    def BSpriceFormula(self, s0, k0, r, interval, sigma, cpflag):
        d1 = (np.log(s0 / k0) + (r + 0.5 * sigma ** 2) * interval) / sigma / np.sqrt(interval)
        d2 = d1 - sigma * np.sqrt(interval)

        if cpflag in ['c', 'C', 'call', 'Call']:
            price = s0 * norm.cdf(d1) - k0 * np.exp(-r * interval) * norm.cdf(d2)
        elif cpflag in ['p', 'P', 'put', 'Put']:
            price = k0 * np.exp(-r * interval) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
        else:
            raise ValueError('Illegal value of cpflag.')

        return price

    def getBSprice(self):
        return self.BSpriceFormula(self.s, self.k, self.r, self.interval, self.sigma, self.type)

    # def _getImpliedVolatility(self, market):
    #     def IV(sigma):
    #         return self.BSpriceFormula(self.s, self.k, self.r, self.interval, sigma, self.type) - market
    #
    #     impliedv = fsolve(IV, 0.1)[0]
    #
    #     return impliedv

    def hasIV(self, c0):
        if self.type in ['c', 'C', 'call', 'Call']:
            if c0 < self.s - self.k * np.exp(-self.r * self.interval) + 0.00011:
                return False
        elif self.type in ['p', 'P', 'put', 'Put']:
            if c0 < self.k * np.exp(-self.r * self.interval) - self.s + 0.00011:
                return False
        else:
            raise ValueError('Illegal value of option type')

        return True

    def getIV(self, market):
        if not self.hasIV(market):
            return 0.0001

        def IV(sigma):
            return self.BSpriceFormula(self.s, self.k, self.r, self.interval, sigma, self.type) - market

        # impliedv = fsolve(IV, 0.1)[0]

        # ţ�ٵ�����
        x0 = 0.5
        x1 = 0

        while True:
            if self._getVega(x0) == 0:
                return 0.0001

            x1 = x0 - IV(x0) / self._getVega(x0)

            if np.abs(x0 - x1) <= 1e-6:
                break

            x0 = x1

        return x1

    def getDelta(self):
        bs1 = self.BSpriceFormula(self.s + 0.01, self.k, self.r, self.interval, self.sigma, self.type)
        bs2 = self.BSpriceFormula(self.s - 0.01, self.k, self.r, self.interval, self.sigma, self.type)

        delta = (bs1 - bs2) / 0.02
        self.delta = delta
        return delta

    def getGamma(self):
        bs1 = self.BSpriceFormula(self.s * 1.01, self.k, self.r, self.interval, self.sigma, self.type)
        bs2 = self.BSpriceFormula(self.s * 0.99, self.k, self.r, self.interval, self.sigma, self.type)
        bs3 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, self.sigma, self.type)

        gamma = (bs1 - 2 * bs3 + bs2) / (0.01 * self.s) ** 2
        self.gamma = gamma
        return gamma

    def _getVega(self, sigma):
        bs1 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, sigma + .01, self.type)
        bs2 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, sigma - .01, self.type)

        return (bs1 - bs2) / 0.02

    def getVega(self):
        vega = self._getVega(self.sigma)
        self.vega = vega
        return vega

    def getTheta(self):
        bs1 = self.BSpriceFormula(self.s, self.k, self.r, self.interval * 1.01, self.sigma, self.type)
        bs2 = self.BSpriceFormula(self.s, self.k, self.r, self.interval * 0.99, self.sigma, self.type)

        theta = -(bs1 - bs2) / 0.02 / self.interval
        self.theta = theta

        return theta

    def getVanna(self):
        self.sigma += 0.01
        res1 = self.getGamma()

        self.sigma -= 0.01
        res2 = self.getGamma()

        vanna = (res1 - res2) / 0.02
        self.vanna = vanna
        self.sigma += 0.01
        self.gamma = self.getGamma()
        return vanna

    def getVolga(self):
        bs1 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, self.sigma + .01, self.type)
        bs3 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, self.sigma - .01, self.type)
        bs2 = self.BSpriceFormula(self.s, self.k, self.r, self.interval, self.sigma      , self.type)

        volga = (bs1 - 2 * bs3 + bs2) / 0.01** 2
        self.volga = volga
        return volga

    # def GreeksFormula(self, s0, k0, r, interval, sigma, call=True):
    #     d1 = (np.log(s0 / k0) + (r + 0.5 * sigma ** 2) * interval) / sigma / np.sqrt(interval)
    #     d2 = d1 - sigma * np.sqrt(interval)
    #
    #     if call:
    #         delta = norm.cdf(d1)
    #         theta = -s0 * norm.pdf(d1) * sigma / 2 / np.sqrt(interval) - r * k0 * np.exp(-r * interval) * norm.cdf(
    #             d2)
    #     else:
    #         delta = -norm.cdf(-d1)
    #         theta = -s0 * norm.pdf(d1) * sigma / 2 / np.sqrt(interval) + r * k0 * np.exp(-r * interval) * norm.cdf(
    #             -d2)
    #
    #     gamma = norm.pdf(d1) / s0 / sigma / np.sqrt(interval)
    #     vega = s0 * norm.pdf(d1) * np.sqrt(interval)
    #
    #     return [delta, gamma, vega, theta]
    #
    # def getGreeks(self):
    #     return self.GreeksFormula(self.s, self.k, self.r, self.interval, self.sigma, self.type)

    def getDollarDelta(self, multiplier):

        dollarDelta = self.delta * self.s * multiplier

        return dollarDelta

    def getDollarGamma(self, multiplier):

        dollarGamma = self.gamma * self.s ** 2 / 100 * multiplier

        return dollarGamma

    def getDollarVega(self, multiplier):

        dollarVega = self.vega * self.sigma * multiplier

        return dollarVega

    def getDollarVanna(self, multiplier):

        dollarVanna = self.vanna  *self.sigma * self.s * multiplier

        return dollarVanna

    def getDollarVolga(self, multiplier):

        dollarVolga = self.volga * self.sigma**2 * multiplier

        return dollarVolga

    def getInfo(self):
        print("The underlying asset price is", self.s)
        print("The strike price is ", self.k)
        print("The risk_free rate is ", self.r)
        print("Time to Maturity is ", self.interval * 240)
        print("The volatility is ", self.sigma)
        print("The BS price is ", self.getBSprice())


class OptionInfo:
    def __init__(self, file):
        option_data = file
        try:
            self.date = option_data['date'][0]
        except:
            self.date = '00000101'

        def is_full_data(df):
            must_cols = ['date', 'code', 'underlying', 'underlyingPrice', 'strike', 'close', 'maturity',
                         'timeToMaturity',
                         'exchg', 'type', 'multiplier', 'adjusted']
            has_cols = df.columns
            for col in must_cols:
                if col not in has_cols:
                    # print('ȱ�� %s, �����޷���������' % col)
                    pass
            return True

        if is_full_data(option_data):
            self.option_data = option_data
            self.full_data = option_data.sort(by=['exchg', 'timeToMaturity', 'strike', 'type'])

    def forthWed(self, month):
        thisYear = int(self.date[:4])
        thisMonth = int(self.date[4:6])
        month = int(month)

        if month >= thisMonth:
            if calendar.monthcalendar(thisYear, month)[0][2] == 0:
                day = calendar.monthcalendar(thisYear, month)[4][2]
            else:
                day = calendar.monthcalendar(thisYear, month)[3][2]
        else:
            thisYear += 1
            if calendar.monthcalendar(thisYear, month)[0][2] == 0:
                day = calendar.monthcalendar(thisYear, month)[4][2]
            else:
                day = calendar.monthcalendar(thisYear, month)[3][2]

        return datetime(thisYear, month, day).strftime('%Y%m%d')

    def isDelayedWed(self, date):
        if not rqdatac.is_trading_date(date):
            date = rqdatac.get_next_trading_date(date)
        return date

    def selectFund(self, underlying, df):
        # return df[df['underlying'] == underlying].reset_index().drop(columns=['index'])
        return df.filter(pl.col('underlying') == underlying)

    # def cleanDF(self, df):
    #     df.drop(df[pd.isnull(df['name'])].index, inplace=True)
    #     df.loc[:, 'name'] = df['name'].apply(lambda x: str(x))
    #     df.loc[:, 'name'] = df['name'].apply(
    #         lambda x: [x.split('F')[1][0], x.split('F')[1].split('��')[0][1:], x[-1]])
    #
    #     df.loc[:, 'type'] = df['name'].apply(lambda x: 'call' if x[0] == '��' else 'put')
    #
    #     if df['underlying'].iloc[0] == '510050':
    #         df['adjusted'] = df['name'].apply(lambda x: True if x[-1] == 'A' else False)
    #     else:
    #         df['adjusted'] = 'TBD'
    #
    #     df['multiplier'] = df['adjusted'].apply(
    #         lambda x: np.ceil(10000 * self.closeDiv / (self.closeDiv - self.div)) if x else 10000)
    #
    #     def getUnderlyingPrice(x):
    #         try:
    #             return self.option_data[self.option_data['code'] == int(x)]['close'].iloc[0]
    #         except:
    #             if x[0] == '5':
    #                 return rqdatac.get_price('%s.XSHG' % x, self.date, self.date, fields='close')['close'].iloc[0]
    #             else:
    #                 return rqdatac.get_price('%s.XSHE' % x, self.date, self.date, fields='close')['close'].iloc[0]
    #
    #     df.loc[:, 'underlyingPrice'] = df['underlying'].apply(getUnderlyingPrice)
    #
    #     # calculate time to maturity
    #     holidays = self.holidays
    #
    #     df.loc[:, 'maturity'] = df['name'].apply(lambda x: self.isDelayedWed(self.forthWed(x[1])))
    #
    #     currentDate = df['date'].iloc[0]
    #     df.loc[:, 'timeToMaturity'] = df['maturity'].apply(
    #         lambda x: np.busday_count(datetime.date(datetime.strptime(str(currentDate), "%Y%m%d")),
    #                                   datetime.date(datetime.strptime(x, "%Y%m%d"))))
    #     holidayOffset = df['maturity'].apply(lambda x: np.sum(np.array(holidays) <= x))
    #     df.loc[:, 'timeToMaturity'] -= holidayOffset
    #
    #     df.drop(columns=['name'], inplace=True)
    #     df = df.sort_values(by=['exchg', 'strike', 'timeToMaturity', 'type']).reset_index().drop(columns=['index'])
    #
    #     return df

    def singleExposures(self, underlyingPrice, strike, RFrate, timeToMaturity, iv, opt_type, close, multiplier, fields):
        s0 = underlyingPrice
        k0 = strike
        r = RFrate
        days = timeToMaturity
        sigma = iv
        call = opt_type
        c0 = close
        m = multiplier

        currentOption = Option(s0, k0, r, days, sigma, call)
        
        res = []
        for field in fields:
            try:
                tmp = eval("currentOption.get%s()"%field)
            except:
                tmp = eval("currentOption.get%s(%s)"%(field, m))
            res.append(tmp)


        return tuple(res)

    def singleDeposit(self, df):
        s0 = df['underlyingPrice']
        k = df['strike']
        und = df['underlying']
        optionType = df['type']
        c = df['close']
        multiplier = df['multiplier']

        und_type = 'stock_idx'
        if und in ['510050', '510300', '510500', '159915', '159922', '159901', '159919']:
            und_type = 'etf'

        if und_type in ['eft', 'ETF']:
            if optionType in ['c', 'C', 'Call', 'call']:  # call
                return (c + max(0.12 * s0 - max(k - s0, 0), 0.07 * s0)) * multiplier
            elif optionType in ['p', 'P', 'Put', 'put']:
                return min(c + max(0.12 * s0 - max(s0 - k, 0), 0.07 * k), k) * multiplier
            else:
                raise TypeError('Illegal ValueError')

        # ���Ͻ����ϵ�� 0.15, ��ͱ���ϵ��0.5
        else:
            # �Ϲ���Ȩ����ֿ��ֱ�֤�� = (��Լǰ����ۡ���Լ������+max(���ָ��ǰ���̼ۡ���Լ��������Լ��֤�����ϵ��-�Ϲ���Ȩ��ֵ����ͱ���ϵ�������ָ��ǰ���̼ۡ���Լ���� �� ��Լ��֤�����ϵ����
            # �Ϲ���Ȩ����ֿ��ֱ�֤��=(��Լǰ����ۡ���Լ������+max(���ָ��ǰ���̼ۡ���Լ����x��Լ��֤�����ϵ��-�Ϲ���Ȩ��ֵ���ͱ���ϵ������Լ��Ȩ�۸����Լ��������Լ��֤�����ϵ����
            if optionType in ['c', 'C', 'Call', 'call']:  # call
                return c * multiplier + multiplier * max(s0 * 0.15 - max(k - s0, 0), 0.5 * s0 * 0.15)
            elif optionType in ['p', 'P', 'Put', 'put']:
                return c * multiplier + multiplier * max(s0 * 0.15 - max(s0 - k, 0), 0.5 * k * 0.15)
            else:
                raise TypeError('Illegal ValurError')

    def addRFIV(self, data):
        df = data.clone()
        df = df.with_columns(pl.lit(0).alias('RFrate'))

        # i = 0
        # while i <= df.shape[0] - 1:
        #     rf = np.log((df['close'][i + 1] + df['underlyingPrice'][i + 1] - df['close'][i]) / df['strike'][i]) / (
        #             - df['timeToMaturity'][i] / 240)
        #     df['RFrate'].iloc[i] = rf
        #     df['RFrate'].iloc[i + 1] = rf
        #     i += 2



        def singleIV(underlyingPrice, strike, RFrate, timeToMaturity, opt_type, close):
            s0 = underlyingPrice
            k0 = strike
            r = RFrate
            days = timeToMaturity
            call = opt_type
            c0 = close

            sigma = 0.5
            currentOption = Option(s0, k0, r, days, sigma, call)

            currentOption.sigma = currentOption.getIV(c0)

            return currentOption.sigma

        cols = list(df.columns)
        # print(df.columns)
        used_cols = ['underlyingPrice', 'strike', 'RFrate', 'timeToMaturity', 'type', 'close']
        indices = []
        for used_col in used_cols:
            indices.append(cols.index(used_col))
        # print(indices)

        iv_df = df.apply(lambda x: singleIV(x[indices[0]], 
                                         x[indices[1]], 
                                         x[indices[2]], 
                                         x[indices[3]], 
                                         x[indices[4]], 
                                         x[indices[5]]))
        # print(df.columns)
        iv_df = iv_df.rename({'apply': 'IV'})
        df = pl.concat([df, iv_df], how="horizontal")
        return df

    def Exposures(self, df, fields):
        cols = ['Delta', 'Gamma', 'Vega', 'Theta', 'DollarDelta', 'DollarGamma', 'DollarVega']

        if fields is None:
            fields = cols

        for field in fields:
            if field not in cols:
                raise ValueError(f'{f} not available')
        # df = df.lazy()

        # df = df.select(pl.col('*'), pl.struct(df.columns).apply(self.singleExposures, args=(fields, )).alias('tmp'))
        # df = df.with_columns([pl.col('tmp').apply(lambda x: x[0]).alias(cols[0]),
        #                       pl.col('tmp').apply(lambda x: x[1]).alias(cols[1]),
        #                       pl.col('tmp').apply(lambda x: x[2]).alias(cols[2]),
        #                       pl.col('tmp').apply(lambda x: x[3]).alias(cols[3]),
        #                       pl.col('tmp').apply(lambda x: x[4]).alias(cols[4]),
        #                       pl.col('tmp').apply(lambda x: x[5]).alias(cols[5]),
        #                       pl.col('tmp').apply(lambda x: x[6]).alias(cols[6])])
        new_cols_num = len(fields)
        cols = list(df.columns)
        # print(df.columns)
        used_cols = ['underlyingPrice', 'strike', 'RFrate', 'timeToMaturity', 'IV', 'type', 'close', 'multiplier']
        indices = []
        for used_col in used_cols:
            indices.append(cols.index(used_col))
        # print(indices)

        expo_df = df.apply(lambda x: self.singleExposures(x[indices[0]], 
                                         x[indices[1]], 
                                         x[indices[2]], 
                                         x[indices[3]], 
                                         x[indices[4]], 
                                         x[indices[5]],
                                         x[indices[6]],
                                         x[indices[7]],
                                         fields))
        # print(expo_df)
        for i in range(new_cols_num):
            # print(i, fields[i])
            expo_df = expo_df.rename({'column_%s'%i: fields[i]})
        df = pl.concat([df, expo_df], how="horizontal")

        # print(fields)
        if 'openDeposit' in fields:
            df = df.select(pl.col('*'), pl.struct(df.columns).apply(self.singleDeposit).alias('openDeposit'))

        return df

    def getExposures(self, fields=None, underlying=None, file=False):
        if underlying is not None:
            expo = self.selectFund(str(underlying), self.full_data)
        else:
            expo = self.full_data.clone()

        expo = self.addRFIV(expo)
        # print('IV finished')
        expo = self.Exposures(expo, fields)
        # print('Exposures finished')

        if file:
            file_name = self.date + '_' + str(underlying) + '_Exposures.csv'
            expo.to_csv(file_name)

        return expo

    def singleBSprice(self, df):

        s0 = df['underlyingPrice']
        k0 = df['strike']
        r = df['RFrate']
        days = df['timeToMaturity']
        sigma = df['IV']
        call = df['type']

        currentOption = Option(s0, k0, r, days, sigma, call)

        return currentOption.getBSprice()

    def stressTest(self, data, change, daysAfter, deposit=False):
        df = data.clone()

        df = df.select(pl.col('*'), pl.struct(df.columns).apply(self.singleDeposit).alias('openDeposit'))

        df = df.with_columns(pl.col('timeToMaturity') - daysAfter)

        originalClose = df['close']
        originalUnderlyingPrice = df['underlyingPrice']

        colName = str(change * 100) + '% P&L'

        df = df.with_columns(((1 + change) * originalUnderlyingPrice).alias('underlyingPrice'))

        df = df.select(pl.col('*'), pl.struct(df.columns).apply(self.singleBSprice).alias(colName))

        if deposit:
            df = df.with_columns(pl.col(colName).alias('close'))

            colName2 = str(change * 100) + '% Deposit'
            df = df.select(pl.col('*'), pl.struct(df.columns).apply(self.singleDeposit).alias(colName2))
            # df = df.with_columns((pl.col('openDeposit')-pl.col(colName2)).apply(lambda x: round(x, 4)).alias(colName2))

            df = df.with_columns(originalClose.alias('close'))
        df = df.with_columns(pl.col(colName).apply(lambda x: round(x, 4)))
        df = df.with_columns((pl.col('multiplier') * (pl.col(colName) - pl.col('close'))).alias(colName))
        df = df.with_columns(originalUnderlyingPrice.alias('underlyingPrice'))

        return df

    def getStressTest(self, change, daysAfter, underlying=None, deposit=False, file=False):
        if underlying is not None:
            st = self.selectFund(str(underlying), self.full_data)
        else:
            st = self.full_data.clone()
        st = self.addRFIV(st)

        st = self.stressTest(st, change, daysAfter, deposit)

        if file:
            file_name = self.date + '_' + str(underlying) + '_' + str(daysAfter) + 'days_StressTest.csv'
            try:
                existedFile = pl.read_csv(file_name)
                if deposit:
                    newCol = st[:, -2]
                    existedFile = existedFile.with_columns(newCol.alias(st.columns[-2]))

                    newCol = st[:, -1]
                    existedFile = existedFile.with_columns(newCol.alias(st.columns[-1]))
                else:
                    newCol = st[:, -1]
                    existedFile = existedFile.with_columns(newCol.alias(st.columns[-1]))

                with open(file_name, 'w', encoding='gbk', newline='') as f:
                    existedFile.write_csv(f, index=False)
            except:
                with open(file_name, 'w', encoding='gbk', newline='') as f:
                    st.write_csv(f, index=False)

        return st

    def getOptionComb(self, df):
        data = self.full_data.clone()

        codes = df['code'].unique().to_list()

        optionComb = data.filter(pl.col('code').is_in(codes))
        optionComb = self.addRFIV(optionComb)

        optionComb = optionComb.sort(by='code')

        return optionComb

    def getCombGreeksContribution(self, file, deltaS):
        posName = 'Pos'
        pos = file.clone()
        pos = pos.sort(by=pos.columns[0])
        optionComb = self.getOptionComb(pos)

        expo = self.Exposures(optionComb)

        # expo.loc[:, 'Qty'] = np.array(pos.iloc[:, 1])
        expo = expo.join(pos, on='code')

        expo = expo.with_columns([(pl.col('Delta') * deltaS * pl.col('multiplier')).alias('Delta Contribution'),
                                  (pl.col('Gamma') * deltaS ** 2 / 2 * pl.col('multiplier')).alias('Gamma Contribution'),
                                  (pl.col('Theta') / 240 * pl.col('multiplier')).alias('Theta Contribution')
                                  ])

        return expo

    def getCombExposures(self, pos_data, file=False):
        posName = 'Pos'
        pos = pos_data.clone()
        pos = pos.sort(by=pos.columns[0])
        optionComb = self.getOptionComb(pos)

        expo = self.Exposures(optionComb)

        # expo.loc[:, 'Qty'] = np.array(pos.iloc[:, 1])
        expo = expo.join(pos, on='code')

        shortPos = np.where(np.array(expo['pos']) < 0, np.array(expo['pos']), 0)
        longPos = np.where(np.array(expo['pos']) > 0, np.array(expo['pos']), 0)
        allPos = np.array(expo['pos'])


        combValue = np.sum(allPos * np.array(expo['close']) * np.array(expo['multiplier']))
        combDelta = np.sum(allPos * np.array(expo['Delta']))
        combGamma = np.sum(allPos * np.array(expo['Gamma']))
        combTheta = np.sum(allPos * np.array(expo['Theta']))

        combDollarDelta = np.sum(allPos * np.array(expo['DollarDelta']))
        combDollarGamma = np.sum(allPos * np.array(expo['DollarGamma']))
        combDollarVega = np.sum(allPos * np.array(expo['DollarVega']))
        combDeposit = np.sum(shortPos * np.array(expo['openDeposit'])) * -1

        combExpo = pl.DataFrame(data=np.array(
            [combValue, combDelta, combGamma, combTheta, combDollarDelta, combDollarGamma, combDollarVega, combDeposit]).reshape(1, 8),
                                schema=['Total Value', 'Total Delta', 'Total Gamma', 'Total Theta',
                                         'Total dollar Delta', 'Total dollar Gamma', 'Total dollar Vega',
                                         'Total Deposit'])

        if file:
            file_name = str(self.date) + '_' + posName + '_Exposures.csv'
            with open(file_name, 'w', encoding='gbk', newline='') as f:
                combExpo.write_csv(f, index=False)
                expo.write_csv(f, index=False)

        return [combExpo, expo]

    def getCombStressTest(self, pos_data, change, daysAfter=5, file=False):
        posName = 'Pos'
        pos = pos_data.clone()

        pos = pos.sort(by=pos.columns[0])
        optionComb = self.getOptionComb(pos)

        st = self.stressTest(optionComb, change, daysAfter, deposit=True)

        # st.loc[:, 'Qty'] = np.array(pos.iloc[:, 1])
        st = st.join(pos, on='code')

        shortPos = np.where(np.array(st['pos']) < 0, np.array(st['pos']), 0)
        longPos = np.where(np.array(st['pos']) > 0, np.array(st['pos']), 0)
        allPos = st['pos']

        combPNL = np.sum(np.array(allPos) * np.array(st[str(change * 100) + '% P&L']))
        combDeposit = np.sum(shortPos * np.array(st[str(change * 100) + '% Deposit'])) * -1

        combExpo = pl.DataFrame(data=np.array([combPNL, combDeposit]).reshape(1, 2),
                                schema=['%s Total PNL' % (str(change * 100) + '%'),
                                         '%s Margin' % (str(change * 100) + '%')])

        if file:
            file_name = str(self.date) + '_' + posName + '_StressTest.csv'
            with open(file_name, 'w', encoding='gbk', newline='') as f:
                combExpo.write_csv(f, index=False)
                st.write_csv(f, index=False)

        return [combExpo, st]


if __name__ == "__main__":
    # data = pl.read_csv('C:/Users/pc336/PycharmProjects/Option Data/IC_IM_MO/ETF&options/processed/159901_option.csv')


    # def col_to_str(data, cols):
    #     for col in cols:
    #         data = data.with_columns(pl.col(col).apply(str))
    #     return data

    # data = col_to_str(data, ['date', 'code', 'underlying', 'maturity'])

    # print(OptionInfo(data).getStressTest(0.05, 2))
    # pos = pd.read_csv('pos.csv')
    # options.getExposures(510050, True)
    # options.getStressTest(underlying=510050, change=0.05, daysAfter=2, deposit=True, file=True)
    # options.getCombExposures('pos.csv', file=True)
    # options.getCombStressTest('pos.csv', 0.1, 2, True)

    option = pl.read_csv('E:/PycharmProjects/Option Data/IC_IM_MO/ETF&options/processed/510300_option.csv')
    option = option.filter(pl.col('timeToMaturity') <= 21).filter(pl.col('timeToMaturity') > 0)
    option_expo = OptionInfo(option.filter(pl.col('date') == 20220121)).getExposures(fields=['Gamma'])
