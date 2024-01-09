import pandas as pd

from OptionInfo import *


class StressTest:
    def __init__(self, option=None, dayAfter=1, sp_range=None, iv_range=None):
        self.option = option
        self.days = dayAfter
        self.margin = 0

        if sp_range is None:
            self.sp_range = [-0.1, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.1]
        else:
            self.sp_range = sp_range

        if iv_range is None:
            self.iv_range = [-0.99, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1]
        else:
            self.iv_range = iv_range

    def st_on_s(self, iv=None):
        if iv is None:
            iv = self.option.sigma

        if self.option.sigma == 0:
            print(1)
            if self.option.price != 0:
                self.option.sigma = self.option.getIV(self.option.price)
            else:
                raise ValueError('No IV available')

        real_iv = self.option.sigma
        self.option.sigma = iv

        price_ranges = np.full(len(self.sp_range), np.nan)

        real_s = self.option.s
        real_interval = self.option.interval

        self.option.interval = real_interval - self.days / 240
        for i in range(len(self.sp_range)):
            self.option.s = (self.sp_range[i] + 1) * real_s
            price_ranges[i] = self.option.getBSprice()
            # print(self.option.sigma)

        profit = np.array(price_ranges) - self.option.price

        self.option.s = real_s
        self.option.sigma = real_iv
        self.option.interval = real_interval

        return profit

    def st_on_iv(self, s=None):
        if s is None:
            s = self.option.s

        if self.option.sigma == 0:
            if self.option.price != 0:
                self.option.sigma = self.option.getIV(self.option.price)
            else:
                raise ValueError('No IV available')

        real_s = self.option.s
        self.option.s = s

        price_ranges = np.full(len(self.iv_range), np.nan)

        real_iv = self.option.sigma
        real_interval = self.option.interval

        self.option.interval = real_interval - self.days / 240
        for i in range(len(self.iv_range)):
            self.option.sigma = (self.iv_range[i] + 1) * real_iv
            price_ranges[i] = self.option.getBSprice()

        profit = price_ranges - self.option.price

        self.option.s = real_s
        self.option.sigma = real_iv
        self.option.interval = real_interval

        return profit

    def st(self, margin=True, multiplier=10000):
        if self.option.sigma == 0:
            if self.option.price != 0:
                self.option.sigma = self.option.getIV(self.option.price)
            else:
                raise ValueError('No IV available')

        res = np.zeros((len(self.iv_range), len(self.sp_range)))

        for i in range(len(self.iv_range)):
            temp = self.st_on_s((self.iv_range[i] + 1) * self.option.sigma)
            res[i, :] = temp

        row_name = [str(iv * 100) + '% IV' for iv in self.iv_range]
        col_name = [str(sp * 100) + '% und' for sp in self.sp_range]

        res_df = pd.DataFrame(data=res, columns=col_name, index=row_name)

        real_margin = self.margin_cal_etf(self.option.price, self.option.s, self.option.k, multiplier, self.option.type)

        if margin:
            res_price = res_df + self.option.price

            for i in range(len(self.sp_range)):
                s0 = (self.sp_range[i] + 1) * self.option.s
                k = self.option.k
                multiplier = multiplier
                type = self.option.type

                res_price[col_name[i]] = res_price[col_name[i]].apply(self.margin_cal_etf,
                                                                      args=(s0, k, multiplier, type))
                # print(res_price)

            res_margin = res_price - real_margin
            self.margin = real_margin

        else:
            res_margin = pd.DataFrame(data=0, columns=col_name, index=row_name)

        # print(real_margin)

        return res_df, res_margin

    def margin_cal_etf(self, c, s0, k, multiplier, type):
        if type in ['c', 'C', 'Call', 'call']:  # call
            return (c + max(0.12 * s0 - max(k - s0, 0), 0.07 * s0)) * multiplier
        elif type in ['p', 'P', 'Put', 'put']:
            return min(c + max(0.12 * s0 - max(s0 - k, 0), 0.07 * k), k) * multiplier
        else:
            raise ValueError('Illegal TypeError')

class StressTest_Portfolio:
    def __init__(self, options=None, dayAfter=1, sp_range=None, iv_range=None):
        self.options = options
        self.days = dayAfter
        self.pos = []

        if sp_range is None:
            self.sp_range = [-0.1, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.1]
        else:
            self.sp_range = sp_range

        if iv_range is None:
            self.iv_range = [-0.99, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1]
        else:
            self.iv_range = iv_range


    def vanilla(self, multiplier, pos):
        self.pos = pos

        res = None
        res_m = None

        if type(multiplier) != list:
            multiplier = np.full(len(self.options), multiplier)

        for i in range(len(self.options)):
            stresstest = StressTest(self.options[i])
            st, st_m = stresstest.st(margin=True, multiplier=multiplier[i])

            try:
                res += st * self.pos[i] * multiplier[i]
                if self.pos[i] < 0:
                    res_m += st_m * self.pos[i]
            except:
                res = st * self.pos[i] * multiplier[i]
                if self.pos[i] < 0:
                    res_m = st_m * self.pos[i]

        return res, res_m


    def ds(self, multiplier, pos=[-1, -1]):
        self.pos = pos
        res_m = []
        res_p = []
        real_margins = []
        if type(multiplier) != list:
            multiplier = np.full(len(self.options), multiplier)
        for i in range(len(self.options)):
            stresstest = StressTest(self.options[i])
            st, st_m = stresstest.st(margin=True, multiplier=multiplier[i])

            real_margins.append(stresstest.margin)

            try:
                res += st * self.pos[i] * multiplier[i]
            except:
                res = st * self.pos[i] * multiplier[i]

            res_m.append(st_m + stresstest.margin)
            res_p.append(st + self.options[i].price)

        row_name = [str(iv * 100) + '% IV' for iv in self.iv_range]
        col_name = [str(sp * 100) + '% und' for sp in self.sp_range]

        res_margin = pd.DataFrame(data=0, columns=col_name, index=row_name)
        for i in range(res_margin.shape[0]):
            for j  in range(res_margin.shape[1]):
                if res_m[0].iloc[i, j] < res_m[1].iloc[i, j]:
                    margin = res_m[1].iloc[i, j] + res_p[0].iloc[i, j] * multiplier[0]
                else:
                    margin = res_m[0].iloc[i, j] + res_p[1].iloc[i, j] * multiplier[1]
                res_margin.iloc[i, j] = margin

        if real_margins[0] < real_margins[1]:
            real_margin = real_margins[1] + self.options[0].price * multiplier[0]
        else:
            real_margin = real_margins[0] + self.options[1].price * multiplier[1]

        res_margin -= real_margin

        return res, res_margin

    def spread(self, multiplier, pos=[-1, 1]):
        self.pos = pos
        if type(multiplier) != list:
            multiplier = np.full(len(self.options), multiplier)
        for i in range(len(self.options)):
            st, st_m = StressTest(self.options[i]).st(margin=True, multiplier=multiplier[i])

            try:
                res += st * self.pos[i] * multiplier[i]
            except:
                res = st * self.pos[i] * multiplier[i]

            if self.pos[i] > 0:
                res_margin = st_m

        return res, res_margin




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


if __name__ == '__main__':
    def col_to_str(data, cols):
        for col in cols:
            data[col] = data[col].apply(str)
        return data

    rqdatac.init('18616633529', 'wuzhi2020')

    accounts = ['50_citic', '50_nanhua', '51_citic', 'zx_guojun']

    date = str(datetime.today() - timedelta(0))[:10].replace('-', '')

    etfs = ['510050', '510300', '159919', '510500', '159915', '159922', '159901']


    for account in accounts:
        print('现在对账户%s进行压力测试' % account)
        options = dict(zip(etfs, [[], [], [], [], [], [], []]))
        path = 'E:\\OptionMonitor3\\OptionMonitor\\bin\\x86\\Debug\\Outputs\\Positions\\' + account + '/' + date + '.csv'
        df = pd.read_csv(path)
        df = df.rename(columns={'contract':'code'})
        df['code'] = df['code'].apply(str)

        codes = df['code'].apply(str).unique()

        properties = rqdatac.options.get_contract_property(order_book_ids=codes,
                                                           start_date=date,
                                                           end_date=date,
                                                           fields='product_name').reset_index()

        properties['underlying'] = properties['product_name'].apply(lambda x: x[0:6])

        for und, _ in options.items():
            options[und] = properties[properties['underlying'] == und]['order_book_id'].unique()

        with pd.ExcelWriter('StressTest/%s/%s.xlsx' % (account, date)) as writer:
            for und, codes in options.items():
                # print(und)
                if len(codes) == 0:
                    st, st_m = None, None
                else:
                    data = pd.read_csv('C:/Users/pc336/PycharmProjects/Option Data/IC_IM_MO/ETF&options/processed/%s_option.csv' % und)
                    data = col_to_str(data, ['date', 'code', 'underlying', 'maturity'])

                    data = data[(data['date'] == date) & (data['code'].isin(codes))].reset_index(drop=True)

                    data_option = data_to_option(data)

                    df_pos = df[df['code'].isin(codes)][['code', 'qty']]


                    data = pd.merge(data, df_pos, on='code')
                    multiplier = list(data['multiplier'])
                    pos = list(data['qty'])

                    st, st_m = StressTest_Portfolio(options=data_option).vanilla(multiplier, pos)

                    if st is not None:
                        st.to_excel(writer, str(und), index=True, startrow=0, startcol=0)
                    if st_m is not None:
                        st_m.to_excel(writer, str(und), index=True, startrow=0, startcol=12)

    print('测试完成')

