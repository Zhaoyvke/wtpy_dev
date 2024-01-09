from OptionInfo import *
from SupplementaryFunc import *

def exec(df_open:pl.DataFrame, df_close:pl.DataFrame, trading_dates):
    df_open = df_open.with_columns(pl.lit('not important').alias('exchg'))
    df_close = df_close.with_columns(pl.lit('not important').alias('exchg'))

    df_open = df_open.with_columns(pl.lit('CHN').alias('underlying'))
    df_close = df_close.with_columns(pl.lit('CHN').alias('underlying'))

    try:
        s_close = df_close['underlyingPrice'][0]
        s_open = df_open['underlyingPrice'][0]
    except:
        print(12)

    df_open = OptionInfo(df_open).getExposures()
    df_close = OptionInfo(df_close).getExposures()
    
    # delta attribution 
    delta_open = df_open.select((pl.col('Delta') * pl.col('pos') * pl.col('multiplier')).sum().alias('delta_open'))['delta_open'][0]
    # delta_close = df_close.select((pl.col('Delta') * pl.col('pos') * pl.col('multiplier')).sum().alias('delta_close'))['delta_close'][0]

    # d_delta = delta_close - delta_open
    attr_delta = delta_open * (s_close - s_open)


    # gamma attribution 
    gamma_open = df_open.select((pl.col('Gamma') * pl.col('pos') * pl.col('multiplier')).sum().alias('gamma_open'))['gamma_open'][0]
    # gamma_close = df_open.select((pl.col('Gamma') * pl.col('pos') * pl.col('multiplier')).sum().alias('gamma_close'))['gamma_close'][0]

    attr_gamma = 1 / 2 * gamma_open * (s_close - s_open)**2


    # theta attribution
    theta_open = df_open.select((pl.col('Theta') * pl.col('pos') * pl.col('multiplier')).sum().alias('gamma_open'))['gamma_open'][0]

    date_open = str(df_open['trading_time'][0])[:8]
    date_close = str(df_close['trading_time'][0])[:8]

    if type(trading_dates[0]) != str:
        trading_dates = [str(x) for x in trading_dates]

    dt = (np.where(np.array(trading_dates) == date_close)[0][0] - np.where(np.array(trading_dates) == date_open)[0][0]) / 240

    # time_open = datetime.strptime(df_open['trading_time'][0], '%Y%m%d%H%M%S')
    # time_close = datetime.strptime(df_close['trading_time'][0], '%Y%m%d%H%M%S')
    
    # dt = (time_close - time_open).total_seconds() / 240 / 24 / 60 / 60 #用一天24小时还是4小时交易时间

    attr_theta = dt * theta_open

    # vega_attribution
    df_close_iv = df_close.select(pl.col('code'), pl.col('IV').alias('close_IV'))
    df_open = df_open.join(df_close_iv, on='code', how='left')

    attr_vega = df_open.select(( (pl.col('close_IV') - pl.col('IV')) * pl.col('Vega') * pl.col('pos') * pl.col('multiplier')).sum().alias('attr_vega'))['attr_vega'][0]


    res = [attr_delta, attr_gamma, attr_theta, attr_vega]
    res.append(sum(res))

    return res




