import numpy as np
import pandas as pd


def IndNeutralize(series, factor_class):
    """
    IndNeutralize: Neutralizes a given series with respect to a specified factor class.

    Parameters:
    - series: Pandas Series to be neutralized.
    - factor_class: Factor class for neutralization (e.g., IndClass.sector, IndClass.industry).

    Returns:
    - Neutralized Pandas Series.
    """
    # Define your neutralization logic here
    # For example, you might want to subtract the mean of each group defined by factor_class
    # from the original series.
    neutralized_series = series.groupby(factor_class).transform(lambda x: x - x.mean())
    return neutralized_series

def scale(series, multiplier):
    """
    scale: Scales a given series by a specified multiplier.

    Parameters:
    - series: Pandas Series to be scaled.
    - multiplier: Scaling multiplier.

    Returns:
    - Scaled Pandas Series.
    """
    scaled_series = series * multiplier
    return scaled_series


def delta(series, period=1):
    return series.diff(period)

def alpha1(data, window_returns=20, power=2, argmax_window=5):
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : Close), 2.), 5)) - 0.5)

    Parameters:
    - window_returns: Lookback period for returns calculation
    - power: Power for SignedPower calculation
    - argmax_window: Window for Ts_ArgMax

    Returns:
    - A pandas Series representing the alpha signal
    """
    returns = data['Close'].pct_change()
    signal = (returns < 0).rolling(window=window_returns).std().apply(
        lambda x: np.argmax(np.power(np.abs(x), power), axis=0)
    ).rank().rolling(window=argmax_window).mean() - 0.5
    return signal

# Sample code for generating signal condition
# signal_condition_alpha1 = alpha1(data)

def alpha2(data, delta_lookback=2, window_corr=6):
    """
    Alpha#2: (-1 * correlation(rank(delta(log(Volume), 2)), rank(((Close - Open) / Open)), 6))

    Parameters:
    - delta_lookback: Lookback period for delta calculation
    - window_corr: Window for correlation calculation

    Returns:
    - A pandas Series representing the alpha signal
    """
    delta_Volume = np.log(data['Volume']).diff(delta_lookback)
    delta_price_ratio = (data['Close'] - data['Open']) / data['Open']
    signal = -1 * pd.Series(delta_Volume.rank(), name='delta_Volume_rank').rolling(
        window=window_corr).corr(pd.Series(delta_price_ratio.rank(), name='delta_price_ratio_rank'))
    return signal

# Sample code for generating signal condition
# signal_condition_alpha2 = alpha2(data)

def alpha3(data, window_corr=10):
    """
    Alpha#3: (-1 * correlation(rank(Open), rank(Volume), 10))

    Parameters:
    - window_corr: Lookback period for correlation calculation

    Returns:
    - A pandas Series representing the alpha signal
    """
    signal = -1 * pd.Series(data['Open'].rank(), name='Open_rank').rolling(
        window=window_corr
    ).corr(pd.Series(data['Volume'].rank(), name='Volume_rank'))
    return signal

# Sample code for generating signal condition
# signal_condition_alpha3 = alpha3(data)

def alpha4(data, ts_rank_lookback=9):
    """
    Alpha#4: (-1 * Ts_Rank(rank(low), 9))

    Parameters:
    - ts_rank_lookback: Lookback period for Ts_Rank

    Returns:
    - A pandas Series representing the alpha signal
    """
    signal =  ((data["Low"].rank(ascending=False, method='max', pct=True)
              .rolling(window=3)
              .apply(lambda x: x[-1])
              )
            .fillna(0)) 
    return signal

# Sample code for generating signal condition
# signal_condition_alpha4 = alpha4(data)

def alpha5(data, vwap_lookback=10):
    """
    Alpha#5: (rank((Open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((Close - vwap)))))

    Parameters:
    - vwap_lookback: Lookback period for VWAP calculation

    Returns:
    - A pandas Series representing the alpha signal
    """
    vwap = data['Close'].rolling(window=vwap_lookback).mean()
    signal = pd.Series(data['Open'].rank(), name='Open_rank').rolling(window=vwap_lookback).apply(
        lambda x: (x[-1] - vwap[-1]) / vwap[-1], raw=True
    ).rank() * -1 * np.abs(pd.Series(data['Close'] - vwap).rank())
    return signal

# Sample code for generating signal condition
# signal_condition_alpha5 = alpha5(data)

def alpha6(data, window_corr=10):
    """
    Alpha#6: (-1 * correlation(Open, Volume, 10))

    Parameters:
    - window_corr: Window for correlation calculation

    Returns:
    - A pandas Series representing the alpha signal
    """
    signal = -1 * pd.Series(data['Open'].rolling(window=window_corr).corr(data['Volume']))
    return signal

# Sample code for generating signal condition
# signal_condition_alpha6 = alpha6(data)

def alpha7(data, adv_window=20, ts_rank_lookback=60):
    """
    Alpha#7: ((adv20 < Volume) ? ((-1 * ts_rank(abs(delta(Close, 7)), 60)) * sign(delta(Close, 7))) : (-1 * 1))

    Parameters:
    - adv_window: Window for calculating the average daily Volume
    - ts_rank_lookback: Lookback period for ts_rank

    Returns:
    - A pandas Series representing the alpha signal
    """
    adv20 = data['Volume'].rolling(window=adv_window).mean()
    condition = adv20 < data['Volume']
    delta_Close_7 = data['Close'].diff(7)
    signal = np.where(
        condition,
        -1 * pd.Series(np.abs(delta_Close_7).rolling(window=ts_rank_lookback).apply(
            lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)), raw=True
        )).rank() * np.sign(delta_Close_7),
        -1
    )
    return signal


def alpha8(data, window_sum=5, window_delay=10):
    """
    Alpha#8: (-1 * rank(((sum(open, window_sum) * sum(returns, window_sum)) -
    delay((sum(open, window_sum) * sum(returns, window_sum)), window_delay))))
    """
    data['returns'] = data['Close'].pct_change()

    sum_open_returns = data['Open'].rolling(window=window_sum).sum() * data['returns'].rolling(window=window_sum).sum()
    alpha8_values = -1 * sum_open_returns.rank() + sum_open_returns.shift(window_delay).rank()
    return alpha8_values

def alpha9(data, window_delta=5):
    """
    Alpha#9: ((0 < ts_min(delta(close, 1), window_delta)) ? delta(close, 1) :
    ((ts_max(delta(close, 1), window_delta) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    """
    delta_close = data['Close'].diff(1)
    condition1 = (delta_close > 0) & (delta_close.rolling(window=window_delta).min() > 0)
    condition2 = (delta_close < 0) & (delta_close.rolling(window=window_delta).max() < 0)
    alpha9_values = np.where(condition1, delta_close, np.where(condition2, -1 * delta_close, 0))
    return alpha9_values

def alpha10(data, window_delta=4):
    """
    Alpha#10: rank(((0 < ts_min(delta(close, 1), window_delta)) ? delta(close, 1) :
    ((ts_max(delta(close, 1), window_delta) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    """
    delta_close = data['Close'].diff(1)
    condition1 = (delta_close > 0) & (delta_close.rolling(window=window_delta).min() > 0)
    condition2 = (delta_close < 0) & (delta_close.rolling(window=window_delta).max() < 0)
    alpha10_values = delta_close.rank(ascending=False) * (np.where(condition1, delta_close,
                                       np.where(condition2, -1 * delta_close, 0)))
    return alpha10_values

def alpha11(data, vwap_lookback=3, window_volume=3):
    """
    Alpha#11: ((rank(ts_max((vwap - close), window_vwap)) + rank(ts_min((vwap - close), window_vwap))) *
    rank(delta(volume, window_volume)))
    """

    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    vwap_close_diff = data['VWAP'] - data['Close']
    alpha11_values = ((vwap_close_diff.rolling(window=vwap_lookback).max().rank() +
                       vwap_close_diff.rolling(window=vwap_lookback).min().rank()) *
                      data['Volume'].diff(window_volume).rank())
    return alpha11_values

def alpha12(data, window_volume=1):
    """
    Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    alpha12_values = np.sign(data['Volume'].diff(1)) * -1 * data['Close'].diff(1)
    return alpha12_values

def alpha13(data, window_covariance=5):
    """
    Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), window_covariance)))
    """
    alpha13_values = -1 * data['Close'].rank(ascending=False).rolling(window=window_covariance).cov(
        data['Volume'].rank(ascending=False).rolling(window=window_covariance))
    return alpha13_values

def alpha14(data, window_returns=3, window_corr=10):
    """
    Alpha#14: ((-1 * rank(delta(returns, window_returns))) * correlation(open, volume, window_corr))
    """
    data['returns'] = data['Close'].pct_change()

    alpha14_values = -1 * data['returns'].diff(window_returns).rank() * data['Open'].rolling(window=window_corr).corr(
        data['Volume'])
    return alpha14_values

def alpha15(data, window_corr=3, window_rank=3):
    """
    Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), window_corr)), window_rank))
    """
    alpha15_values = -1 * data['High'].rank(ascending=False).rolling(window=window_corr).corr(
        data['Volume'].rank(ascending=False)).rank().rolling(window=window_rank).sum()
    return alpha15_values

def alpha16(data, window_covariance=5):
    """
    Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), window_covariance)))
    """
    alpha16_values = -1 * data['High'].rank(ascending=False).rolling(window=window_covariance).cov(
        data['Volume'].rank(ascending=False).rolling(window=window_covariance))
    return alpha16_values

def alpha17(data, window_ts_rank=10, window_delta=1, window_ts_rank_volume=5):
    """
    Alpha#17: (((-1 * rank(ts_rank(close, window_ts_rank))) * rank(delta(delta(close, 1), window_delta))) *
    rank(ts_rank((volume / adv20), window_ts_rank_volume)))
    """
    alpha17_values = (((-1 * data['Close'].rolling(window=window_ts_rank).rank().rank()) *
                       data['Close'].diff(window_delta).diff(window_delta).rank()) *
                      data['Volume'].rolling(window=window_ts_rank_volume).rank())
    return alpha17_values

def alpha18(data, window_std=5, window_corr=10):
    """
    Alpha#18: (-1 * rank(((stddev(abs((close - open)), window_std) + (close - open)) +
    correlation(close, open, window_corr))))
    """
    close_open_diff = data['Close'] - data['Open']
    alpha18_values = -1 * ((np.abs(close_open_diff).rolling(window=window_std).std() +
                            close_open_diff + data['Close'].rolling(window=window_corr).corr(
        data['Open'])).rank())
    return alpha18_values



import pandas as pd
import numpy as np

# Utility function for conditional expressions
def if_else(condition, true_value, false_value):
    return np.where(condition, true_value, false_value)

def alpha_19(data, window_returns=250, window_sum=250):
    returns = data['Close'].pct_change()
    alpha = -1 * np.sign(((data['Close'] - data['Close'].shift(7)) + pd.Series.rolling(returns, window=window_sum).sum().rank(ascending=False)))
    return alpha

def alpha_20(data, window=1):
    open_rank = pd.Series.rank(data['Open'] - data['High'].shift(window))
    close_rank = pd.Series.rank(data['Open'] - data['Close'].shift(window))
    low_rank = pd.Series.rank(data['Open'] - data['Low'].shift(window))
    alpha = -1 * open_rank * close_rank * low_rank
    return alpha

def alpha_21(data, window1=8, window2=2, volume=None, adv20=None):
    volume = data['Volume'] if volume is None else volume
    adv20 = data['Volume'].rolling(window=20).mean() if adv20 is None else adv20
    sum_close_8 = pd.Series.rolling(data['Close'], window=window1).sum() / window1
    std_close_8 = pd.Series.rolling(data['Close'], window=window1).std()
    condition = (sum_close_8 + std_close_8) < (pd.Series.rolling(data['Close'], window=window2).sum() / window2)
    alpha = if_else(condition, -1, if_else((pd.Series.rolling(data['Close'], window=window2).sum() / window2) < ((sum_close_8 - std_close_8)), 1, if_else((1 < (volume / adv20)) | ((volume / adv20) == 1), 1, -1)))
    return alpha

def alpha_22(data, window_corr=5, window_std=20):
    alpha = -1 * (pd.Series.rolling(data['High'].corr(data['Volume'], window=window_corr), window=5).diff(5) * pd.Series.rank(pd.Series.rolling(data['Close'], window=window_std).std()))
    return alpha

def alpha_23(data, window=20):
    condition = (pd.Series.rolling(data['High'], window=window).sum() / window) < data['High']
    alpha = if_else(condition, -1 * (data['High'] - pd.Series.shift(data['High'], 2)), 0)
    return alpha

def alpha_24(data, window=100):
    delta_close_100 = (pd.Series.rolling(data['Close'], window=window).sum().diff(window) / pd.Series.shift(data['Close'], window=window))
    condition = (delta_close_100 < 0.05) | (delta_close_100 == 0.05)
    alpha = if_else(condition, -1 * (data['Close'] - data['Close'].expanding(window).min()), -1 * pd.Series.rolling(data['Close'], window=3).diff(3))
    return alpha

def alpha_25(data, adv20=None, vwap=None):
    adv20 = data['Volume'].rolling(window=20).mean() if adv20 is None else adv20
    vwap = (data['High'] + data['Low'] + data['Close']) / 3 if vwap is None else vwap
    returns = -1 * data['Close'].pct_change()
    alpha = pd.Series.rank(((returns * adv20) * vwap * (data['High'] - data['Close'])))
    return alpha

def alpha_26(data, window_corr=5):
    volume_corr = data['Volume'].rolling(window=window_corr).corr(data['High'])
    alpha = -1 * pd.Series.expanding(volume_corr).rolling(3).max()
    return alpha

def alpha_27(data, window_rank=6,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    condition = (0.5 < pd.Series.rank(pd.Series.rolling(pd.Series.rolling(data['Volume'].rank(), window=window_rank).corr(data['VWAP'].rank(), window=window_rank), window=2).sum() / 2))
    alpha = if_else(condition, -1, 1)
    return alpha

def alpha_28(data, window_corr=5):
    alpha = (pd.Series.rolling(data['Volume'].rank().rolling(window=window_corr).corr(data['Low'], window=window_corr), window=5) + ((data['High'] + data['Low']) / 2) - data['Close']).rank()
    return alpha

def alpha_29(data, window1=2, window2=1, window_rank=5, window_delay=6):
    alpha = (pd.Series.rolling(pd.Series.rolling(pd.Series.rank(pd.Series.rank(pd.Series.expanding(np.log(pd.Series.rolling(pd.Series.rolling(pd.Series.rank(pd.Series.rank((-1 * pd.Series.rank(delta(data['Close'] - 1, 5))))), window=2).min(), 1).product())), window=5).rank())), window=1).min(), 5) + pd.Series.rank(pd.Series.shift(-1 * data['Close'].pct_change(), window=window_delay)).rank()
    return alpha

def alpha_30(data, window1=1, window2=5, window_sum=5, window_sum_long=20):
    condition = (pd.Series.rank(pd.Series.rank(pd.Series.rank(((pd.Series.shift(data['Close'], window=window1) - data['Close']).sign() + pd.Series.shift((data['Close'] - pd.Series.shift(data['Close'], window=1)).sign(), window=1)) + (data['Close'] - pd.Series.shift(data['Close'], window=window2)).sign())).rolling(window=5).sum())) / pd.Series.rolling(data['Volume'], window=window_sum).sum() > pd.Series.rolling(data['Volume'], window=window_sum_long).sum()
    alpha = (1 - condition) * pd.Series.rolling(data['Volume'], window=window1).sum() / pd.Series.rolling(data['Volume'], window=window2).sum()
    return alpha

def alpha_31(data, window1=10, window2=3, window_corr=12,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (pd.Series.rank(pd.Series.rank(pd.Series.rank(pd.Series.rolling((1 - pd.Series.rank(pd.Series.rank(delta(data['Close'], window=window1)))), window=10).sum(), window=10)))) + pd.Series.rank(-1 * delta(data['Close'], window=window2)) + np.sign(pd.Series.rolling(data['VWAP'].corr(data['Low'], window=window_corr), window=window_corr))
    return alpha

def alpha_32(data, window1=7, window2=5, window_corr=230,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (pd.Series.rolling((pd.Series.rolling(data['Close'], window=window1).sum() / window1 - data['Close']) + 20 * pd.Series.rolling(data['VWAP'].corr(pd.Series.shift(data['Close'], window=5), window=window_corr), window=5)).rank() + pd.Series.rolling(data['VWAP'].rank(), window=5).rank())
    return alpha

def alpha_33(data):
    alpha = pd.Series.rank((1 - (data['Open'] / data['Close'])) ** 1)
    return alpha

def alpha_34(data, window_std_short=2, window_std_long=5, window_delta=1):
    alpha = (1 - pd.Series.rank((pd.Series.rolling(data['Returns'].std(), window=window_std_short) / pd.Series.rolling(data['Returns'].std(), window=window_std_long)).rank() + (1 - pd.Series.rank(data['Close'].diff(window=window_delta)))))
    return alpha

def alpha_35(data, window1=32, window2=16, window3=32):
    alpha = (pd.Series.rolling(data['Volume'].rank(), window=window1) * (1 - pd.Series.rolling(((data['Close'] + data['High']) - data['Low']), window=window2).rank()) * (1 - pd.Series.rolling(data['Returns'], window=window3).rank()))
    return alpha

def alpha_36(data, window_corr1=15, window_corr2=6, window_rank=5,window_volume=5):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (2.21 * pd.Series.rank(data['Close'].corr(pd.Series.shift(data['Volume'], 1), window=window_corr1)) + 0.7 * pd.Series.rank(data['Open'] - data['Close']) + 0.73 * pd.Series.rank(pd.Series.rolling(delta(data['Returns'], 6), window=5).rank()) + pd.Series.rank(np.abs(data['Close'].corr(data['VWAP'], data['Adv20'], window=window_corr2))) + 0.6 * pd.Series.rank(((pd.Series.rolling(data['Close'], window=200).sum() / 200 - data['Open']) * (data['Close'] - data['Open']))))
    return alpha

def alpha_37(data, window_corr=1, window_rank=200):
    alpha = (pd.Series.rank(pd.Series.rolling(pd.Series.shift(data['Open'] - data['Close'], 1), window=window_corr).corr(data['Close'], window=window_corr)) + pd.Series.rank(data['Open'] - data['Close']))
    return alpha

def alpha_38(data, window_rank1=10, window_rank2=10):
    alpha = (-1 * pd.Series.rank(pd.Series.rolling(data['Close'].rank(), window=window_rank1))) * pd.Series.rank(data['Close'] / data['Open'])
    return alpha

def alpha_39(data, window_delta=7, window_rank=250, window_exp=9):
    alpha = (-1 * pd.Series.rank(delta(data['Close'], window=window_delta) * (1 - pd.Series.rank(pd.Series.expanding(data['Volume'] / data['Adv20'], window=window_exp).diff(window_exp))))) * (1 + pd.Series.rank(pd.Series.rolling(data['Returns'], window=window_rank).sum()))
    return alpha

def alpha_40(data, window_std=10, window_corr=10):
    alpha = (-1 * pd.Series.rank(pd.Series.rolling(data['High'].std(), window=window_std))) * data['High'].corr(data['Volume'], window=window_corr)
    return alpha

def alpha_41(data,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (np.sqrt(data['High'] * data['Low']) - data['VWAP'])
    return alpha

def alpha_42(data,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (pd.Series.rank(data['VWAP'] - data['Close']) / pd.Series.rank(data['VWAP'] + data['Close']))
    return alpha

def alpha_43(data, window_rank1=20, window_rank2=8, window_delta=7):
    alpha = pd.Series.rolling(data['Volume'] / data['Adv20'], window=window_rank1).rank() * pd.Series.rolling(-1 * delta(data['Close'], window=window_delta), window=window_rank2).rank()
    return alpha


def alpha_44(data, window_corr=5):
    alpha = -1 * data['High'].rolling(window=window_corr).corr(data['Volume'].rank())
    return alpha

def alpha_45(data, window1=20, window2=2):
    sum_close_20 = pd.Series.rolling(data['Close'].shift(5), window=window1).sum() / window1
    corr_close_volume = pd.Series.rolling(data['Close'], window=window2).corr(data['Volume'])
    corr_sum_close_5_20 = pd.Series.rolling(data['Close'].rolling(window=5).sum(), window=window2).corr(pd.Series.rolling(data['Close'].rolling(window=20).sum(), window=window2))
    
    alpha = -1 * (sum_close_20.rank() * corr_close_volume.rank() * corr_sum_close_5_20.rank())
    return alpha

def alpha_46(data, multiplier=0.1):
    condition = ((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < (-1 * multiplier)
    alpha = if_else(condition, -1, if_else(((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < 0, 1, -1 * (data['Close'] - data['Close'].shift(1))))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

# Example for alpha_47
def alpha_47(data, multiplier=0.1,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    rank_1_over_close = pd.Series.rank(1 / data['Close'])
    volume_adv20_ratio = (data['Volume'] / pd.Series.rolling(data['Volume'], window=20).mean())
    high_rank = pd.Series.rank(data['High'] - data['Close'])
    sum_high_5_over_5 = pd.Series.rolling(data['High'], window=5).sum() / 5
    rank_vwap_delay = pd.Series.rank(data['VWAP'] - data['VWAP'].shift(5))
    
    alpha = -1 * ((rank_1_over_close * data['Volume'] / data['Volume'].rolling(window=20).mean() * high_rank * (data['High'] - data['Close']) / sum_high_5_over_5) - rank_vwap_delay)
    return alpha


# ... (Previous alpha functions)

# Example for alpha_48
def alpha_48(data, window_corr=250):
    close_delta_1 = delta(data['Close'], period=1)
    close_indneutralize = close_delta_1 / data['Close']
    indneutralize_corr = pd.Series.rolling(close_indneutralize, window=window_corr).corr(data['Volume'])
    
    delta_close_1_sq = delta(data['Close'], period=1) ** 2
    sum_delta_close_1_sq = pd.Series.rolling(delta_close_1_sq, window=window_corr).sum()
    
    alpha = -1 * indneutralize_corr / sum_delta_close_1_sq
    return alpha

# Example for alpha_49
def alpha_49(data, window_delta=7, window_corr=250, window_sum=250):
    close_delta_7 = delta(data['Close'], period=window_delta)
    rank_close_delta_7 = pd.Series.rank(close_delta_7)
    
    decay_linear_volume = pd.Series.rolling(data['Volume'] / pd.Series.rolling(data['Volume'].rolling(window=20).mean(), window=window_corr).mean(), window=window_sum).apply(lambda x: x**(1/window_sum))
    
    alpha = -1 * rank_close_delta_7 * pd.Series.rank(decay_linear_volume)
    return alpha

# Example for alpha_50
def alpha_50(data, window_corr=5,window_volume=5):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    rank_corr = pd.Series.rank(pd.Series.rolling(data['Volume'].rank(), window=window_corr).corr(pd.Series.rolling(data['VWAP'].rank(), window=window_corr)))
    alpha = -1 * pd.Series.rolling(rank_corr, window=5).max()
    return alpha

# Example for alpha_51
def alpha_51(data, multiplier=0.05):
    condition = ((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < (-1 * multiplier)
    alpha = if_else(condition, 1, -1 * (data['Close'] - data['Close'].shift(1)))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

# Example for alpha_52
def alpha_52(data, window1=5, window2=240):
    ts_min_low_5 = pd.Series.rolling(data['Low'], window=5).min()
    ts_min_low_5_delay_5 = ts_min_low_5.shift(5)
    returns_sum_240_20 = pd.Series.rolling(data['Close'].pct_change(), window=240).sum() - pd.Series.rolling(data['Close'].pct_change(), window=20).sum()
    ts_rank_volume_5 = pd.Series.rolling(data['Volume'].rank(), window=5)
    
    alpha = ((ts_min_low_5 - ts_min_low_5_delay_5) * pd.Series.rank(returns_sum_240_20 / 220) * ts_rank_volume_5)
    return alpha

# Example for alpha_53
def alpha_53(data, window=9):
    close_minus_low = data['Close'] - data['Low']
    high_minus_close = data['High'] - data['Close']
    close_minus_low_ratio = close_minus_low / (data['Close'] - data['Low'])
    alpha = -1 * delta(close_minus_low_ratio, period=window)
    return alpha

# Example for alpha_54
def alpha_54(data):
    low_minus_close = data['Low'] - data['Close']
    open_pow_5 = data['Open']**5
    low_high_ratio = (data['Low'] - data['High']) * (data['Close']**5)
    alpha = -1 * (low_minus_close * open_pow_5) / low_high_ratio
    return alpha

# Example for alpha_55
def alpha_55(data, window_low=12, window_high=12, window_volume=6):
    close_ts_min_low_ratio = (data['Close'] - pd.Series.rolling(data['Low'], window=window_low).min()) / (pd.Series.rolling(data['High'], window=window_high).max() - pd.Series.rolling(data['Low'], window=window_low).min())
    rank_close_ts_min_low = pd.Series.rank(close_ts_min_low_ratio)
    rank_volume = pd.Series.rank(data['Volume'])
    alpha = -1 * pd.Series.rolling(rank_close_ts_min_low, window=window_volume).corr(rank_volume)
    return alpha

# Example for alpha_56
def alpha_56(data, window_returns_10=10, window_returns_2=2, window_returns_3=3, cap=None):
    sum_returns_10 = pd.Series.rolling(data['Close'].pct_change(), window=window_returns_10).sum()
    sum_returns_2_3 = pd.Series.rolling(data['Close'].pct_change(), window=window_returns_3).sum() / pd.Series.rolling(pd.Series.rolling(data['Close'].pct_change(), window=window_returns_2).sum(), window=window_returns_3).sum()
    
    cap_multiplier = cap if cap is not None else 1
    alpha = -1 * (pd.Series.rank(sum_returns_10) * pd.Series.rank(sum_returns_2_3) * cap_multiplier)
    return alpha

# Example for alpha_57
def alpha_57(data, window=30,window_volume=6):
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    close_vwap_diff = data['Close'] - data['VWAP']
    decay_linear_max_close = pd.Series.rolling(pd.Series.rank(pd.Series.rolling(close_vwap_diff, window=window).apply(lambda x: x.argmax())), window=2).apply(lambda x: x**(1/2))
    
    alpha = -1 * decay_linear_max_close
    return alpha

# Example for alpha_58
##problem
def alpha_58(data, window_corr=7.89291, window_rank=5.50322,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    vwap_indneutralize_sector = IndNeutralize(data['VWAP'], IndClass.sector)
    corr_indneutralize_volume = pd.Series.rolling(pd.Series.rolling(vwap_indneutralize_sector.corr(data['Volume'], method='pearson'), window=window_corr).apply(lambda x: x**(1/window_corr)), window=window_rank).apply(lambda x: x**(1/window_rank))
    
    alpha = -1 * pd.Series.rank(corr_indneutralize_volume)
    return alpha

# Example for alpha_59
##problem
def alpha_59(data, window_corr=16.2289, window_rank=8.19648, weight_vwap=0.728317,window_volume=5):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    vwap_indneutralize_industry = IndNeutralize(((data['VWAP'] * weight_vwap) + (data['VWAP'] * (1 - weight_vwap))), IndClass.industry)
    corr_indneutralize_volume = pd.Series.rolling(pd.Series.rolling(vwap_indneutralize_industry.corr(data['Volume'], method='pearson'), window=window_corr).apply(lambda x: x**(1/window_corr)), window=window_rank).apply(lambda x: x**(1/window_rank))
    
    alpha = -1 * pd.Series.rank(corr_indneutralize_volume)
    return alpha

# Example for alpha_60
def alpha_60(data, scale_multiplier=2, window_rank=10):
    low_high_ratio = (data['Low'] - data['High']) / (data['High'] - data['Low'])
    volume_rank = pd.Series.rank(data['Volume'])
    
    alpha = -1 * ((scale(pd.Series.rank(low_high_ratio * data['Volume']), scale_multiplier) - scale(pd.Series.rank(pd.Series.rolling(low_high_ratio, window=window_rank).apply(lambda x: x.argmax())), scale_multiplier)))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

# Example for alpha_61
def alpha_61(data, window_ts_min=16.1219,window_volume=5):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    rank_close_ts_min = pd.Series.rank(data['Close'] - pd.Series.rolling(data['Close'], window=window_ts_min).min())
    rank_corr_vwap_adv180 = pd.Series.rank(data['VWAP'].rolling(window=17.9282).corr(data['adv180']))
    
    alpha = if_else(rank_close_ts_min < rank_corr_vwap_adv180, 1, -1)
    return alpha

# ... (rest of the alpha functions)







