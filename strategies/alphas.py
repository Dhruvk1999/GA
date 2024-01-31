import numpy as np
import pandas as pd


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

def alpha2(data, delta_lookback=2, window_corr=6,threshold = 0):
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
        window=window_corr
    ).corr(pd.Series(delta_price_ratio.rank(), name='delta_price_ratio_rank'))
    signal = np.where(signal>threshold,1,0)
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




