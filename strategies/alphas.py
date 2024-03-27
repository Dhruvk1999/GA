import numpy as np
import pandas as pd
from scipy.stats import rankdata



# region Auxiliary functions
def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.to_numpy()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])


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
# Step 1: Compute ranks
    close_rank = data['Close'].rank()
    volume_rank = data['Volume'].rank()

    # Step 2: Calculate rolling covariance between ranks of 'Close' and 'Volume'
    rolling_cov = close_rank.rolling(window=3).cov(volume_rank)

    # Step 3: Apply the transformation based on the formula
    alpha13_values = -1 * rolling_cov
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
    high_rank = data['High'].rank()
    volume_rank = data['Volume'].rank()

    # Step 2: Calculate rolling correlation between ranks of 'High' and 'Volume'
    rolling_corr = high_rank.rolling(window=window_corr).corr(volume_rank)

    # Step 3: Sum correlations over a rolling window
    alpha15_values = -1 * rolling_corr.rolling(window=window_rank).sum()

    return alpha15_values


def alpha16(data, window_covariance=5):
    """
    Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), window_covariance)))
    """
    # Step 1: Compute ranks
    high_rank = data['High'].rank()
    volume_rank = data['Volume'].rank()

    # Step 2: Calculate rolling covariance between ranks of 'High' and 'Volume'
    rolling_cov = high_rank.rolling(window=5).cov(volume_rank)

    # Step 3: Apply the transformation based on the formula
    alpha16_values = -1 * rolling_cov
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

# Utility function for conditional expressions
def if_else(condition, true_value, false_value):
    return np.where(condition, true_value, false_value)

def alpha19(data, window_returns=250, window_sum=250):
    returns = data['Close'].pct_change()
    alpha = -1 * np.sign(((data['Close'] - data['Close'].shift(7)) + pd.Series.rolling(returns, window=window_sum).sum().rank(ascending=False)))
    return alpha

def alpha20(data, window=1):
    open_rank = pd.Series.rank(data['Open'] - data['High'].shift(window))
    close_rank = pd.Series.rank(data['Open'] - data['Close'].shift(window))
    low_rank = pd.Series.rank(data['Open'] - data['Low'].shift(window))
    alpha = -1 * open_rank * close_rank * low_rank
    return alpha

def alpha21(data, window1=8, window2=2, volume=None, adv20=None):
    volume = data['Volume'] if volume is None else volume
    adv20 = data['Volume'].rolling(window=20).mean() if adv20 is None else adv20
    sum_close_8 = pd.Series.rolling(data['Close'], window=window1).sum() / window1
    std_close_8 = pd.Series.rolling(data['Close'], window=window1).std()
    condition = (sum_close_8 + std_close_8) < (pd.Series.rolling(data['Close'], window=window2).sum() / window2)
    alpha = if_else(condition, -1, if_else((pd.Series.rolling(data['Close'], window=window2).sum() / window2) < ((sum_close_8 - std_close_8)), 1, if_else((1 < (volume / adv20)) | ((volume / adv20) == 1), 1, -1)))
    return alpha

def alpha22(data, window_corr=5, window_std=20):
# Assuming 'data' is your DataFrame containing 'High', 'Volume', and 'Close' columns

    # Step 1: Calculate correlation between 'High' and 'Volume'
    correlation_high_volume = data['High'].rolling(window=5).corr(data['Volume'])

    # Step 2: Compute 5-day percentage change of correlation
    delta_correlation = correlation_high_volume.pct_change(periods=window_corr)

    # Step 3: Calculate standard deviation of 'Close' over a 20-day rolling window
    stddev_close = data['Close'].rolling(window=window_std).std()

    # Step 4: Compute rank of the standard deviation
    rank_stddev_close = stddev_close.rank()

    # Step 5: Multiply percentage change of correlation by rank of standard deviation and negate the result
    alpha22_values = -1 * delta_correlation * rank_stddev_close

def alpha23(data, window=20):
    condition = (pd.Series.rolling(data['High'], window=window).sum() / window) < data['High']
    alpha = if_else(condition, -1 * (data['High'] - pd.Series.shift(data['High'], 2)), 0)
    return alpha


def alpha24(close, window=100):
    # Preprocess close series to fill missing values
    close = close.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate the moving average
    sma_close = sma(close, window)
    
    # Calculate the condition
    cond = delta(sma_close, window) / delay(close, window) <= 0.05
    
    # Calculate the alpha values
    alpha = -1 * delta(close, 3)
    alpha[cond] = -1 * (close - ts_min(close, window))
    
    return alpha

def alpha25(data, adv20=None, vwap=None):
    adv20 = data['Volume'].rolling(window=20).mean() if adv20 is None else adv20
    vwap = (data['High'] + data['Low'] + data['Close']) / 3 if vwap is None else vwap
    returns = -1 * data['Close'].pct_change()
    alpha = pd.Series.rank(((returns * adv20) * vwap * (data['High'] - data['Close'])))
    return alpha

# Alpha#26
def alpha26(data, window1=5, window2=5, window3=3):
    rank_volume = data['Volume'].rolling(window=window1).apply(lambda x: pd.Series(x).rank().iloc[-1]).dropna()
    rank_high = data['High'].rolling(window=window2).apply(lambda x: pd.Series(x).rank().iloc[-1]).dropna()
    correlation = rank_volume.rolling(window=window3).corr(rank_high).dropna()
    return -1 * correlation.rolling(window=window3).max()

def alpha27(data, window1=6, window2=2):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=5).sum() / data['Volume'].rolling(window=5).sum()

    correlation_rank = sma(correlation(rank(data['Volume']), rank(data['VWAP']), window1), window2)
    alpha = rank(correlation_rank / 2.0)
    alpha[alpha > 0.5] = -1
    alpha[alpha <= 0.5] = 1
    return alpha

def alpha28(data, window1=20):
    """
    Calculates Alpha#28 according to the formula:

    scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    Parameters:
    data (pd.DataFrame): DataFrame containing columns 'open', 'high', 'close', 'low'
    window1 (int): Window size for the first correlation calculation (default: 20)

    Returns:
    pd.Series: Series containing the Alpha#28 values
    """

    adv20 = data['Volume'].rolling(window=window1).mean()

    # Calculate the average true range (ATR)
    atr = data['High'].ewm(alpha=1 / window1, min_periods=window1).mean() - \
        data['Low'].ewm(alpha=1 / window1, min_periods=window1).mean()

    # Calculate the first correlation term
    corr_term = data['Low'].rolling(window=5).corr(adv20)

    # Calculate the second term
    second_term = (data['High'] + data['Low']) / 2

    # Combine the terms and scale by the ATR
    alpha = ((corr_term + second_term) - data['Close']) / atr

    return alpha

def alpha29(data,window1=5, window2=6):
    returns = data['Close'].pct_change()

    scaled_sum = ts_sum(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((data['Close'] - 1), window1)))), window2))))), window2)
    alpha = ts_min(scaled_sum, window1) + ts_rank(delay((-1 * returns), window2), window1)
    return alpha

# Alpha#30
def alpha30(data, window1=1, window2=1, window3=2, window4=3, window5=5, window6=20):
    sign_changes = ((data['Close'] - data['Close'].shift(window1)).apply(lambda x: 1 if x > 0 else 0) +
                    (data['Close'].shift(window1) - data['Close'].shift(window2)).apply(lambda x: 1 if x > 0 else 0) +
                    (data['Close'].shift(window2) - data['Close'].shift(window3)).apply(lambda x: 1 if x > 0 else 0))
    return ((1.0 - sign_changes.rank()) * data['Volume'].rolling(window=window5).sum() /
            data['Volume'].rolling(window=window6).sum())


def alpha31(data, window1=10, window2=3, window3=12):

    data['Adv20']= data['Volume'].rolling(window=window3).mean()

    delta = data['Close'].diff(window1)
    rank1 = delta.abs().rank(method='min')
    rank2 = rank1.rank(method='min')
    decay = rank2.ewm(alpha=1/window1, min_periods=window1).mean()
    rank3 = decay.rank(method='min')
    alpha = (
        rank3 +
        (-1 * data['Close'].diff(window2)).rank(method='min') +
        pd.Series(np.sign(data['Adv20'].rolling(window=window3).corr(data['Low']))).fillna(0)
    )
    return alpha

def alpha32(data, window1=7, window2=5,window3=230, ):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=5).sum() / data['Volume'].rolling(window=5).sum()

    alpha = (
        pd.Series(((data['Close'].rolling(window=window1).mean() - data['Close']) / data['Close'].std()).fillna(0)) +
        20 * pd.Series(data['VWAP'].rolling(window=window3).corr(data['Close'].shift(window2))).fillna(0)
    )
    return alpha

# Alpha#33
def alpha33(data):
    return (data['Open'] / data['Close']).apply(lambda x: -1 * (1 - x) ** 1).rank()

# Alpha#34
def alpha34(data, window1=2, window2=5):
    returns_stddev2 = data['Close'].pct_change().rolling(window=window1).std()
    returns_stddev5 = data['Close'].pct_change().rolling(window=window2).std()
    return ((1 - returns_stddev2.rank()) + (1 - returns_stddev5.rank()) +
            (1 - (data['Close'].diff(1)).rank()))

def alpha35(data, window1=32, window2=16, window3=32):
    """
    Calculates Alpha#35 according to the formula:

    ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -
    Ts_Rank(returns, 32)))

    Parameters:
    data (pd.DataFrame): DataFrame containing columns 'close', 'high', 'low', 'volume', 'returns'
    window1 (int): Window size for volume rank (default: 32)
    window2 (int): Window size for price change rank (default: 16)
    window3 (int): Window size for returns rank (default: 32)

    Returns:
    pd.Series: Series containing the Alpha#35 values
    """
    data['returns'] = data['Close'].pct_change()

    alpha = (
        data['Volume'].rank(method='average', ascending=False).rolling(window=window1).mean() *
        (1 - ((data['Close'] + data['High']) - data['Low']).rank(method='average', ascending=False).rolling(window=window2).mean()) *
        (1 - data['returns'].rank(method='average', ascending=False).rolling(window=window3).mean())
    )
    return alpha


def alpha36(data, window_corr1=15, window_corr2=6, window_rank=5,window_volume=5):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (2.21 * pd.Series.rank(data['Close'].corr(pd.Series.shift(data['Volume'], 1), window=window_corr1)) + 0.7 * pd.Series.rank(data['Open'] - data['Close']) + 0.73 * pd.Series.rank(pd.Series.rolling(delta(data['Returns'], 6), window=5).rank()) + pd.Series.rank(np.abs(data['Close'].corr(data['VWAP'], data['Adv20'], window=window_corr2))) + 0.6 * pd.Series.rank(((pd.Series.rolling(data['Close'], window=200).sum() / 200 - data['Open']) * (data['Close'] - data['Open']))))
    return alpha

def alpha37(data, window=200):
    """
    Calculates Alpha#37 according to the formula:

    (rank(correlation(delay((Open - Close), 1), Close, 200)) + rank((Open - Close)))

    Parameters:
    data (pd.DataFrame): DataFrame containing columns 'Open', 'Close'
    window (int): Window size for correlation and rank (default: 200)

    Returns:
    pd.Series: Series containing the Alpha#37 values
    """

    shifted_diff = data['Open'].diff().shift(1)
    corr_term = shifted_diff.rolling(window=window).corr(data['Close']).rank(method='min')  # Calculate correlation within the rolling window
    open_close_term = (data['Open'] - data['Close']).rank(method='min')

    alpha = corr_term + open_close_term

    return alpha

def alpha38(data, window=10):

    close_rank = data['Close'].rank(method='min', ascending=False).rolling(window=window).mean()
    close_open_ratio = data['Close'] / data['Open']
    close_open_rank = close_open_ratio.rank(method='min')

    alpha = -1 * close_rank * close_open_rank

    return alpha

def alpha39(data, window1=7, window2=9, window3=250):


    data['Returns'] = data['Close'].pct_change()

    close_rank = data['Close'].rank(method='min', ascending=False).rolling(window=window1).mean()
    close_open_ratio = data['Close'] / data['Open']
    close_open_rank = close_open_ratio.rank(method='min')

    decay_term = close_open_rank.ewm(alpha=1/window2, min_periods=window2).mean()
    decay_rank = decay_term.rank(method='min')

    returns_sum = data['Returns'].rolling(window=window3).sum()
    returns_rank = returns_sum.rank(method='min')

    alpha = -1 * close_rank * decay_rank * (1 + returns_rank)

    return alpha

def alpha40(data, window=10):

    stddev_term = data['High'].rolling(window=window).std().rank(method='min')
    corr_term = data['High'].rolling(window=window).corr(data['Volume']).rank(method='min')
    alpha = -1 * stddev_term * corr_term

    return alpha

def alpha41(data,window_volume=6):

    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    alpha = (np.sqrt(data['High'] * data['Low']) - data['VWAP'])

    return alpha

def alpha42(data,window_volume=6):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

    alpha = (pd.Series.rank(data['VWAP'] - data['Close']) / pd.Series.rank(data['VWAP'] + data['Close']))
    return alpha

def alpha43(data, window1=20, window2=8):
    # Calculate ADV
    adv20 = data['Volume'].rolling(window=20).mean()
    # Calculate volume/adv20 ratio
    volume_ratio = data['Volume'] / adv20
    # Calculate delta close
    delta_close = data['Close'].diff(window2)
    # Rank volume/adv20 and negative delta close
    volume_ratio_rank = volume_ratio.rank(method='average', ascending=False).rolling(window=window1).mean()
    delta_close_rank = (-1 * delta_close).rank(method='average', ascending=False).rolling(window=window2).mean()

    # Calculate alpha
    alpha = volume_ratio_rank * delta_close_rank

    return alpha


def alpha44(data, window_corr=5):
    alpha = -1 * data['High'].rolling(window=window_corr).corr(data['Volume'].rank())
    return alpha

def alpha45(data, window1=5, window2=20, window3=2):


    delayed_close_sum = data['Close'].shift(window1).rolling(window=window2).sum()
    close_volume_corr = data['Close'].rolling(window=window2).corr(data['Volume'])
    short_close_sum = data['Close'].rolling(window=window1).sum()
    long_close_sum = data['Close'].rolling(window=window2).sum()
    short_long_corr = short_close_sum.rolling(window=window3).corr(long_close_sum)

    alpha = -1 * ((delayed_close_sum.rank(method='min') * close_volume_corr.rank(method='min')) * short_long_corr.rank(method='min'))

    return alpha

def alpha46(data, multiplier=0.1):
    condition = ((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < (-1 * multiplier)
    alpha = if_else(condition, -1, if_else(((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < 0, 1, -1 * (data['Close'] - data['Close'].shift(1))))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

# Example for alpha47
def alpha47(data,window_volume=6):
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

# Example for alpha48
def alpha48(data, window_corr=250):
    close_delta_1 = delta(data['Close'], period=1)
    close_indneutralize = close_delta_1 / data['Close']
    indneutralize_corr = pd.Series.rolling(close_indneutralize, window=window_corr).corr(data['Volume'])
    
    delta_close_1_sq = delta(data['Close'], period=1) ** 2
    sum_delta_close_1_sq = pd.Series.rolling(delta_close_1_sq, window=window_corr).sum()
    
    alpha = -1 * indneutralize_corr / sum_delta_close_1_sq
    return alpha

# Example for alpha49
def alpha49(data, window_delta=7, window_corr=250, window_sum=250):
    close_delta_7 = delta(data['Close'], period=window_delta)
    rank_close_delta_7 = pd.Series.rank(close_delta_7)
    
    decay_linear_volume = pd.Series.rolling(data['Volume'] / pd.Series.rolling(data['Volume'].rolling(window=20).mean(), window=window_corr).mean(), window=window_sum).apply(lambda x: x**(1/window_sum))
    
    alpha = -1 * rank_close_delta_7 * pd.Series.rank(decay_linear_volume)
    return alpha

def alpha50(data, window1=5, window2=5):
    vwap = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
    vwap = vwap.cumsum() / data['Volume'].cumsum()
    volume_rank = data['Volume'].rank(method='min')
    vwap_rank = vwap.rank(method='min')
    corr_term = volume_rank.rolling(window=window1).corr(vwap_rank)
    corr_rank = corr_term.rank(method='min')
    max_rank = corr_rank.rolling(window=window2).max()
    alpha = -1 * max_rank

    return alpha

# Example for alpha51
def alpha51(data, multiplier=0.05):
    condition = ((data['Close'].shift(20) - data['Close'].shift(10)) / 10 - ((data['Close'].shift(10) - data['Close']) / 10)) < (-1 * multiplier)
    alpha = if_else(condition, 1, -1 * (data['Close'] - data['Close'].shift(1)))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

def alpha52(data, window1=5, window2=240, window3=20, window4=30):

    min_low = data['Low'].rolling(window=window1).min()

    delayed_min_low = min_low.shift(window1)

    returns_diff = (data['Returns'].rolling(window=window2).sum() -
                    data['Returns'].rolling(window=window3).sum()) / window3

    returns_diff_rank = returns_diff.rank(method='min')
    volume_rank = data['Volume'].rank(method='min').rolling(window=window4).mean()
    alpha = (((-1 * min_low + delayed_min_low) * returns_diff_rank) * volume_rank)

    return alpha


# Example for alpha53
def alpha53(data, window=9):
    close_minus_low = data['Close'] - data['Low']
    high_minus_close = data['High'] - data['Close']
    close_minus_low_ratio = close_minus_low / (data['Close'] - data['Low'])
    alpha = -1 * delta(close_minus_low_ratio, period=window)
    return alpha

# Example for alpha54
def alpha54(data):
    low_minus_close = data['Low'] - data['Close']
    open_pow_5 = data['Open']**5
    low_high_ratio = (data['Low'] - data['High']) * (data['Close']**5)
    alpha = -1 * (low_minus_close * open_pow_5) / low_high_ratio
    return alpha

# Example for alpha55
def alpha55(data, window_low=12, window_high=12, window_volume=6):
    close_ts_min_low_ratio = (data['Close'] - pd.Series.rolling(data['Low'], window=window_low).min()) / (pd.Series.rolling(data['High'], window=window_high).max() - pd.Series.rolling(data['Low'], window=window_low).min())
    rank_close_ts_min_low = pd.Series.rank(close_ts_min_low_ratio)
    rank_volume = pd.Series.rank(data['Volume'])
    alpha = -1 * pd.Series.rolling(rank_close_ts_min_low, window=window_volume).corr(rank_volume)
    return alpha

# Example for alpha56
def alpha56(data, window_returns_10=10, window_returns_2=2, window_returns_3=3, cap=None):
    sum_returns_10 = pd.Series.rolling(data['Close'].pct_change(), window=window_returns_10).sum()
    sum_returns_2_3 = pd.Series.rolling(data['Close'].pct_change(), window=window_returns_3).sum() / pd.Series.rolling(pd.Series.rolling(data['Close'].pct_change(), window=window_returns_2).sum(), window=window_returns_3).sum()
    
    cap_multiplier = cap if cap is not None else 1
    alpha = -1 * (pd.Series.rank(sum_returns_10) * pd.Series.rank(sum_returns_2_3) * cap_multiplier)
    return alpha



def alpha57(close, vwap, window=30):
    """
    Alpha#057
    Formula:
    0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)).to_frame(), 2).CLOSE))
    """
    # Calculate the moving maximum of close
    max_close = ts_argmax(close, window)
    # Calculate the linear decay of the rank of the maximum close
    decay = decay_linear(rank(max_close).to_frame(), 2).CLOSE
    # Calculate the alpha
    alpha = 0 - (1 * ((close - vwap) / decay))
    return alpha

# Example for alpha58
##problem
# def alpha58(data, window_corr=7.89291, window_rank=5.50322,window_volume=6):
#     typical_price = (data['High'] + data['Low'] + data['Close']) / 3
#     data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

#     vwap_indneutralize_sector = IndNeutralize(data['VWAP'], IndClass.sector)
#     corr_indneutralize_volume = pd.Series.rolling(pd.Series.rolling(vwap_indneutralize_sector.corr(data['Volume'], method='pearson'), window=window_corr).apply(lambda x: x**(1/window_corr)), window=window_rank).apply(lambda x: x**(1/window_rank))
    
#     alpha = -1 * pd.Series.rank(corr_indneutralize_volume)
#     return alpha

# Example for alpha59
##problem
# def alpha59(data, window_corr=16.2289, window_rank=8.19648, weight_vwap=0.728317,window_volume=5):
#     typical_price = (data['High'] + data['Low'] + data['Close']) / 3
#     data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()

#     vwap_indneutralize_industry = IndNeutralize(((data['VWAP'] * weight_vwap) + (data['VWAP'] * (1 - weight_vwap))), IndClass.industry)
#     corr_indneutralize_volume = pd.Series.rolling(pd.Series.rolling(vwap_indneutralize_industry.corr(data['Volume'], method='pearson'), window=window_corr).apply(lambda x: x**(1/window_corr)), window=window_rank).apply(lambda x: x**(1/window_rank))
    
#     alpha = -1 * pd.Series.rank(corr_indneutralize_volume)
#     return alpha

# Example for alpha60
def alpha60(data, scale_multiplier=2, window_rank=10):
    low_high_ratio = (data['Low'] - data['High']) / (data['High'] - data['Low'])
    volume_rank = pd.Series.rank(data['Volume'])
    
    alpha = -1 * ((scale(pd.Series.rank(low_high_ratio * data['Volume']), scale_multiplier) - scale(pd.Series.rank(pd.Series.rolling(low_high_ratio, window=window_rank).apply(lambda x: x.argmax())), scale_multiplier)))
    return alpha

# Continue with the rest of the alpha functions using a similar structure

# Example for alpha61
def alpha61(data, window1=16, window2=17):

    adv180 = data['Volume'].rolling(window=180).mean()
    vwap = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
    vwap = vwap.cumsum() / data['Volume'].cumsum()
    min_vwap = vwap.rolling(window=window1).min()
    corr_term = vwap.rolling(window=window2).corr(adv180)
    vwap_delta_rank = (vwap - min_vwap).rank(method='min')
    corr_rank = corr_term.rank(method='min')
    alpha = np.where(vwap_delta_rank < corr_rank,1,0)

    return alpha
# ... (rest of the alpha functions)



import pandas as pd

def alpha102(data, fast_period=2, slow_period=10):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Parameters:
    - data: Pandas DataFrame with 'Close' prices.
    - fast_period: Fast period for MACD calculation (default: 12).
    - slow_period: Slow period for MACD calculation (default: 26).

    Returns:
    - Pandas Series containing the MACD values.
    """
    if(fast_period>slow_period):
        slow_period,fast_period = fast_period,slow_period
    # Calculate the short-term (fast) EMA
    fast_ema = data['Close'].ewm(span=fast_period, min_periods=1, adjust=False).mean()

    # Calculate the long-term (slow) EMA
    slow_ema = data['Close'].ewm(span=slow_period, min_periods=1, adjust=False).mean()

    # Calculate the MACD line
    macd_line = fast_ema - slow_ema

    return macd_line

def alpha62(data, window1=9, window2=22):

    vwap = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
    vwap = vwap.cumsum() / data['Volume'].cumsum()
    adv20 = data['Volume'].rolling(window=20).mean()
    corr_term = vwap.rolling(window=window1).corr(adv20.rolling(window=window2).sum())
    corr_rank = corr_term.rank(method='min')
    open_rank = (data['Open'].rank(method='min') + data['Open'].rank(method='min'))
    mid_rank = ((data['High'] + data['Low']) / 2).rank(method='min')
    high_rank = data['High'].rank(method='min')
    combined_rank = (open_rank < (mid_rank + high_rank))
    combined_rank_rank = combined_rank.rank(method='min')
    alpha = (corr_rank < combined_rank_rank) * -1

    return alpha


# def alpha63(data, window_industry_delta=2, window_vwap_adv=38):
#     close_neutralized = data['Close'] - data.groupby('Industry')['Close'].transform('mean')
#     decay_linear_close_neutralized = close_neutralized.diff(window_industry_delta).fillna(0).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / window_industry_delta) for i in range(window_industry_delta)]))
#     weighted_vwap_open = data['VWAP'] * 0.318108 + data['Open'] * (1 - 0.318108)
#     correlation_result = weighted_vwap_open.rolling(window_vwap_adv).corr(data['Adv180'].rolling(window_vwap_adv).sum())
#     decay_linear_correlation = correlation_result.rolling(window=13).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 14) for i in range(13)])).fillna(0)
#     rank_1 = decay_linear_close_neutralized.rank()
#     rank_2 = decay_linear_correlation.rank()
#     alphavalue = (rank_1 - rank_2) * -1
#     return alphavalue

def alpha64(data, window1=12, window2=16, window3=3):
    
    vwap = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
    vwap = vwap.cumsum() / data['Volume'].cumsum()
    adv120 = data['Volume'].rolling(window=120).mean()
    weighted_sum = (data['Open'] * 0.178404) + (data['Low'] * (1 - 0.178404))
    weighted_sum_sum = weighted_sum.rolling(window=window1).sum()
    corr_term = weighted_sum_sum.rolling(window=window2).corr(adv120.rolling(window=window2).sum())
    corr_rank = corr_term.rank(method='min')
    mid_price = ((data['High'] + data['Low']) / 2) * 0.178404
    delta = mid_price.diff(periods=1) - vwap.diff(periods=1)
    delta_rank = delta.rank(method='min')
    alpha = (corr_rank < delta_rank) * -1

    return alpha

def alpha65(data, window_volume=9, window_sum_open_vwap=9, window_min_open=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    
    # Compute Adv60
    data['Adv60'] = data['Volume'].rolling(window=60).mean()
    
    sum_open_vwap = (data['Open'] * 0.00817205 + data['VWAP'] * (1 - 0.00817205)).rolling(window=window_sum_open_vwap).sum()
    sum_adv60 = data['Adv60'].rolling(window=60).sum()
    
    rank_correlation = sum_open_vwap.rolling(window=6).corr(sum_adv60).rank()
    rank_min_open = data['Open'].rolling(window=window_min_open).min().rank()
    
    alphavalue = (rank_correlation < rank_min_open) * -1
    return alphavalue

def alpha66(data, window_volume = 9,window_delta_vwap=4):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    
    delta_vwap = data['VWAP'].diff(window_delta_vwap)
    decay_linear_delta_vwap = delta_vwap.rolling(window=7).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 7) for i in range(7)])).fillna(0)
    delta_ratio = ((data['Low'] * 0.96633 + data['Low'] * (1 - 0.96633)) - data['VWAP']) / (data['Open'] - ((data['High'] + data['Low']) / 2))
    decay_linear_delta_ratio = delta_ratio.rolling(window=11).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 11) for i in range(11)])).fillna(0)
    rank_1 = decay_linear_delta_vwap.rank()
    rank_2 = decay_linear_delta_ratio.rank()
    alphavalue = (rank_1 + rank_2) * -1
    return alphavalue

# def alpha67(data, window_min_high=2, window_correlation=6):
#     ts_min_high = data['High'].rolling(window=window_min_high).min()
#     rank_high = (data['High'] - ts_min_high).rank()
#     correlation_sector_subindustry = data.groupby('Sector')['VWAP'].transform(lambda x: x.corr(data.groupby('Subindustry')['Adv20'].transform('mean')))
#     rank_correlation = rank_high.rolling(window=6).corr(correlation_sector_subindustry.rank())
#     alphavalue = (rank_high ** rank_correlation) * -1
#     return alphavalue

def alpha68(data,window_volume=9 ,window_correlation=9, window_delta=(0.518371+1.06157)):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    data['Adv15'] = data['Volume'].rolling(window=15).mean()

    rank_high = data['High'].rank()
    rank_adv15 = data['Adv15'].rank()
    correlation_rank = rank_high.rolling(window=window_correlation).corr(rank_adv15)
    delta_value = ((data['Close'] * 0.518371 + data['Low'] * (1 - 0.518371)) - data['VWAP']).diff(window_delta)
    rank_delta = delta_value.rank()
    alphavalue = (correlation_rank < rank_delta) * -1
    return alphavalue

# def alpha69(data, window_industry_delta=3, window_correlation=5, window_delta=4):
#     delta_vwap_industry = data.groupby('Industry')['VWAP'].diff(window_industry_delta)
#     rank_max_delta_vwap = delta_vwap_industry.rolling(window=window_delta).max().rank()
#     correlation_close_vwap = (data['Close'] * 0.490655 + data['VWAP'] * (1 - 0.490655)).rolling(window_correlation).corr(data['Adv20'])
#     rank_correlation = correlation_close_vwap.rolling(window=window_correlation).rank()
#     alphavalue = (rank_max_delta_vwap ** rank_correlation) * -1
#     return alphavalue

# def alpha70(data, window_delta=1.29456, window_correlation=17.8256):
#     delta_vwap = data['VWAP'].diff(window_delta)
#     rank_delta_vwap = delta_vwap.rank()
#     correlation_close_industry = data.groupby('Industry')['Close'].transform(lambda x: x.corr(data['Adv50']))
#     rank_correlation = correlation_close_industry.rolling(window=window_correlation).rank()
#     alphavalue = (rank_delta_vwap ** rank_correlation) * -1
#     return alphavalue


def alpha71(data, window1=3, window2=12, window3=18, window4=16):
    adv180 = sma(data['Volume'], 180)
    p1 = ts_rank(decay_linear(correlation(ts_rank(data['Close'], window1), ts_rank(adv180, window2), window3).to_frame(), 4).CLOSE, window4)
    p2 = ts_rank(decay_linear((rank(((data['Low'] + data['Open']) - (data['VWAP'] + data['VWAP']))).pow(2)).to_frame(), 16).CLOSE, 4)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df['max'] = np.where(df['p1'] >= df['p2'], df['p1'], df['p2'])
    return df['max']

def alpha72(data, window1=40, window2=9, window3=10, window4=7, window5=4):
    adv40 = sma(data['Volume'], window1)
    return (rank(decay_linear(correlation(((data['High'] + data['Low']) / 2), adv40, window2).to_frame(), window3).CLOSE) /
            rank(decay_linear(correlation(ts_rank(data['VWAP'], window5), ts_rank(data['Volume'], 19), window4).to_frame(), 3).CLOSE))

def alpha73(data, window1=5, window2=2, window3=3, window4=17):
    p1 = rank(decay_linear(delta(data['VWAP'], window1).to_frame(), window2).CLOSE)
    p2 = ts_rank(decay_linear(((delta(((data['Open'] * 0.147155) + (data['Low'] * (1 - 0.147155))), 2) / 
                                         ((data['Open'] * 0.147155) + (data['Low'] * (1 - 0.147155)))) * -1).to_frame(), window3).CLOSE, window4)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df['max'] = np.where(df['p1'] >= df['p2'], df['p1'], df['p2'])
    return -1 * df['max']

def alpha74(data,window_volume=9, window_sum_adv30=37, window_correlation_volume=30):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    data['Adv30'] = data['Volume'].rolling(window=30).mean()

    correlation_close_adv30 = data['Close'].rolling(window=window_correlation_volume).corr(data['Adv30'].rolling(window=window_sum_adv30).sum())
    rank_high_vwap = ((data['High'] * 0.0261661 + data['VWAP'] * (1 - 0.0261661))).rank()
    rank_volume = data['Volume'].rank()
    correlation_rank_high_vwap_volume = rank_high_vwap.rolling(window=11).corr(rank_volume)
    alphavalue = (correlation_close_adv30.rank() < correlation_rank_high_vwap_volume.rank()) * -1
    return alphavalue

def alpha75(data,window_volume=9, window_adv=50,window_correlation_vwap_volume=4, window_correlation_low_adv50=12):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    data['Adv50'] = data['Volume'].rolling(window=window_adv).mean()

    correlation_vwap_volume = data['VWAP'].rolling(window=window_correlation_vwap_volume).corr(data['Volume'])
    rank_low = data['Low'].rank()
    rank_adv50 = data['Adv50'].rank()
    correlation_rank_low_adv50 = rank_low.rolling(window=window_correlation_low_adv50).corr(rank_adv50)
    alphavalue = (correlation_vwap_volume.rank() < correlation_rank_low_adv50.rank()) * -1
    return alphavalue

def alpha76(data, window_volume=9,window_delta_vwap=1, window_decay_correlation=19 ,window_decay=17):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_volume).sum() / data['Volume'].rolling(window=window_volume).sum()
    delta_vwap = data['VWAP'].diff(window_delta_vwap)
    decay_linear_delta_vwap = delta_vwap.rolling(window=11.8259).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 11) for i in range(11)])).fillna(0)
    correlation_low_sector_adv81 = data.groupby('Sector')['Low'].transform(lambda x: x.corr(data['Adv81']))
    correlation_rank_low_sector_adv81 = correlation_low_sector_adv81.rolling(window=window_decay_correlation).rank()
    decay_correlation = correlation_rank_low_sector_adv81.rolling(window=window_decay).apply(lambda x: np.sum(x)).fillna(0)
    alphavalue = max(decay_linear_delta_vwap.rank(), correlation_rank_low_sector_adv81.rolling(window=window_decay_correlation).rank().rolling(window=window_decay).apply(lambda x: np.sum(x) / decay_correlation).rank()) * -1
    return alphavalue


def alpha77(data, window1=20, window2=6):
    adv40 = data['Volume'].rolling(window=40).mean()
    p1 = rank(decay_linear((((data['High'] + data['Low']) / 2) + data['High'] - (data['VWAP'] + data['High'])).to_frame(), window1).CLOSE)
    p2 = rank(decay_linear(correlation(((data['High'] + data['Low']) / 2), adv40, 3).to_frame(), window2).CLOSE)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df['min'] = np.where(df['p1'] >= df['p2'], df['p2'], df['p1'])
    df['min'] = np.where(df['p2'] >= df['p1'], df['p1'], df['p2'])
    return df['min']

def alpha78(data):
    adv40 = data['Volume'].rolling(window=40).mean()
    return (rank(correlation(ts_sum(((data['Low'] * 0.352233) + (data['VWAP'] * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7))
            .pow(rank(correlation(rank(data['VWAP']), rank(data['Volume']), 6))))


def alpha79(data, window_sector_delta=1.23438, window_correlation_vwap_adv150=14.6644):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_sector_delta).sum() / data['Volume'].rolling(window=window_sector_delta).sum()
    delta_sector_neutralized = data.groupby('Sector')['Close'].transform(lambda x: x.diff(window_sector_delta))
    correlation_vwap_adv150 = data['VWAP'].rolling(window=3.60973).corr(data['Adv150'].rolling(window=9.18637))
    correlation_rank_vwap_adv150 = correlation_vwap_adv150.rolling(window=window_correlation_vwap_adv150).rank()
    alphavalue = delta_sector_neutralized.rank() < correlation_rank_vwap_adv150.rank()
    return alphavalue

def alpha80(data, window_industry_delta=4.04545, window_correlation_high_adv10=5.11456):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=window_industry_delta).sum() / data['Volume'].rolling(window=window_industry_delta).sum()
    delta_open_industry = data.groupby('Industry')['Open'].transform(lambda x: x.diff(1))
    rank_signed_delta_open_industry = np.sign(delta_open_industry).rank()
    correlation_high_adv10 = data['High'].rolling(window=5.11456).corr(data['Adv10'])
    rank_correlation_high_adv10 = correlation_high_adv10.rank()
    alphavalue = (rank_signed_delta_open_industry ** rank_correlation_high_adv10) * -1
    return alphavalue

def alpha81(data, window1=10, window2=50, window3=8, window4=15, window5=5):
    adv10 = data['Volume'].rolling(window=window1).mean()
    return ((rank(np.log(product(rank((rank(correlation(data['VWAP'], ts_sum(adv10, window2), window3))).pow(4)), window4))) < 
             rank(correlation(rank(data['VWAP']), rank(data['Volume']), window5))) * -1)

def alpha82(data, window_delta_open=1.46063, window_decay_correlation=17.4842, window_ts_rank=13.4283):
    delta_open = data['Open'].diff(window_delta_open)
    decay_linear_delta_open = delta_open.rolling(window=14.8717).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 14) for i in range(14)])).fillna(0)
    correlation_volume_sector = data.groupby('Sector')['Volume'].transform(lambda x: x.corr(((data['Open'] * 0.634196) + (data['Open'] * (1 - 0.634196)))))
    decay_linear_correlation_volume_sector = correlation_volume_sector.rolling(window=window_decay_correlation).apply(lambda x: np.sum(x)).fillna(0)
    ts_rank_decay_correlation_volume_sector = decay_linear_correlation_volume_sector.rolling(window=window_ts_rank).apply(lambda x: np.sum(x)).rank()
    alphavalue = min(decay_linear_delta_open.rank(), ts_rank_decay_correlation_volume_sector) * -1
    return alphavalue

def alpha83(data):
    return ((rank(delay(((data['High'] - data['Low']) / (ts_sum(data['Close'], 5) / 5)), 2)) * rank(rank(data['Volume']))) /
            (((data['High'] - data['Low']) / (ts_sum(data['Close'], 5) / 5)) / (data['VWAP'] - data['Close'])))

def alpha84(data):
    return np.power(ts_rank((data['VWAP'] - ts_max(data['VWAP'], 15)), 21), delta(data['Close'], 5))

def alpha85(data, window1=30, window2=10, window3=4, window4=10, window5=7):
    adv30 = data['Volume'].rolling(window=window1).mean()
    return (rank(correlation(((data['High'] * 0.876703) + (data['Close'] * (1 - 0.876703))), adv30, window2)).pow(
            rank(correlation(ts_rank(((data['High'] + data['Low']) / 2), window3), ts_rank(data['Volume'], window4), window5))))

def alpha86(data):
    adv20 = data['Volume'].rolling(window=20).mean()
    return ((ts_rank(correlation(data['Close'], sma(adv20, 15), 6), 20) < 
             rank(((data['Open'] + data['Close']) - (data['VWAP'] + data['Open'])))) * -1)


def alpha87(data, window_corr=13, window_delta_close_vwap=1.91233, window_ts_rank_decay_correlation_close=14):
    delta_close_vwap = (data['Close'] * 0.369701 + data['VWAP'] * (1 - 0.369701)).diff(window_delta_close_vwap)
    decay_linear_delta_close_vwap = delta_close_vwap.rolling(window=2.65461).apply(lambda x: np.sum([x.iloc[i] * (1 - (i+1) / 2) for i in range(2)])).fillna(0)
    correlation_adv81_close = data.groupby('Industry')['Close'].transform(lambda x: x.corr(data['Adv81']))
    abs_correlation_adv81_close = correlation_adv81_close.abs()
    decay_linear_abs_correlation_adv81_close = abs_correlation_adv81_close.rolling(window=window_corr).apply(lambda x: np.sum(x)).fillna(0)
    ts_rank_decay_correlation_abs_adv81_close = decay_linear_abs_correlation_adv81_close.rolling(window=window_ts_rank_decay_correlation_close).rank()
    alphavalue = max(decay_linear_delta_close_vwap.rank(), ts_rank_decay_correlation_abs_adv81_close) * -1
    return alphavalue

def alpha88(data, window1=8, window2=21, window3=8, window4=7):
    """
    Alpha#088
    Formula:
    (min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))).to_frame(),8).CLOSE),
         ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60,21), 8).to_frame(), 7).CLOSE, 3)))
    """
    # Calculate the moving average of volume
    adv60 = sma(data['Volume'], 60)
    # Calculate the difference between the ranks of open, low, high, and close
    diff_rank = ((rank(data['Open']) + rank(data['Low'])) - (rank(data['High']) + rank(data['Close']))).to_frame()
    # Calculate the linear decay of the difference ranks
    decay_diff_rank = decay_linear(diff_rank, 8).CLOSE
    # Calculate the correlation between the rank of close and the rank of adv60
    corr_rank = correlation(ts_rank(data['Close'], 8), ts_rank(adv60, 21), 8).to_frame()
    # Calculate the linear decay of the correlation ranks
    decay_corr_rank = decay_linear(corr_rank, 7).CLOSE
    # Calculate the minimum of the decayed difference rank and the rank of decayed correlation
    min_alpha = pd.concat([rank(decay_diff_rank), ts_rank(decay_corr_rank, 3)], axis=1).min(axis=1)
    return min_alpha

# def alpha89(data, window_correlation_adv10=6.94279, window_delta_vwap=3.48158, window_ts_rank=15.3012):
#     correlation_low_adv10 = (data['Low'] * 0.967285 + data['Low'] * (1 - 0.967285)).rolling(window=window_correlation_adv10).corr(data['Adv10'])
#     ts_rank_correlation_low_adv10 = correlation_low_adv10.rolling(window=5.51607).rank().rolling(window=3.79744).apply(lambda x: np.sum(x)).fillna(0)
#     delta_vwap_sector_neutralized = data.groupby('Industry')['VWAP'].transform(lambda x: x.diff(window_delta_vwap))
#     ts_rank_decay_delta_vwap_sector_neutralized = delta_vwap_sector_neutralized.rolling(window=window_ts_rank).apply(lambda x: np.sum(x)).rank()
#     alphavalue = ts_rank_correlation_low_adv10 - ts_rank_decay_delta_vwap_sector_neutralized
#     return alphavalue

# def alpha90(data, window_ts_max_close=4.66719, window_correlation_adv40_low=5.38375):
#     ts_max_close = data['Close'].rolling(window=window_ts_max_close).max()
#     ts_rank_correlation_adv40_low = data.groupby('Subindustry')['Adv40'].rolling(window=window_correlation_adv40_low).rank().rolling(window=3.21856).apply(lambda x: np.sum(x)).fillna(0)
#     alphavalue = ((data['Close'] - ts_max_close).rank() ** ts_rank_correlation_adv40_low) * -1
#     return alphavalue

# def alpha91(data, window_correlation_volume_sector=9.74928, window_ts_rank_adv30=4.01303):
#     correlation_volume_sector = data.groupby('Industry')['Volume'].transform(lambda x: x.corr(data['Close']))
#     decay_linear_correlation_volume_sector = correlation_volume_sector.rolling(window=window_correlation_volume_sector).apply(lambda x: np.sum(x)).fillna(0)
#     ts_rank_decay_correlation_volume_sector = decay_linear_correlation_volume_sector.rolling(window=4.8667).rank().rolling(window=3.83219).apply(lambda x: np.sum(x)).fillna(0)
#     correlation_vwap_adv30 = data['VWAP'].rolling(window=4.01303).corr(data['Adv30'])
#     rank_decay_linear_correlation_vwap_adv30 = correlation_vwap_adv30.rolling(window=2.6809).rank()
#     alphavalue = (ts_rank_decay_correlation_volume_sector - rank_decay_linear_correlation_vwap_adv30) * -1
#     return alphavalue

def alpha92(data, window1=30, window2=15, window3=19, window4=8, window5=7):
    adv30 = data['Volume'].rolling(window=window1).mean()
    p1 = ts_rank(decay_linear(((((data['High'] + data['Low']) / 2) + data['Close']) < (data['Low'] + data['Open'])).to_frame(), window2).CLOSE, window3)
    p2 = ts_rank(decay_linear(correlation(rank(data['Low']), rank(adv30), window4).to_frame(), window5).CLOSE, window5)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df['min'] = np.where(df['p1'] >= df['p2'], df['p2'], df['p1'])
    df['min'] = np.where(df['p2'] >= df['p1'], df['p1'], df['p2'])
    return df['min']

def alpha93(data, window_correlation_vwap_adv81=17.4193, window_decay_linear_delta_close_vwap=20.4523, window_rank=16.2664):
    correlation_vwap_adv81 = data['VWAP'].rolling(window=window_correlation_vwap_adv81).corr(data['Adv81'])
    decay_linear_delta_close_vwap = ((data['Close'] * 0.524434) + (data['VWAP'] * (1 - 0.524434))).diff().rolling(window=window_decay_linear_delta_close_vwap).apply(lambda x: np.sum(x)).fillna(0)
    alphavalue = (correlation_vwap_adv81.rolling(window=19.848).rank().rolling(window=7.54455).apply(lambda x: np.sum(x)).fillna(0) / decay_linear_delta_close_vwap.rank()) * -1
    return alphavalue

def alpha94(data, window1=12, window2=20, window3=4, window4=18):
    """
    Alpha#094
    Formula:
    (rank((vwap - ts_min(vwap, 12))).pow(ts_rank(correlation(ts_rank(vwap,20), ts_rank(adv60, 4), 18), 3)) * -1)
    """

    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=5).sum() / data['Volume'].rolling(window=5).sum()
    # Calculate the moving average of volume
    adv60 = sma(data['Volume'], 60)
    # Calculate the difference between data['VWAP'] and its minimum
    diff_vwap_min = data['VWAP'] - ts_min(data['VWAP'], 12)
    # Calculate the rank of the difference
    rank_diff_vwap_min = rank(diff_vwap_min)
    # Calculate the correlation between the ranks of data['VWAP'] and adv60
    corr_rank = correlation(ts_rank(data['VWAP'], 20), ts_rank(adv60, 4), 18)
    # Calculate the rank of the correlation
    rank_corr_rank = ts_rank(corr_rank, 3)
    # Calculate the final alpha
    alpha = rank_diff_vwap_min.pow(rank_corr_rank) * -1
    return alpha


def alpha95(data, window1=19, window2=12, window3=40):
    """
    Alpha#095
    Formula:
    (rank((open - ts_min(open, 12))) < ts_rank((rank(correlation(sum(((high + low)/ 2), 19), sum(adv40, 19), 12))), 11))
    """
    # Calculate the moving average of volume
    adv40 = sma(data['Volume'], 40)
    # Calculate the average of high and low
    avg_high_low = (data['High'] + data['Low']) / 2
    # Calculate the sum of the average and the rank of the sum
    sum_avg_rank = rank(correlation(ts_sum(avg_high_low, 19), ts_sum(adv40, 19), 12))
    # Calculate the minimum of open
    min_open = ts_min(data['Open'], 12)
    # Calculate the rank of the minimum
    rank_min_open = rank(min_open)
    # Calculate the final alpha
    alpha = rank_min_open < ts_rank(sum_avg_rank, 11)
    return alpha

def alpha96(data, window1=40, window2=20, window3=13):
    adv60 = data['Volume'].rolling(window=window1).mean()
    p1 = ts_rank(decay_linear(correlation(rank(data['VWAP']), rank(data['Volume']).to_frame(), 4), 4).CLOSE, window2)
    p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(data['Close'], 7), ts_rank(adv60, 4), 4).to_frame(), 13), 14).CLOSE, window3)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df['max'] = np.where(df['p1'] >= df['p2'], df['p1'], df['p2'])
    df['max'] = np.where(df['p2'] >= df['p1'], df['p2'], df['p1'])
    return -1 * df['max']

def alpha97(data, window1=60, window2=7, window3=13, window4=18, window5=6):
    adv60 = data['Volume'].rolling(window=window1).mean()
    p1 = rank(decay_linear(delta(indneutralize(((data['Low'] * 0.721001) + (data['VWAP'] * (1 - 0.721001))), indclass.sector), 3.3705), 20.4523))
    p2 = ts_rank(decay_linear(ts_rank(correlation(ts_rank(data['Low'], window2), ts_rank(adv60, window3), 5), window4), 15.7152), window5)
    return -1 * (p1 - p2)

def alpha98(data, window1=5, window2=26, window3=21, window4=9, window5=7, window6=8):
    adv5 = data['Volume'].rolling(window=window1).mean()
    adv15 = data['Volume'].rolling(window=window2).mean()
    return (rank(decay_linear(correlation(data['VWAP'], sma(adv5, window2), 5).to_frame(), window1).CLOSE) -
            rank(decay_linear(ts_rank(ts_argmin(correlation(rank(data['Open']), rank(adv15), window3), window4), window5).to_frame(), window6).CLOSE))

def alpha99(data, window1=60, window2=20, window3=9, window4=6):
    adv60 = data['Volume'].rolling(window=window1).mean()
    return ((rank(correlation(ts_sum(((data['High'] + data['Low']) / 2), window2), ts_sum(adv60, window2), window3)) <
             rank(correlation(data['Low'], data['Volume'], window4))) * -1)

# def alpha100(data):
#     indneutralize_rank_close_low_high_volume = indneutralize(indneutralize(data['Close'] - data['Low'] - data['High'] - data['Close'] / (data['High'] - data['Low']) * data['Volume'], data['Subindustry']), data['Subindustry'])
#     correlation_close_rank_adv20 = data['Close'].rolling(window=5).corr(data['Adv20'].rank())
#     rank_ts_argmin_close_30 = data['Close'].rolling(window=30).apply(lambda x: np.argmin(x))
#     scale_indneutralize_correlation_rank_ts_argmin_close_30 = scale(indneutralize(correlation_close_rank_adv20 - rank_ts_argmin_close_30, data['Subindustry']), data['Subindustry'])
#     alphavalue = 0 - (1 * (((1.5 * scale(indneutralize_rank_close_low_high_volume, data['Subindustry'])) - scale_indneutralize_correlation_rank_ts_argmin_close_30) * (data['Volume'] / data['Adv20'])))
#     return alphavalue

def alpha101(data):
    alphavalue = (data['Close'] - data['Open']) / ((data['High'] - data['Low']) + 0.001)
    return alphavalue





