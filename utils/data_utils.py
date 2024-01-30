import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=1):
    """
    Calculate the Sharpe Ratio.

    Parameters:
    - returns: numpy array or pandas Series, representing the returns of an investment/portfolio.
    - risk_free_rate: float, representing the risk-free rate, default is 0.

    Returns:
    - Sharpe Ratio: float
    """
    average_return = np.mean(returns)
    risk = np.std(returns)
    sharpe_ratio = (average_return - risk_free_rate) / risk
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate=0):
    downside_returns = np.minimum(returns - risk_free_rate, 0)
    downside_deviation = np.std(downside_returns, ddof=1)
    
    expected_return = np.mean(returns)
    sortino_ratio = (expected_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def normalize_alphas(column):
    column = column.values.reshape(-1, 1)

    # Create the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    normalized_column = scaler.fit_transform(column)

    return normalized_column
