#Option with polyfit from numpy instead of lingress; same problem as lingress with Numba, results in NaN Error; would work if nopython=False- To Test

import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime


def calc_beta_numba(market_ret, asset_ret, window):
    """
    Calculate rolling beta using a Numba-accelerated approach.
    """
    n = len(market_ret)
    rolling_betas = np.full(n, np.nan)  # Preallocate array for speed
    
    for i in range(window - 1, n):  # Start from window-1 to match rolling behavior
        x = market_ret[i - window + 1 : i + 1]  # Market returns window
        y = asset_ret[i - window + 1 : i + 1]   # Asset returns window

        # Check for NaNs and Infs
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("Input contains NaNs")
        if np.isinf(x).any() or np.isinf(y).any():
            raise ValueError("Input contains infinite values")
                
        if len(x) >= 2:  # Need at least 2 points for linregress
            beta, _ = np.polyfit(x, y, 1)  # Degree 1 linear regression (slope)
            rolling_betas[i] = beta
        else:
            raise ValueError("Too few arguments")
            
    return rolling_betas

# Optimized rolling beta function
def rolling_beta_fast(prices, window=252):
    """
    Optimized function to calculate rolling beta using Numba.
    """
    # Convert data to NumPy for speed
    symbols = prices['symbol'].unique()
    betas = np.full(len(prices), np.nan)  # Preallocate array for results

    for symbol in symbols:
        symbol_mask = prices['symbol'] == symbol  # Get rows for this symbol
        market_ret = prices.loc[symbol_mask, 'market_ret'].values
        asset_ret = prices.loc[symbol_mask, 'returns'].values

        # Compute rolling betas using Numba function
        betas[symbol_mask] = calc_beta_numba(market_ret, asset_ret, window)

    # Assign calculated betas back to DataFrame
    prices['beta'] = betas
    return prices