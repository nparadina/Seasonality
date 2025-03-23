
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from numba import njit, prange
import psutil

# Manual beta calculation within numba, takes 13 hours, worse
@njit(nopython=True,parallel=True)
def calc_beta_numba(market_ret, asset_ret, window):
    """
    Calculate rolling beta using a Numba-accelerated approach.
    """
    n = len(market_ret)
    rolling_betas = np.full(n, np.nan)  # Preallocate array for speed
    
    for i in prange(window - 1, n):  # Start from window-1 to match rolling behavior
    #for i in prange(window - 1, n):  # Start from window-1 to match rolling behavior; with prange takes 10+ hours
        x = market_ret[i - window + 1 : i + 1]  # Market returns window
        y = asset_ret[i - window + 1 : i + 1]   # Asset returns window
        
        if len(x) >= 2:  # Need at least 2 points for linregress
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Compute covariance(x, y) and variance(x)
            cov_xy = np.sum((x - x_mean) * (y - y_mean))
            var_x = np.sum((x - x_mean) ** 2)

            if var_x > 0:  # Prevent division by zero
                beta = cov_xy / var_x
                rolling_betas[i] = beta  # Assign beta at the last position of the window
                #print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
                #print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

            
    return rolling_betas

# Optimized rolling beta function
def rolling_beta_fast(prices, window=252):
    """
    Optimized function to calculate rolling beta using Numba.
    """
    # Convert data to NumPy for speed
    symbols = prices['symbol'].unique()
    betas = np.full(len(prices), np.nan)  # Preallocate array for results
    print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
    for symbol in symbols:
        symbol_mask = prices['symbol'] == symbol  # Get rows for this symbol
        market_ret = prices.loc[symbol_mask, 'market_ret'].values
        asset_ret = prices.loc[symbol_mask, 'returns'].values

        # Compute rolling betas using Numba function
        betas[symbol_mask] = calc_beta_numba(market_ret, asset_ret, window)

    # Assign calculated betas back to DataFrame
    prices['beta'] = betas
    return prices
