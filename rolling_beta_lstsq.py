import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from numba import njit, prange

#Option with np.linalg.lstsq instead of polyfit and linregress. It performs least squares regression, which is equivalent to polyfit for degree-1 (linear) regression.
# Unfortunatelly np.linalg.lstsq also isn't Numba compatible

@njit(nopython=True,parallel=True)
def calc_beta_numba(market_ret, asset_ret, window):
    """
    Calculate rolling beta using a Numba-accelerated approach.
    """
    n = len(market_ret)
    rolling_betas = np.full(n, np.nan)  # Preallocate array for speed
    
    for i in prange(window - 1, n):  # Start from window-1 to match rolling behavior
        x = market_ret[i - window + 1 : i + 1]  # Market returns window
        y = asset_ret[i - window + 1 : i + 1]   # Asset returns window
        
        if len(x) >= 2:  # Need at least 2 points for linregress
            A = np.empty((len(x), 2))  # Manually create design matrix
            A[:, 0] = x
            A[:, 1] = 1.0  # Bias term (intercept)
            beta, _ = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve for slope
            rolling_betas[i] = beta
            
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