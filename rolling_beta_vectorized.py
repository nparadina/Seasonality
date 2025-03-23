import pandas as pd
import numpy as np
from datetime import datetime
from numba import njit

# Rolling beta that takes 4 hours
def rolling_beta(prices, window=252):
    """
    Optimized function to calculate rolling beta for each symbol using vectorized operations.
    """
    # Initialize beta column with NaN
    prices['beta'] = np.nan  

    # Group by symbol and calculate rolling beta
    def calc_beta(group):
        """
        Compute rolling beta using vectorized operations.
        """
        market_ret = group['market_ret'].values
        asset_ret = group['returns'].values

        # Perform rolling regression using numba
        rolling_betas = rolling_linregress(market_ret, asset_ret, window)
        
        return pd.Series(rolling_betas, index=group.index)

    # Apply optimized rolling beta calculation
    prices['beta'] = prices.groupby('symbol', group_keys=False).apply(calc_beta)
    return prices

@njit
def rolling_linregress(market_ret, asset_ret, window):
    """
    Compute rolling linear regression using numba for JIT compilation.
    """
    n = len(market_ret)
    betas = np.full(n, np.nan)
    
    
    for i in range(window, n):
        x = market_ret[i-window:i]
        y = asset_ret[i-window:i]
        if len(x) > 1:
            # Calculate slope manually
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov_xy = np.sum((x - x_mean) * (y - y_mean))
            var_x = np.sum((x - x_mean) ** 2)
            slope = cov_xy / var_x if var_x != 0 else np.nan
            betas[i] = slope
    

    return betas