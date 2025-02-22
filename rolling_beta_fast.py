import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime

#Rolling beta that takes 4 hours
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
        market_ret = group['market_ret']
        asset_ret = group['returns']

        # Perform rolling regression
        rolling_betas = market_ret.rolling(window, min_periods=2).apply(
            lambda x: linregress(x, asset_ret.loc[x.index])[0] if len(x) > 1 else np.nan,
            raw=False
        )
        
        return rolling_betas

    # Apply optimized rolling beta calculation
    prices['beta'] = prices.groupby('symbol', group_keys=False).apply(calc_beta)
    return prices