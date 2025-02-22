#My 7hours function
# Rolling beta calculation (252-day rolling beta)
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime

def rolling_beta(prices, window=252):
    def calc_beta(window_data):
        if len(window_data) < 2:
            return np.nan
        asset = window_data['returns']
        market = window_data['market_ret']
        slope, _, _, _, _ = linregress(market, asset)
        return slope
    
    prices['beta'] = np.nan  # Initialize beta column
    for symbol, group in prices.groupby('symbol'):
        # For each group, apply rolling window
        rolling_betas = []
        for start in range(len(group) - window + 1):
            window_data = group.iloc[start:start + window]  # Select the window data
            beta = calc_beta(window_data[['market_ret', 'returns']])  # Calculate beta for the window
            rolling_betas.append(beta)  # Append the beta to the list
        
        # Now assign the calculated betas back to the original DataFrame
        prices.loc[group.index[window - 1:], 'beta'] = rolling_betas  # Start assigning from the 252nd row onwards
    
    return prices