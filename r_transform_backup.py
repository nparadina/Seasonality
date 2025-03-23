# Importing necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from numba import njit, prange
import psutil

print("Start", datetime.now())

# Load data
prices = pd.read_csv('C:/Users/nikap/QuantImp/stocks_daily.csv', engine='pyarrow')
#prices = pd.read_csv('C:/Users/nikap/QuantImp/sample/stocks_daily-2.csv')


# Convert column names to lowercase and replace spaces with underscores
prices.columns = prices.columns.str.lower().str.replace(' ', '_')
print("Here1",datetime.now())

# Remove duplicate rows based on 'symbol' and 'date'
prices = prices.drop_duplicates(subset=['symbol', 'date'])
print("Here2",datetime.now())

# Identify and remove rows with duplicated symbols (e.g., 'phun' and 'phun.1')
# Duplicates based on symbol with a trailing ".number"
# Create `symbol_short` by removing any trailing '.number' suffix
prices['symbol_short'] = prices['symbol'].str.replace(r'\.\d$', '', regex=True)
# Extract first character of symbol for `symbol_first`
#prices['symbol_first'] = prices['symbol'].str[0]
print("Here3",datetime.now())

dups = prices.groupby(['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_short']).size().reset_index(name='n')
print("Here4",datetime.now())
dups = dups.merge(prices[['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close','symbol_short', 'symbol']],
                  on=['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_short'],
                  how='left')
print("Here5",datetime.now())
symbols_remove = dups[dups['n'] > 1]
#symbols_remove.to_csv('symbols_remove.csv', index=False)
print("Here6",datetime.now())
prices = prices.merge(symbols_remove, on=prices.columns.tolist(), how='left', indicator=True)
print("Here7",datetime.now())
# Keep only rows that are not in symbols_remove
prices[prices['_merge'] == 'left_only']
print("Here8",datetime.now())
prices.drop(columns=['_merge', 'symbol_short','n'],inplace=True )
print("Here9",datetime.now())
# Adjust prices
prices['adj_rate'] = prices['adj_close'] / prices['close']
prices['open'] *= prices['adj_rate']
prices['high'] *= prices['adj_rate']
prices['low'] *= prices['adj_rate']
prices.rename(columns={'close': 'close_raw', 'adj_close': 'close'}, inplace=True)
print("Here10",datetime.now())
prices.drop(columns=['adj_rate'], inplace=True)
print("Here11",datetime.now())

# Filter out rows where any price column is below 1e-8
prices = prices[(prices['open'] > 1e-8) & (prices['high'] > 1e-8) & (prices['low'] > 1e-8) & (prices['close'] > 1e-8)]
print("Here12",datetime.now())
# Sort by symbol and date
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here13",datetime.now())
# Calculate returns
prices['returns'] = prices.groupby('symbol')['close'].pct_change()
print("Here14",datetime.now())
# Remove missing values
prices.dropna(inplace=True)
print("Here15",datetime.now())
# Set SPY returns as market returns
spy_ret = prices[prices['symbol'] == 'spy'][['date', 'returns']].rename(columns={'returns': 'market_ret'})
print("Here16",datetime.now())
prices = prices.merge(spy_ret, on='date', how='left')
print("Here17",datetime.now())

# Keep only symbols with at least 253 observations
symbol_counts = prices['symbol'].value_counts()
print("Here18",datetime.now())
remove_symbols = symbol_counts[symbol_counts < 253].index
print("Here19",datetime.now())
prices = prices[~prices['symbol'].isin(remove_symbols)]
print("Here20",datetime.now())

# Calculate rolling dollar volume for filtering
prices['dollar_volume_month'] = prices['close_raw'] * prices['volume']
print("Here21",datetime.now())
prices['dollar_volume_month'] = prices.groupby('symbol')['dollar_volume_month'].rolling(window=22, min_periods=1).sum().reset_index(0, drop=True)
print("Here22",datetime.now())

# Function to label most liquid assets. 
def calculate_liquid(prices, n):
    dt = prices.copy()
    #dt.sort_values(by=['date', 'dollar_volume_month'], ascending=[True, False], inplace=True)
    dt['liquid_' + str(n)] = dt.groupby('date').cumcount() < n
    return dt

prices.sort_values(by=['date', 'dollar_volume_month'], ascending=[True, False], inplace=True)
print("Here23",datetime.now())
prices = calculate_liquid(prices, 100)
prices = calculate_liquid(prices, 200)
prices = calculate_liquid(prices, 500)
prices = calculate_liquid(prices, 1000)
print("Here24",datetime.now())
# Drop dollar_volume_month column as it's no longer needed
prices.drop(columns=['dollar_volume_month'], inplace=True)
print("Here25",datetime.now())

'''
My 7hours function
# Rolling beta calculation (252-day rolling beta)
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
'''


''' Rolling beta that takes 4 hhours
def rolling_beta_fast(prices, window=252):
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

#Call to a slower rolling beta function
#prices = rolling_beta(prices)

# Ensure data is sorted before grouping
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())
# Run optimized function
prices = rolling_beta_fast(prices)
print("Here26.2",datetime.now())
'''



'''
# Numba-optimized rolling beta calculation; errors because of the use of linregress with numba
@njit(parallel=True)
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
            slope, _, _, _, _ = linregress(x, y)
            rolling_betas[i] = slope
            
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

# Ensure data is sorted before running
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())
prices = rolling_beta_fast(prices)
print("Here26.2",datetime.now())
'''


#Option with polyfit from numpy instead of lingress; same problem as lingress with Numba, results in NaN Error; would work if nopython=False- To Test

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

# Ensure data is sorted before running
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())
prices = rolling_beta_fast(prices)
print("Here26.2",datetime.now())


'''
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

# Ensure data is sorted before running
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())
prices = rolling_beta_fast(prices)
print("Here26.2",datetime.now())
'''

'''
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

# Ensure data is sorted before running
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())
print(prices.shape)
print(prices.memory_usage(deep=True).sum() / 1e9, "GB")
prices = rolling_beta_fast(prices)
print("Here26.2",datetime.now())
'''



# Rank symbols by beta and label top 5%, 10%, and 1%. But the ranking is done on the least volatile stocks
prices['beta_rank'] = prices.groupby('date')['beta'].transform(lambda x: x.abs().rank(method='dense', ascending=True))
print("Here27",datetime.now())
prices['beta_rank_pct'] = prices.groupby('date')['beta_rank'].apply(lambda x: x / x.max())
print("Here28",datetime.now())

for pct in [0.99, 0.95, 0.90]:
    col_name = f'beta_rank_largest_{int(pct * 100)}'
    prices[col_name] = (prices['beta_rank_pct'] > pct).astype(int)
print("Here29",datetime.now())
# Sort by symbol and date as final step
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here30",datetime.now())
# Display the result for inspection

print(prices[prices['beta_rank_largest_99'] == 1].head())

prices[prices['beta_rank_largest_90'] == 1].to_csv('prices.csv')
print(datetime.now().time())