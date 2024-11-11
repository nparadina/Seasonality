# Importing necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load data
prices = pd.read_csv('C:/Users/nikap/QuantImp/sample/stocks_daily-2.csv')

# Convert column names to lowercase and replace spaces with underscores
prices.columns = prices.columns.str.lower().str.replace(' ', '_')


# Remove duplicate rows based on 'symbol' and 'date'
prices = prices.drop_duplicates(subset=['symbol', 'date'])


# Identify and remove rows with duplicated symbols (e.g., 'phun' and 'phun.1')
# Duplicates based on symbol with a trailing ".number"
# Create `symbol_short` by removing any trailing '.number' suffix
prices['symbol_short'] = prices['symbol'].str.replace(r'\.\d$', '', regex=True)
# Extract first character of symbol for `symbol_first`
#prices['symbol_first'] = prices['symbol'].str[0]

dups = prices.groupby(['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_short']).size().reset_index(name='n')
dups = dups.merge(prices[['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close','symbol_short', 'symbol']],
                  on=['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_short'],
                  how='left')
symbols_remove = dups[dups['n'] > 1]
#symbols_remove.to_csv('symbols_remove.csv', index=False)

prices = prices.merge(symbols_remove, on=prices.columns.tolist(), how='left', indicator=True)

# Keep only rows that are not in symbols_remove
prices[prices['_merge'] == 'left_only']
prices.drop(columns=['_merge', 'symbol_short','n'],inplace=True )

# Adjust prices
prices['adj_rate'] = prices['adj_close'] / prices['close']
prices['open'] *= prices['adj_rate']
prices['high'] *= prices['adj_rate']
prices['low'] *= prices['adj_rate']
prices.rename(columns={'close': 'close_raw', 'adj_close': 'close'}, inplace=True)
prices.drop(columns=['adj_rate'], inplace=True)

# Filter out rows where any price column is below 1e-8
prices = prices[(prices['open'] > 1e-8) & (prices['high'] > 1e-8) & (prices['low'] > 1e-8) & (prices['close'] > 1e-8)]

# Sort by symbol and date
prices.sort_values(by=['symbol', 'date'], inplace=True)

# Calculate returns
prices['returns'] = prices.groupby('symbol')['close'].pct_change()

# Remove missing values
prices.dropna(inplace=True)

# Set SPY returns as market returns
spy_ret = prices[prices['symbol'] == 'spy'][['date', 'returns']].rename(columns={'returns': 'market_ret'})
prices = prices.merge(spy_ret, on='date', how='left')

# Keep only symbols with at least 253 observations
symbol_counts = prices['symbol'].value_counts()
remove_symbols = symbol_counts[symbol_counts < 253].index
prices = prices[~prices['symbol'].isin(remove_symbols)]

# Calculate rolling dollar volume for filtering
prices['dollar_volume_month'] = prices['close_raw'] * prices['volume']
prices['dollar_volume_month'] = prices.groupby('symbol')['dollar_volume_month'].rolling(window=22, min_periods=1).sum().reset_index(0, drop=True)

# Function to label most liquid assets
def calculate_liquid(prices, n):
    dt = prices.copy()
    dt.sort_values(by=['date', 'dollar_volume_month'], ascending=[True, False], inplace=True)
    dt['liquid_' + str(n)] = dt.groupby('date').cumcount() < n
    return dt

prices = calculate_liquid(prices, 100)
prices = calculate_liquid(prices, 200)
prices = calculate_liquid(prices, 500)
prices = calculate_liquid(prices, 1000)

# Drop dollar_volume_month column as it's no longer needed
prices.drop(columns=['dollar_volume_month'], inplace=True)

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
prices = rolling_beta(prices)

# Rank symbols by beta and label top 5%, 10%, and 1%
prices['beta_rank'] = prices.groupby('date')['beta'].rank(method='dense', ascending=False)
prices['beta_rank_pct'] = prices.groupby('date')['beta_rank'].apply(lambda x: x / x.max())

for pct in [0.99, 0.95, 0.90]:
    col_name = f'beta_rank_largest_{int(pct * 100)}'
    prices[col_name] = (prices['beta_rank_pct'] > pct).astype(int)

# Sort by symbol and date as final step
prices.sort_values(by=['symbol', 'date'], inplace=True)

# Display the result for inspection
print(prices(prices['beta_rank_largest_90']==1).head())
