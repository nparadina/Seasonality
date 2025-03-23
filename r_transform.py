# Importing necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from numba import njit, prange
import psutil
import rolling_beta_og as rb
import rolling_beta_fast as rbf
import rolling_beta_lingress_numba as rbln
import rolling_beta_lstsq as rbls
import rolling_beta_manual_numba as rbmn
import rolling_beta_polyfit_NaN as rbpn
import rolling_beta_vectorized as  rbv


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
#prices = calculate_liquid(prices, 200)
#prices = calculate_liquid(prices, 500)
#prices = calculate_liquid(prices, 1000)
print("Here24",datetime.now())
# Drop dollar_volume_month column as it's no longer needed
prices.drop(columns=['dollar_volume_month'], inplace=True)
print("Here25",datetime.now())


# Ensure data is sorted before grouping
prices.sort_values(by=['symbol', 'date'], inplace=True)
print("Here26.1",datetime.now())

#Call to a  rolling beta function
prices = rbv.rolling_beta(prices)

print("Here26.2",datetime.now())


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