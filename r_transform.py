# Importing necessary libraries
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import gc

# SET UP ------------------------------------------------------------------
# Global vars
PATH = "F:/data/equity/us"

# PRICE DATA --------------------------------------------------------------
# Import QC daily data
prices = pd.read_csv("C:/Users/nikap/QuantImp/sample1/stocks_daily-1.csv")

# Convert column names to lowercase and replace spaces with underscores
prices.columns = prices.columns.str.lower().str.replace(' ', '_')

# Remove duplicates
prices.drop_duplicates(subset=["symbol", "date"], inplace=True)

# Remove duplicates for symbols like 'phun' and 'phun.1'
dups = prices.groupby(['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']) \
             .apply(lambda x: x if len(x) == 1 else x[x['symbol'].str.contains(r'\.\d$')]) \
             .reset_index(drop=True)

symbols_remove = dups['symbol'].unique()
prices = prices[~prices['symbol'].isin(symbols_remove)]

# Adjust all columns
prices['adj_rate'] = prices['adj_close'] / prices['close']
prices['open'] *= prices['adj_rate']
prices['high'] *= prices['adj_rate']
prices['low'] *= prices['adj_rate']
prices.rename(columns={'close': 'close_raw', 'adj_close': 'close'}, inplace=True)
prices.drop(columns=['adj_rate'], inplace=True)

# Remove observations where price columns are very small (optional step for filtering)
prices = prices[(prices['open'] > 1e-8) & (prices['high'] > 1e-8) & (prices['low'] > 1e-8) & (prices['close'] > 1e-8)]

# Sort data by symbol and date
prices.sort_values(by=['symbol', 'date'], inplace=True)

# Calculate returns
prices['returns'] = prices.groupby('symbol')['close'].pct_change()

# Remove missing values
prices.dropna(inplace=True)

# Set SPY returns as market returns
spy_ret = prices[prices['symbol'] == 'spy'][['date', 'returns']].rename(columns={'returns': 'market_ret'})
prices = prices.merge(spy_ret, on='date', how='left')

# Remove symbols with fewer than 253 observations
symbol_counts = prices.groupby('symbol').size()
remove_symbols = symbol_counts[symbol_counts < 253].index
prices = prices[~prices['symbol'].isin(remove_symbols)]

# Free memory
gc.collect()

# FILTERING ---------------------------------------------------------------
# Calculate 22-day rolling dollar volume
prices['dollar_volume_month'] = prices.groupby('symbol')['close_raw'].transform(lambda x: x * prices['volume']).rolling(window=22, min_periods=1).sum()

def calculate_liquid(prices, n):
    dt = prices.copy()
    dt = dt.sort_values(by=['date', 'dollar_volume_month'], ascending=[True, False])
    filtered = dt.groupby('date').head(n)
    dt['liquid_' + str(n)] = dt['symbol'].isin(filtered['symbol'])
    dt.fillna(False, inplace=True)
    return dt

prices = calculate_liquid(prices, 100)
prices = calculate_liquid(prices, 200)
prices = calculate_liquid(prices, 500)
prices = calculate_liquid(prices, 1000)

# Remove columns we don't need
prices.drop(columns=['dollar_volume_month'], inplace=True)

# PREDICTORS --------------------------------------------------------------
# Sort data by symbol and date
prices.sort_values(by=['symbol', 'date'], inplace=True)

# Rolling beta calculation
def rolling_beta(data, window):
    y = data['returns']
    X = data['market_ret']
    model = RollingOLS(y, X, window=window)
    rres = model.fit()
    return rres.params

prices['beta'] = prices.groupby('symbol').apply(lambda x: rolling_beta(x, 252)).reset_index(drop=True)

# Beta ranking and flags for top 1%, 5%, 10%
prices['beta_rank'] = prices.groupby('date')['beta'].rank(method='dense', ascending=False)
prices['beta_rank_pct'] = prices.groupby('date')['beta_rank'].transform(lambda x: x / x.max())

# Create flags for top ranks
prices['beta_rank_largest_99'] = (prices['beta_rank_pct'] > 0.99).astype(int)
prices['beta_rank_largest_95'] = (prices['beta_rank_pct'] > 0.95).astype(int)
prices['beta_rank_largest_90'] = (prices['beta_rank_pct'] > 0.90).astype(int)

# Sort again to maintain order
prices.sort_values(by=['symbol', 'date'], inplace=True)
