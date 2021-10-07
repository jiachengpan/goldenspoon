import goldenspoon
import numpy as np

db = goldenspoon.Database('data')

funds = db.get_funds()
print(funds.describe())
print(funds.head()[list(db.k_key_columns)])

topn_stock_holdings = db.get_fund_stats('topn_stocks')
print(topn_stock_holdings.describe())
print(topn_stock_holdings.head())
print(topn_stock_holdings.indicator.unique())
print(np.nan in topn_stock_holdings.value.unique())

