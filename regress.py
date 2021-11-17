import glob
import pandas as pd
import goldenspoon

db = goldenspoon.Database('data')

for filename in glob.glob('*.pickle'):
    df = pd.read_pickle(filename)
    print(filename)

    df = df[df.num_of_funds_mean.notnull()]
    df = df[df.num_of_funds_std.notnull()]
    df = df[df.close_price_mean.notnull()]
    df = df[df.close_price_std.notnull()]
    print(df)
    print(df.info())
    filtered = df[['close_price_mean', 'id']]
    problematic_ids = filtered[filtered.close_price_mean.isna()]

    #ind = goldenspoon.Indicator(db)
    #stocks_general = ind.get_stock_general()
    #problematic_stocks = pd.merge(problematic_ids, stocks_general, left_on='id', right_on='股票代码')
    #print(problematic_stocks)
