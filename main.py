import goldenspoon
import numpy as np
import argparse
import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

argparser = argparse.ArgumentParser(description='Run!')
argparser.add_argument('end_date', type=str, help='')

args = argparser.parse_args()

end_date = parser.parse(args.end_date)

db = goldenspoon.Database('data')

columns = [
        goldenspoon.Indicator.k_column_date,
        #goldenspoon.Indicator.k_stock_column_code,
        goldenspoon.Indicator.k_stock_column_close_price,
        ]
for i in range(3):
    new_end_date = end_date + relativedelta(months=i+1)

    print(i, end_date, new_end_date)

    ind = goldenspoon.Indicator(db, end_date=new_end_date.date().isoformat())
    labels = ind.get_stocks_timed().dropna(subset=columns) \
            .sort_values(goldenspoon.Indicator.k_column_date) \
            .groupby(goldenspoon.Indicator.k_stock_column_code)[columns] \
            .last()
    labels.to_pickle(f'labels.{i+1}_month.{args.end_date}.pickle')

ind = goldenspoon.Indicator(db, end_date=end_date.date().isoformat())
database = goldenspoon.compute_indicators(ind)
print(database.describe())
database.to_pickle(f'indicators.{args.end_date}.pickle')
