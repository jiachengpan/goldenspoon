import goldenspoon
import numpy as np
import argparse
import datetime
import calendar
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

ind = goldenspoon.Indicator(db,
        start_date = (end_date + relativedelta(months=-12)).date().isoformat(),
        end_date   = end_date.date().isoformat())

labels_baseline = ind.get_stocks_timed().dropna(subset=columns) \
            .sort_values(goldenspoon.Indicator.k_column_date) \
            .groupby(goldenspoon.Indicator.k_stock_column_code)[columns] \
            .last()

for i in range(3):
    new_end_date = end_date + relativedelta(months=i+1)
    _, day = calendar.monthrange(new_end_date.year, new_end_date.month)
    new_end_date = datetime.date(new_end_date.year, new_end_date.month, day)

    new_ind = goldenspoon.Indicator(db, end_date=new_end_date.isoformat())
    labels  = new_ind.get_stocks_timed().dropna(subset=columns) \
            .sort_values(goldenspoon.Indicator.k_column_date) \
            .groupby(goldenspoon.Indicator.k_stock_column_code)[columns] \
            .last()

    labels['close_price_diff_ratio'] = \
            labels[goldenspoon.Indicator.k_stock_column_close_price] / \
            labels_baseline[goldenspoon.Indicator.k_stock_column_close_price] - 1
    labels.dropna(inplace=True)
    print(new_end_date)
    print(labels)
    labels.to_pickle(f'labels.{i+1}_month.{args.end_date}.pickle')

database = goldenspoon.compute_indicators(ind)

# filter the presumbly correct indicators
k_filter_columns = [
        'margin_diff_mean',
        'amplitutde_mean',
        ]

for col in k_filter_columns:
    database = database[database[col] <= 10]

print(database.describe())
database.to_pickle(f'indicators.{args.end_date}.pickle')
