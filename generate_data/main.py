import data_utils
import numpy as np
import argparse
import datetime
from dateutil import parser
import calendar
from dateutil.relativedelta import relativedelta
import os

argparser = argparse.ArgumentParser(description='Run!')
argparser.add_argument('end_date', type=str, help='')
argparser.add_argument('regress_dir', type=str, help='')
argparser.add_argument('past_quater_number', type=str, help='')
args = argparser.parse_args()

end_date = parser.parse(args.end_date)
db = data_utils.Database('data')

regress_dir = args.regress_dir
past_quater_number = args.past_quater_number
if not os.path.exists(regress_dir):
    os.makedirs(regress_dir)

columns = [
    data_utils.Indicator.k_column_date,
    # data_utils.Indicator.k_stock_column_code,
    data_utils.Indicator.k_stock_column_close_price,
]

ind = data_utils.Indicator(db,
                           start_date=(
                               end_date + relativedelta(months=-12)).date().isoformat(),
                           end_date=end_date.date().isoformat())

labels_baseline = ind.get_stocks_timed().dropna(subset=columns) \
    .sort_values(data_utils.Indicator.k_column_date) \
    .groupby(data_utils.Indicator.k_stock_column_code)[columns] \
    .last()

# NOTE
ind_0 = data_utils.Indicator(db, end_date=end_date.date().isoformat())
labels_0 = ind_0.get_stocks_timed().dropna(subset=columns) \
    .sort_values(data_utils.Indicator.k_column_date) \
    .groupby(data_utils.Indicator.k_stock_column_code)[columns] \
    .last()
labels_0.to_pickle(f'{regress_dir}/labels.{0}_month.{args.end_date}.pickle')

for i in range(3):
    new_end_date = end_date + relativedelta(months=i+1)
    _, day = calendar.monthrange(new_end_date.year, new_end_date.month)
    new_end_date = datetime.date(new_end_date.year, new_end_date.month, day)
    new_ind = data_utils.Indicator(db, end_date=new_end_date.isoformat())
    labels = new_ind.get_stocks_timed().dropna(subset=columns) \
        .sort_values(data_utils.Indicator.k_column_date) \
        .groupby(data_utils.Indicator.k_stock_column_code)[columns] \
        .last()

    labels['close_price_diff_ratio'] = \
        labels[data_utils.Indicator.k_stock_column_close_price] / \
        labels_baseline[data_utils.Indicator.k_stock_column_close_price] - 1
    labels.dropna(inplace=True)
    labels.to_pickle(
        f'{regress_dir}/labels.{i+1}_month.{args.end_date}.pickle')

database = data_utils.compute_indicators(ind, end_date, past_quater_number)

# filter the presumbly correct indicators
# k_filter_columns = ['avg_price_mean_x', 'avg_price_std_x', 'fund_shareholding_mean', 'fund_shareholding_std', 'fund_number_mean', 'fund_number_std', 'close_price_mean', 'close_price_std', 'avg_price_mean_y', 'avg_price_std_y', 'turnover_rate_mean', 'turnover_rate_std', 'amplitutde_mean', 'amplitutde_std', 'margin_diff_mean', 'margin_diff_std', 'share_ratio_of_funds_mean', 'share_ratio_of_funds_std', 'num_of_funds_mean', 'num_of_funds_std', 'fund_owner_affinity_mean', 'fund_owner_affinity_std', 'cyclical_industry']
k_filter_columns = database.columns.values.tolist()
for col in k_filter_columns:
    print("k_filter_columns:", k_filter_columns)
    print("database[col]:", database[col])
    if col not in ['id', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股']:
        database = database[database[col] <= 10]

database = database.fillna(0.0)

print("-----database:{}\n".format(database))
print("-----end_date:{}\n".format(end_date))
print("-----database.describe:{}\n".format(database.describe()))
database.to_pickle(f'{regress_dir}/indicators.{args.end_date}.pickle')
