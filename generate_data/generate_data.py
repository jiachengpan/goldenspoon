from cProfile import label
import os
import requests
import json
import pickle
import calendar
import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

import data_utils

k_dataset_url = 'https://api.github.com/repos/jiachengpan/eastmoney_dataset/releases/latest'

def download_and_pickle_assets(url, files = ['raw_data.pkl']):
    res = requests.get(url)
    data = json.loads(res.text)

    os.makedirs('downloads', exist_ok=True)

    result = []
    for asset in data['assets']:
        if asset['name'] not in files:
            continue

        filename = f"downloads/{asset['name']}"
        if not os.path.isfile(filename):
            print(f"downloading {asset['name']} from {asset['browser_download_url']}")
            url = asset['browser_download_url']
            ret = os.system(f"wget {url} -O {filename}")
            assert ret == 0, 'download failed'

        result.append(pickle.load(open(filename, 'rb')))

    return result

def main(args):
    try:
        assets = download_and_pickle_assets(args.url, args.files)
        dfs = [df for asset in assets for df in asset.values()]

        if args.verbose:
            for df in dfs:
                print(f'dataframe with index: {df.index.names}')
                print(f'columns:')
                for c in df.columns:
                    first_valid_index = df[c].first_valid_index()
                    first_valid_value = df[c].loc[first_valid_index] if first_valid_index is not None else None
                    print(f'  {c:20s} : {first_valid_value}')
                print(df.head())

        ds  = data_utils.Dataset(dfs)

        end_date   = parser.parse(args.end_date)
        start_date = parser.parse(args.start_date) if args.start_date is not None else \
                                 (end_date + relativedelta(months=-12))
        ind = data_utils.Indicator(ds,
                                    start_date = start_date.date().isoformat(),
                                    end_date   = end_date.date().isoformat())

        # compute indicators
        indicators = data_utils.compute_indicators(ind, end_date.date().isoformat(), args.past_quarters)
        print('DBG: indicators: shape: ', indicators.shape)

        if args.value_threshold is not None:
            # filter the presumbly correct indicators
            k_exclude_filter_columns = ['id', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股']
            filter_columns = [c for c in indicators.columns if c not in k_exclude_filter_columns]
            for col in filter_columns:
                print(f'DBG: filtering on {col}')
                print(indicators[col].describe())
                old_shape = indicators.shape
                indicators = indicators[indicators[col] <= args.value_threshold]
                new_shape = indicators.shape
                print(f'DBG: {old_shape} -> {new_shape}')

        indicators = indicators.fillna(0.0)

        print("DBG: indicators.describe: {}".format(indicators.describe()))

        os.makedirs(args.output, exist_ok=True)
        indicators.to_pickle(f'{args.output}/indicators.{end_date.date().isoformat()}.pickle')

        # compute labels

        label_columns = [
            data_utils.Indicator.k_column_date,
            # data_utils.Indicator.k_stock_column_code,
            data_utils.Indicator.k_stock_column_close_price,
        ]

        labels_baseline = ind.get_stocks_timed().dropna(subset=label_columns) \
            .sort_values(data_utils.Indicator.k_column_date) \
            .groupby(data_utils.Indicator.k_stock_column_code)[label_columns] \
            .last()

        # NOTE
        ind_0 = data_utils.Indicator(ds, end_date=end_date.date().isoformat())
        labels_0 = ind_0.get_stocks_timed().dropna(subset=label_columns) \
            .sort_values(data_utils.Indicator.k_column_date) \
            .groupby(data_utils.Indicator.k_stock_column_code)[label_columns] \
            .last()
        labels_0.to_pickle(f'{args.output}/labels.{0}_month.{end_date.date().isoformat()}.pickle')

        for i in range(3):
            new_end_date = end_date + relativedelta(months=i+1)
            _, day = calendar.monthrange(new_end_date.year, new_end_date.month)
            new_end_date = datetime.date(new_end_date.year, new_end_date.month, day)
            new_ind = data_utils.Indicator(ds, end_date=new_end_date.isoformat())
            labels = new_ind.get_stocks_timed().dropna(subset=label_columns) \
                .sort_values(data_utils.Indicator.k_column_date) \
                .groupby(data_utils.Indicator.k_stock_column_code)[label_columns] \
                .last()

            labels['close_price_diff_ratio'] = \
                labels[data_utils.Indicator.k_stock_column_close_price] / \
                labels_baseline[data_utils.Indicator.k_stock_column_close_price] - 1

            labels.dropna(inplace=True)
            labels.to_pickle(f'{args.output}/labels.{i+1}_month.{end_date.date().isoformat()}.pickle')

    except Exception as e:
        import traceback as tb
        tb.print_exc()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description='generate dataset')
    argparser.add_argument('--url',                default=k_dataset_url, type=str,                help='url')
    argparser.add_argument('--files',              default=['raw_data.pkl'], type=str, nargs='+',  help='files to download')
    argparser.add_argument('--verbose',            default=False, action='store_true',             help='verbose')
    argparser.add_argument('--start-date', '-sd',  default=None)
    argparser.add_argument('--end-date', '-ed',    default='2021-09-30')
    argparser.add_argument('--past-quarters',      default=4, type=int)
    argparser.add_argument('--value-threshold',    default=None, type=float)
    argparser.add_argument('--output',             default='output')

    args = argparser.parse_args()

    #import cProfile
    main(args)
    #cProfile.run('main(args)', 'rstats')
