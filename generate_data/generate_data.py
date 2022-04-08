import os
import glob
import calendar
import datetime
import pandas as pd
from dateutil import parser
from dateutil.relativedelta import relativedelta
from utils.logger import G_LOGGER

import data_utils

k_index = ['time',
           'stock_id',
           'stock_name',
           'fund_id',
           'fund_name',
            ]

def collect_raw_data(path):
    result = []
    for filename in glob.glob(os.path.join(path, '*.csv.gz')):
        df = pd.read_csv(filename)
        index = [i for i in df.columns if i in k_index]
        if 'time' in index:
            df['time'] = pd.to_datetime(df['time']).dt.date
        result.append(df.set_index(index))
    return result

def main(args):
    try:
        dfs = collect_raw_data(args.path)

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

        print(f'DBG: preparing data with end: {end_date} start: {start_date}')
        ind = data_utils.Indicator(ds,
                                    start_date = start_date.date().isoformat(),
                                    end_date   = end_date.date().isoformat())

        os.makedirs(args.output, exist_ok=True)
        ind.get_stock_general().sort_values('股票代码') \
                .reset_index(drop=True) \
                .to_csv(f'{args.output}/stock_general.csv')
        ind.get_stock_performance().sort_values('股票代码') \
                .reset_index(drop=True) \
                .to_csv(f'{args.output}/stock_perf.csv')
        ind.get_stock_holding_funds_share().sort_values(['股票代码', '日期']) \
                .reset_index(drop=True) \
                .to_csv(f'{args.output}/stock_holding_funds.csv')

        # compute indicators
        indicators = data_utils.compute_indicators(ind, end_date.date().isoformat(), args.past_quarters)
        print('DBG: indicators: shape: ', indicators.shape)

        if args.value_threshold is not None:
            # filter the presumbly correct indicators
            k_exclude_filter_columns = ['id', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股']
            filter_columns = [c for c in indicators.columns if c not in k_exclude_filter_columns]
            for col in filter_columns:
                describe = indicators[col].describe()
                old_shape = indicators.shape
                indicators = indicators[indicators[col] <= args.value_threshold]
                new_shape = indicators.shape

                print(f'DBG: filtering on {col} value: {args.value_threshold}')
                print(f'DBG: {old_shape} -> {new_shape}')
                print(describe)
                print(f'-' * 20)

        print("DBG: indicators.describe: {}".format(indicators.describe()))
        indicators = indicators.fillna(0.0)

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

            if args.label_for_priormonth and i>0:
                # assert(0)
                local_labels = labels[data_utils.Indicator.k_stock_column_close_price]
            else:
                local_labels = labels[data_utils.Indicator.k_stock_column_close_price]
                pre_labels = labels_baseline[data_utils.Indicator.k_stock_column_close_price]

            labels['close_price_diff_ratio'] = local_labels / pre_labels - 1

            labels.dropna(inplace=True)
            labels.to_pickle(f'{args.output}/labels.{i+1}_month.{end_date.date().isoformat()}.pickle')

            if args.label_for_priormonth:
                pre_labels = local_labels.copy()

    except Exception as e:
        import traceback as tb
        tb.print_exc()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description='generate dataset')
    argparser.add_argument('--path',                 default='raw_data', type=str,        help='path')
    argparser.add_argument('--verbose',              default=False, action='store_true',  help='verbose')
    argparser.add_argument('--start-date', '-sd',    default=None)
    argparser.add_argument('--end-date', '-ed',      default='2021-09-30')
    argparser.add_argument('--past-quarters',        default=4, type=int)
    argparser.add_argument('--value-threshold',      default=None, type=float)
    argparser.add_argument('--output',               default='output')
    argparser.add_argument('--label_for_priormonth', action='store_true',  help='label for priormonth.')

    args = argparser.parse_args()
    for arg in vars(args):
        G_LOGGER.info(format(arg, '<40')+format(" -----> " + str(getattr(args, arg)), '<'))
    G_LOGGER.info('\n')

    #import cProfile
    main(args)
    #cProfile.run('main(args)', 'rstats')
