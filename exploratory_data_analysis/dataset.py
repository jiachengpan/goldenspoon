from cmath import pi
from copyreg import pickle
import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import *
from collections import defaultdict

class Args:
    def __init__(self, *,
            data_path,
            train_dates,
            test_dates,
            predict_months          = 3,
            change_threshold_train  = -1e-9,
            change_threshold_test   = -1e-9,
            classifier_cut_points   = None,
            norm                    = None,
            stocks_only             = '',
            stocks_number           = 10,
            debug                   = False,
            ):

        self.data_path      = data_path
        self.train_dates    = list(sorted(train_dates))
        self.test_dates     = list(sorted(test_dates))
        self.predict_months = predict_months
        self.change_threshold_train = change_threshold_train
        self.change_threshold_test  = change_threshold_test
        self.classifier_cut_points  = list(sorted(classifier_cut_points)) if classifier_cut_points is not None else None

        self.norm = norm

        self.stocks_only   = stocks_only
        self.stocks_number = stocks_number

        self.debug = debug

    @property
    def month_label(self):
        return f'{self.predict_month}_{self.predict_mode}'

    def describe(self):
        return self.__dict__

class DataSet:
    def __init__(self, *,
            x_test,
            y_test,
            y_test_cls,
            x_train,
            y_train,
            y_train_cls,
            cls_values,
            ):

        self.x_test      = x_test
        self.y_test      = y_test
        self.y_test_cls  = y_test_cls
        self.x_train     = x_train
        self.y_train     = y_train
        self.y_train_cls = y_train_cls

        self.cls_values  = cls_values

    def __str__(self):
        result = ['DataSet:']
        result.append(f'  x_train:      {self.x_train.shape}')
        result.append(f'  y_train:      {self.y_train.shape}')
        result.append(f'  y_train_cls:  {self.y_train_cls.shape}')
        result.append(f'  x_test:       {self.x_test.shape}')
        result.append(f'  y_test:       {self.y_test.shape}')
        result.append(f'  y_test_cls:   {self.y_test_cls.shape}')
        return '\n'.join(result)

    def get_melted(self):
        x_train_melted  = pd.melt(self.x_train)
        x_test_melted   = pd.melt(self.x_test)

        return x_train_melted, x_test_melted

    @staticmethod
    def values_to_labels(values, snap_points):
        result = np.ones(values.shape, dtype=int) * -1
        offset = np.ones(values.shape) * np.inf
        for index, point in enumerate(snap_points):
            offset_ = np.abs(values - point)
            result[offset_ < offset] = index
            offset = np.minimum(offset_, offset)

        return result

    @staticmethod
    def get_indicators(data_path, date):
        pickle_name = os.path.join(data_path, f'indicators.{date}.pickle')
        df = pd.read_pickle(pickle_name)
        return df.set_index('id')

    @staticmethod
    def get_labels(data_path, date, months = [0, 1, 2, 3]):
        dfs = []
        for month in months:
            pickle_name = os.path.join(data_path, f'labels.{month}_month.{date}.pickle')
            df = pd.read_pickle(pickle_name)

            unique_dates = df['日期'].unique()
            if len(unique_dates) == 1:
                unique_date = unique_dates[0]
            else:
                print(f'ERROR: month: {month} invalid dates: {unique_dates}')
                from dateutil import parser
                from dateutil import relativedelta
                unique_date = parser.parse(date).date() + relativedelta.relativedelta(months=month)
                df['日期'] = unique_date

            df = df[['月收盘价 [复权方式]前复权']]
            df.columns = pd.MultiIndex.from_product([['close'], [unique_date]])

            dfs.append(df)

            # print('date', date, 'month:', month, 'shape:', df.shape)
            # print(df.info())
            # print(df.head())

        closes = pd.concat(dfs, axis=1, join='inner')
        closes.index.name = 'id'

        result = pd.DataFrame(index=closes.index)

        for i, (prev, curr) in enumerate(zip(closes.columns[0:], closes.columns[1:])):
            column_name = f'close_diff%+{i+1}_month'
            result[column_name] = (closes[curr] - closes[prev]) / closes[prev]

        # print('result shape:', result.shape)
        # print(result.info())
        # print(result.head())

        return result

    @classmethod
    def prepare_data(cls, data_path, dates, months = 3, norm = None):
        result = defaultdict(list)
        for date in dates:
            indicators = cls.get_indicators(data_path, date)
            labels     = cls.get_labels(data_path, date, months = range(months+1))

            assert len(labels.columns) == months, f'invalid labels columns: {labels.columns}'

            for month in range(months):
                data = pd.concat([indicators, labels.iloc[:, :month+1]], axis=1, join='inner')
                x, y = data.iloc[:, :-1], data.iloc[:, -1]

                if norm is not None:
                    x.iloc[:, :] = normalize(x, norm = norm)
                result[month+1].append((x, y))

        for k, v in result.items():
            xs, ys = zip(*v)
            x, y   = [pd.concat(xs, axis=0), pd.concat(ys, axis=0)]

            result[k] = x, y
        return result

    @classmethod
    def create(cls, args: Args):
        train_data = cls.prepare_data(args.data_path, args.train_dates, args.predict_months, args.norm)
        test_data  = cls.prepare_data(args.data_path, args.test_dates, args.predict_months, args.norm)

        assert set(train_data.keys()) == set(test_data.keys())

        result = {}
        for month in train_data.keys():
            x_train, y_train = train_data[month]
            x_test,  y_test  = test_data[month]

            if args.classifier_cut_points is not None:
                y_test_cls  = pd.Series(cls.values_to_labels(y_test,  args.classifier_cut_points))
                y_train_cls = pd.Series(cls.values_to_labels(y_train, args.classifier_cut_points))
            else:
                y_test_cls  = None
                y_train_cls = None

            result[month] = DataSet(
                y_test_cls  = y_test_cls,
                y_test      = y_test,
                x_test      = x_test,
                y_train_cls = y_train_cls,
                y_train     = y_train,
                x_train     = x_train,
                cls_values  = args.classifier_cut_points,
            )
        return result
