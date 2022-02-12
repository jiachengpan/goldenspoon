import pandas as pd
import numpy as np

from data_utils import preprocess_data

class Dataset:
    def __init__(self, inputs):
        dfs = [preprocess_data(df) for df in inputs]

        self.init_id_map(dfs)
        self.init_data(dfs)

    def init_id_map(self, dfs):
        self.stock_map = {}
        self.fund_map  = {}
        self.stock_map_reverse = {}
        self.fund_map_reverse  = {}

        for df in dfs:
            if 'stock_id' in df.index.names and 'stock_name' in df.index.names:
                self.stock_map.update(dict(zip(
                    df.index.get_level_values('stock_id'),
                    df.index.get_level_values('stock_name'))))
                self.stock_map_reverse.update(dict(zip(
                    df.index.get_level_values('stock_name'),
                    df.index.get_level_values('stock_id'))))

            if 'fund_id' in df.index.names and 'fund_name' in df.index.names:
                self.fund_map.update(dict(zip(
                    df.index.get_level_values('fund_id'),
                    df.index.get_level_values('fund_name'))))
                self.fund_map_reverse.update(dict(zip(
                    df.index.get_level_values('fund_name'),
                    df.index.get_level_values('fund_id'))))

    def init_data(self, dfs):
        self.indexed_data = {df.index.names: df for df in dfs}

        self.stock_stats = {}
        self.stock_stats['generic'] = \
            self.preprocess(self.indexed_data[('stock_id', 'stock_name')].reset_index())
        self.stock_stats['date'] = \
            self.preprocess(self.indexed_data[('stock_id', 'stock_name', 'time')].reset_index())

        self.fund_stats = {}
        self.fund_stats['generic'] = \
            self.preprocess(self.indexed_data[('fund_id',  'fund_name')].reset_index())
        self.fund_stats['date'] = \
            self.preprocess(self.indexed_data[('fund_id',  'fund_name', 'time')].reset_index())

        df_topn_stocks = \
            self.preprocess(self.indexed_data[('fund_id',  'fund_name', 'time', 'stock_id', 'stock_name')].reset_index())
        df_topn_stocks = df_topn_stocks.melt(
            id_vars=df_topn_stocks.columns.tolist()[:5],
            var_name='indicator',
            value_name='value')
        self.fund_stats['topn_stocks'] = df_topn_stocks

        return


        for k, v in self.stock_stats.items():
            print(f'stock stat: {k}, {v.shape}')
            print(f'columns: {len(v.columns)}')
            for c in v.columns:
                print(f'  {c}')

        for k, v in self.fund_stats.items():
            print(f'fund_stat: {k}, {v.shape}')
            print(f'columns: {len(v.columns)}')
            for c in v.columns:
                print(f'  {c}')

    def preprocess(self, df):
        df = df.rename({
            'stock_id':     '股票代码',
            'stock_name':   '股票名称',
            'fund_id':      '基金代码',
            'fund_name':    '基金名称',
            'time':         '日期',
        }, axis=1).dropna(how='all')

        return df

    def get_funds(self):
        return self.fund_stats['generic']

    def get_stocks(self):
        return self.stock_stats['generic']

    def get_fund_stats(self, name):
        return self.fund_stats[name]

    def get_stock_stats(self, name):
        return self.stock_stats[name]
