import os
import re
import json
import glob
import numpy as np
import pandas as pd
import datetime
import calendar
import pprint
from collections import defaultdict
from . import utils

class GenericIndexBase:
    def __init__(self):
        self.result = defaultdict(dict)

    def get_indexed_data(self):
        rows = []
        for k, v in self.result.items():
            if isinstance(v, dict):
                for key_name, key_value in zip(self.key_names, k):
                    v[key_name] = key_value
                rows.append(v)
            elif isinstance(v, (tuple, list)):
                for vv in v:
                    if not v: continue
                    for key_name, key_value in zip(self.key_names, k):
                        vv[key_name] = key_value
                    rows.append(vv)
            else:
                assert 0, 'unsupported type: {}'.format(type(v))

        if not rows:
            return None
        return pd.DataFrame(rows)

# generic indexer for columns without special metadata
class GenericNameIndex(GenericIndexBase):
    name = 'generic'
    key_names = ['证券代码', '证券名称']

    def run(self, row, col, metadata):
        if pd.isna(row[col]):
            return False

        indexed = self.result
        key = (row['证券代码'], row['证券名称'])

        if col in indexed[key] and not np.isclose(indexed[key][col], row[col]):
            print('WARN: column "{}" value already assigned: key: {}, value: {} vs {}'.format(
                col, key, indexed[key][col], row[col]))
        indexed[key][col] = row[col]
        return True

# generic indexer for columns with date
class GenericDateIndex(GenericIndexBase):
    name = 'date'
    key_names = ['证券代码', '日期']

    def run(self, row, col, metadata):
        if 'date' not in metadata:
            return False
        if pd.isna(row[col]):
            return False

        indexed = self.result
        key = (row['证券代码'], metadata['date'])

        name = ' '.join(
            [metadata['name']] +
            ['[%s]%s' % (k, v) for k, v in metadata.items() if k not in ('name', 'date')])

        if name in indexed[key] and not np.isclose(indexed[key][name], row[col]):
            print('WARN: column "{}" value already assigned: key: {}, value: {} vs {}'.format(
                name , key, indexed[key][name], row[col]))
        indexed[key][name] = row[col]
        return True

# indexer for funds topN stock holding stats
class TopNStockHoldingsIndex(GenericIndexBase):
    name = 'topn_fund_stock_holding'
    key_names = ['证券代码', '日期']

    def run(self, row, col, metadata):
        k_columns = ('重仓股股票市值', '重仓股持仓占流通股比例', '前十大重仓股名称')
        if metadata['name'] not in k_columns:
            return False

        assert 'date' in metadata

        indexed = self.result
        key = (row['证券代码'], metadata['date'])
        if pd.isna(key[0]):
            return False

        if key not in indexed:
            indexed[key] = defaultdict(dict)

        if metadata['name'] == k_columns[-1]:
            if pd.isna(row[col]):
                return True
            stock_names = row[col].split(',')
            for i, name in enumerate(stock_names):
                try:
                    indexed[key][i]['证券名称'] = name
                except:
                    assert 0, 'i: {} stock: {} {}'.format(i, name, stock_names)
        else:
            topN = metadata['topN']-1
            assert topN >= 0
            indexed[key][topN]['indicator'] = metadata['name']
            indexed[key][topN]['value']     = row[col]
        return True

    def get_indexed_data(self):
        self.result = {k: list(v.values()) for k, v in self.result.items()}
        return super().get_indexed_data()

class Database:
    k_key_columns = (
        '证券代码',
        '证券名称',
        )

    def __init__(self, path):
        self.k_cached = os.path.join(os.getcwd(), 'cached')
        os.makedirs(self.k_cached, exist_ok=True)

        self.load_files(path)
        self.index_data()

    def get_funds(self):
        return self.fund_stats['generic']

    def get_stocks(self):
        return self.stock_stats['generic']

    def get_fund_stats(self, name):
        return self.fund_stats[name]

    def get_stock_stats(self, name):
        return self.stock_stats[name]

    @staticmethod
    def compute_column_metadata(col):
        re_report_timestamp = re.compile('.报告期.(\d{4})年(.*)')
        re_trans_timestamp  = re.compile('.*日期..*(\d{4})-(\d{2})-(\d{2})')
        re_topn_name        = re.compile('.名次.*第(\d+)名')
        re_meta_name        = re.compile('\[([^]]+)\](\S+)')

        k_report_mapping = {
                '一季':         (3, 31),
                '二季/中报':     (6, 30),
                '三季':         (9, 30),
                '年报':         (12, 31),
                }

        tokens   = []
        metadata = {}
        for tok in col.split():
            m = re_report_timestamp.match(tok)
            if m:
                year, rpt = m.groups()
                month, day = k_report_mapping[rpt]
                assert 'date' not in metadata
                metadata['date'] = datetime.date(int(year), month, day)
                continue

            m = re_trans_timestamp.match(tok)
            if m:
                assert 'date' not in metadata
                year, month, day = m.groups()
                metadata['date'] = datetime.date(int(year), int(month), int(day))
                continue

            m = re_topn_name.match(tok)
            if m:
                topn = int(m.groups()[0])
                metadata['topN'] = topn
                continue

            m = re_meta_name.match(tok)
            if m:
                key, value = m.groups()
                metadata[key] = value
                continue

            tokens.append(tok)

        metadata['name'] = '_'.join(tokens)
        return metadata

    def load_files(self, path):
        self.df = {}
        for filename in glob.glob(os.path.join(path, '*.xls*'), recursive=True):
            print('loading {}'.format(filename))
            df = pd.read_excel(filename)
            df = df.replace('——', np.nan)
            df.columns = [' '.join(col.split()) for col in df.columns]
            self.df[os.path.basename(filename)] = df

    def index_data(self):
        self.fund_stats = utils.pickle_cache(os.path.join(self.k_cached, 'indexed_fund_stats.pkl'), lambda :
                self.index_by_column_metadata([df for name, df in self.df.items() if name.startswith('funds')]))
        self.stock_stats = utils.pickle_cache(os.path.join(self.k_cached, 'indexed_stock_stats.pkl'), lambda :
                self.index_by_column_metadata([df for name, df in self.df.items() if name.startswith('stocks')]))

    def index_by_column_metadata(self, dfs):
        indexers = [
            TopNStockHoldingsIndex(),
            GenericDateIndex(),
            GenericNameIndex(),
        ]

        for df in dfs:
            self.index_by_column_metadata_impl(indexers, df)

        result = {}
        for indexer in indexers:
            data = indexer.get_indexed_data()
            if data is None:
                continue

            cols =  list(self.k_key_columns)
            cols += list(sorted(set(data.columns.tolist()) - set(cols)))
            cols =  [col for col in cols if col in data.columns]

            result[indexer.name] = data[cols]

        return result

    def index_by_column_metadata_impl(self, indexers, df):
        columns_metadata = {}
        for col in df.columns:
            if col in self.k_key_columns:
                continue
            columns_metadata[col] = self.compute_column_metadata(col)

        for _, row in df.iterrows():
            if any(pd.isna(row[col]) for col in self.k_key_columns):
                continue

            for col, metadata in columns_metadata.items():
                for indexer in indexers:
                    if indexer.run(row, col, metadata):
                        break

