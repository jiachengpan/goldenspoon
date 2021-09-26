import os
import re
import json
import numpy as np
import pandas as pd
import subprocess
import shlex
import datetime
import calendar
from collections import defaultdict
import utils

class Database:
    k_files = {
        'funds.basic_info':                       '1d2RkH8CmpoKtCrquAdgxnnRgaUGnDtEw',
        'funds.topn_stock_mkt_value':             '1eZqXMu1fhUiMoB6Yf9HfrTBdHJAzb7Fr',
        'funds.topn_stock_name':                  '1AMjUJVwbxvT9fuSgv-t6waSwJ2cT389X',
        'funds.topn_stock_share_ratio':           '1uhl7SQIw1HPF9Yr6fqTb7hIbz-yytS0t',
        'funds.total_value_and_stock_investment': '14LuEIdajG2e2M_dykID7roFdi70PijNB',
        'funds.total_value_incr_ratio':           '17W7yZFuqwK2YmWhSgfibZUtuCpzUrK_s',
        'stocks.basic_info':                      '1Zn1bgbAStGLny3lMYFd4TvODcHgHr3G8',
        'stocks.margin_and_short_diff':           '1vdcUFNhRhJRUPFxeG3DDUJYegau8tuaN',
        'stocks.topn_funds_holding':              '1oL3JFVIlvKaG-7JC3dzUxbogFATpb2IZ',
        'stocks.monthly_performance_2020':        '12IGV0KDFwzU2SGMRgbo1h-Fs86oFhB0z',
        'stocks.monthly_performance_2021':        '1in6nK7gNJ7RlKgsw520eTtVKYX21PQ2A',
        'stocks.general':                         '1YdCp7wKM7Ck1_1ZiWEd3bLvueG74HRBs',
        'stocks.quarterly_report':                '1BbJobIn0Ds7K7g2ZEGoPVNMHbeUY0N3Y',
        }

    def __init__(self):
        self.k_cached = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cached')

    def _download_files(self):
        os.makedirs(self.k_cached, exist_ok=True)

        for name, fid in self.k_files.items():
            name += '.xls'
            file_name = os.path.join(self.k_cached, name)
            if os.path.exists(file_name):
                continue

            os.system('wget -O {name} https://drive.google.com/u/0/uc?id={fid}&export=download'.format(
                name = file_name,
                fid  = fid))

    def _parse_columns(self, df):
        re_report_timestamp = re.compile('.报告期.(\d{4})年(.*)')
        re_trans_timestamp  = re.compile('.*日期..*(\d{4})-(\d{2})-(\d{2})')
        re_topn_name = re.compile('.名次.*第(\d+)名')
        re_unit_name = re.compile('.单位.')

        k_report_mapping = {
                '一季':       '0331',
                '二季/中报':  '0630',
                '三季':       '0930',
                '年报':       '1231',
                }

        result = []
        for col in df.columns:
            tokens = []
            token_data = []
            for tok in col.split():
                m = re_report_timestamp.match(tok)
                if m:
                    year, rpt = m.groups()
                    date = k_report_mapping[rpt]
                    token_data.append(('date', '%s%s' % (year, date)))
                    continue

                m = re_trans_timestamp.match(tok)
                if m:
                    year, month, date = m.groups()
                    token_data.append(('date', '%s%s%s' % (year, month, date)))
                    continue

                m = re_topn_name.match(tok)
                if m:
                    topn = int(m.groups()[0])
                    token_data.append(('top', topn))
                    continue

                m = re_unit_name.match(tok)
                if m:
                    continue

                tokens.append(tok)

            new_col = ' '.join(map(str, tokens))
            if token_data:
                new_col = (new_col, json.dumps(dict(token_data)))
            result.append(new_col)
        return result

    def load_files(self):
        self._download_files()

        def load_df():
            all_df = {}
            for name, fid in self.k_files.items():
                name += '.xls'
                df = pd.read_excel(os.path.join(self.k_cached, name))
                df.columns = self._parse_columns(df)

                all_df[name] = df
            return all_df

        self.df = utils.pickle_cache(os.path.join(self.k_cached, 'all_data.pkl'), load_df)

    def index_data(self):
        utils.pickle_cache(os.path.join(self.k_cached, 'funds_stats.pkl'), lambda :
                self.index_data_by_column_date([df for name, df in self.df.items()
                                                if name.startswith('funds')]))

        return
        col2sub = defaultdict(list)
        for name, df in self.df.items():
            for col in df.columns:
                if isinstance(col, (tuple, list)):
                    colname = col[0]
                    coltail = col[1:]
                else:
                    colname = col
                    coltail = None


        import pprint
        pprint.pprint(col2sub)

    def index_data_by_column_date(self, dfs):
        result = defaultdict(dict)
        for df in dfs:
            indexed_columns = []

            for col in df.columns:
                if not isinstance(col, (tuple, list)):
                    continue
                colname, colmeta = col
                colmeta = json.loads(colmeta)

                if 'date' not in colmeta:
                    continue

                date = datetime.datetime.strptime(colmeta['date'], '%Y%m%d')
                new_day = calendar.monthrange(date.year, date.month)[1]
                colmeta['date'] = datetime.date(date.year, date.month, new_day)

                indexed_columns.append((col, colname, colmeta))

            for index, row in df.iterrows():
                for col, colname, colmeta in indexed_columns:
                    date = colmeta['date']
                    suffix = '_'.join('%s%s' % (k, v) for k, v in colmeta.items() if k != 'date')
                    if suffix:
                        colname += '_' + suffix

                    key = (row['证券代码'], date)
                    result[key][colname] = row[col]

        rows = []
        for k, v in result.items():
            v['证券代码'], v['date'] = k
            rows.append(v)

        result = pd.DataFrame(rows)
        return result

if __name__ == '__main__':
    database = Database()
    database.load_files()
    database.index_data()

