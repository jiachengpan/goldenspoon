#! /usr/bin/env python3

import os
import re
import glob
import pandas as pd
import pickle
import pprint
import json
import itertools
from collections import defaultdict

from sklearn import ensemble

def concat_files(files):
    dfs = []
    for file in files:
        df = pickle.load(open(file, 'rb'))
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def analyze_models(df, metrics):
    df.params_name = df.params_name.astype(str)
    grp = df.groupby(by=['model_name', 'model_type', 'model_month', 'params_name'])
    model_df = grp.last()
    model_df[metrics] = grp.mean()[metrics]

    model_df = model_df.reset_index()
    model_df = model_df[model_df.model_type == 'test'][['model_name', 'model_month', 'params_name', metrics]]

    result = defaultdict(dict)
    for (month, model), df in model_df.groupby(by=['model_month', 'params_name']):
        df_month = df.sort_values(by=[metrics], ascending=False)
        #print(month, model)
        #print(df_month)

        result[month][model] = df_month
    return dict(result)

def collect_topn_models(models, topn = 1):
    result = {}
    for model_type, model_df in models.items():
        models = []
        for id, model_info in enumerate(model_df.head(topn).to_dict('records')):
            matched = re.match(r'.*\.pkl', model_info['model_name'])
            assert matched

            model_name = matched.group(0)
            model_data = pickle.load(open(os.path.join('output/models', model_name), 'rb'))

            model_data['params']['filename'] = model_name
            models.append(model_data)
        result[model_type] = models
    return result

def generate_voting_models(topn, models, max_size = 5):
    topn_models = [m[:topn] for m in models.values()]
    topn_models = [mm for m in topn_models for mm in m]
    for selected_count in range(2, min(len(topn_models) + 1, max_size+1)):
        for selected_models in itertools.combinations(topn_models, selected_count):
            estimators = [(m['params']['filename'], m['model']) for m in selected_models]
            params = dict(
                model_selected = [m['params']['filename'] for m in selected_models],
                model_params = [m['params'] for m in selected_models],
                model_policy = f'top{topn}_select{selected_count}',
            )
            yield params, ensemble.VotingClassifier(estimators=estimators, voting='soft')

def main(args):
    df  = concat_files(args.files)
    res = analyze_models(df, args.metrics)

    all_models = set()
    unique_id  = defaultdict(lambda : 0)

    os.makedirs(args.output, exist_ok=True)
    for month, models in res.items():
        models = collect_topn_models(models, topn = 3)
        for topn in (1, 3):
            for params, model in generate_voting_models(topn = topn, models = models):
                key = ';;'.join(params['model_selected'])
                if key in all_models:
                    continue
                all_models.add(key)

                model_policy = params["model_policy"]
                model_name = f'{model_policy}.{unique_id[model_policy]}'
                output_name = os.path.join(args.output, f'{model_name}.pkl')
                unique_id[model_policy] += 1
                # print(model_name)
                # pprint.pprint(params)

                data = {'name': model_name, 'model': model, 'params': params}
                pickle.dump(data, open(output_name, 'wb'))

    pprint.pprint(unique_id)
    print(sum(unique_id.values()))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files',        type=str, nargs='+')
    parser.add_argument('--metrics',    type=str, default='big_positive_tops10_profit')
    parser.add_argument('--output',     type=str, default='output/voting_models')
    args = parser.parse_args()

    main(args)