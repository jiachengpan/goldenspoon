#! /usr/bin/env python3

import os
import re
import glob
from xml.etree.ElementInclude import include
import pandas as pd
import pickle
import pprint
import json
from collections import defaultdict

def summary_one(filename, model_names, include_stocks):
    stats = pickle.load(open(filename, 'rb'))

    month_name = filename.split('/')[-2]
    model_name = filename.split('/')[-6]
    std_scale  = filename.split('/')[-5]
    label_norm = filename.split('/')[-4]
    run_id     = filename.split('/')[-3]
    model_type = filename.split('.')[-2]

    model_params = os.path.join(os.path.dirname(filename), 'model_params.json')
    model_params = json.load(open(model_params, 'r'))
    model_params['names'] = model_names
    #model_params = json.dumps(model_params, indent=2)

    params = {}
    if isinstance(model_params, list):
        assert len(model_names) == len(model_params)
        for i, kv in enumerate(model_params):
            params[(f'params{i}', 'name')] = model_names[i]
            for k, v in kv.items():
                params[(f'params{i}', k)] = v
    elif isinstance(model_params, dict):
        params[('params', 'name')] = model_names
        for k, v in model_params.items():
            params[('params', k)] = v
    else:
        assert 0, f'Unknown model_params type: {type(model_params)}'

    accuracy = {}
    for acc_type, acc_values in stats['accuracy'].items():
        values = defaultdict(lambda : None)
        values.update(acc_values)

        accuracy[(acc_type, 'acc')]          = values['acc']
        accuracy[(acc_type, 'acc_ex_small')] = values['exclude_small']
        accuracy[(acc_type, 'num')]          = '{T_num} / {F_num} / {F_exclude_num}'.format_map(values)

    profit = {}
    for profit_type, profit_values in stats.get('profit', {}).items():
        values = defaultdict(lambda : None)
        values.update(profit_values)

        profit[(profit_type, 'profit')]        = values['profit']
        profit[(profit_type, 'tops10_profit')] = values['tops10_profit']
        profit[(profit_type, 'tops20_profit')] = values['tops20_profit']

        if include_stocks:
            values = dict(values)
            profit[(profit_type, 'tops20_details_id')] = json.dumps([
                detail['id'] for detail in values.get('tops20_details', [])])

            def process_detail(k, v):
                if k in ('id', 'true_c', 'pred_prob'):
                    return v
                elif k == 'true_r':
                    assert len(v) == 1
                    return v[0]
            profit[(profit_type, 'tops20_details')] = json.dumps([
                {k: process_detail(k, v) for k, v in detail.items() if process_detail(k, v)}
                for detail in values.get('tops20_details', [])])

    record = {}
    record[('model', 'name')]       = model_name
    record[('model', 'type')]       = model_type
    record[('model', 'month')]      = month_name
    record[('model', 'std_scale')]  = std_scale
    record[('model', 'label_norm')] = label_norm
    record[('model', 'params')]     = model_params
    record[('model', 'run_id')]     = run_id

    record.update(params)
    record.update(profit)
    record.update(accuracy)

    record = {'_'.join(k): v for k, v in record.items()}

    return record

def summary(args):
    records = []

    for path in args.path:
        files = glob.glob(os.path.join(path, '**/*.pkl'), recursive=True)
        for filename in files:
            model_file = re.match(f'.*{args.models}/(.*[^/]+)\.pkl', os.path.dirname(filename)).group(1)
            model_data = pickle.load(open(os.path.join(args.models, f'{model_file}.pkl'), 'rb'))
            model_names = [model_data['name']]

            records.append(summary_one(filename, model_names, args.include_stocks))

    df = pd.DataFrame(records)
    print(df)
    df.to_excel(args.output, index=False)
    df.to_pickle(args.output.replace('.xlsx', '.pkl'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path',             type=str, nargs="+")
    parser.add_argument('--output', '-o',   type=str, default='summary.xlsx')
    parser.add_argument('--models',         type=str, default='output/models')
    parser.add_argument('--include-stocks', action='store_true', default=False)
    args = parser.parse_args()

    summary(args)
