from cProfile import label
import os
import glob
import pandas as pd
import pickle
import pprint
import json
from collections import defaultdict

def summary_one(filename):
    stats = pickle.load(open(filename, 'rb'))
    print(filename)

    month_name = filename.split('/')[-2]
    label_norm = filename.split('/')[-4]
    std_scale  = filename.split('/')[-5]
    model_name = filename.split('/')[-6]
    model_type = filename.split('.')[-2]

    model_params = os.path.join(os.path.dirname(filename), 'model_params.json')
    model_params = json.load(open(model_params, 'r'))
    #model_params = {('model_params', k): v for k, v in model_params.items()}
    model_params = json.dumps(model_params, indent=2)

    accuracy = {}
    for acc_type, acc_values in stats['accuracy'].items():
        values = defaultdict(lambda : None)
        values.update(acc_values)

        accuracy[(month_name, acc_type, 'acc')]          = values['acc']
        accuracy[(month_name, acc_type, 'acc_ex_small')] = values['exclude_small']
        accuracy[(month_name, acc_type, 'num')]          = '{T_num} / {F_num} ({F_exclude_num})'.format_map(values)

    profit = {}
    for profit_type, profit_values in stats.get('profit', {}).items():
        values = defaultdict(lambda : None)
        values.update(profit_values)

        profit[(month_name, profit_type, 'profit')]        = values['profit']
        profit[(month_name, profit_type, 'tops10_profit')] = values['tops10_profit']
        profit[(month_name, profit_type, 'tops20_profit')] = values['tops20_profit']

    record = {}
    record[('model', 'name')]       = model_name
    record[('model', 'type')]       = model_type
    record[('model', 'std_scale')]  = std_scale
    record[('model', 'label_norm')] = label_norm
    record[('model', 'params')]     = model_params
    #record[('run', 'month')]        = month_name

    record.update(profit)
    record.update(accuracy)

    record = {'_'.join(k): v for k, v in record.items()}

    return record

def summary(args):
    with open(args.stats) as fh:
        stats = [s.strip() for s in fh.readlines()]

    records = []
    for filename in stats:
        records.append(summary_one(filename))

    df = pd.DataFrame(records)
    #df.columns = pd.MultiIndex.from_arrays(list(zip(*df.columns.tolist())))
    df.to_excel(args.output, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stats',            type=str, default='stats.txt')
    parser.add_argument('--output', '-o',   type=str, default='summary.xlsx')
    args = parser.parse_args()
    summary(args)