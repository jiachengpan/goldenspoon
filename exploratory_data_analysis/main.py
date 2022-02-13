import os
import sys

sys.path.append('..')

import dataset
import analysis
import model

k_data_path = '../generate_data/data/thres_inf/'
k_train_dates = [
    #'2020-03-31',
    '2020-06-30',
    '2020-09-30',
    '2020-12-31',
    '2021-03-31',
    '2021-06-30',
]

#k_test_dates = ['2021-06-30']
k_test_dates = ['2021-09-30']
#k_test_dates = ['2021-12-31']

k_change_threshold_train = -1e-9
k_change_threshold_test  = -1e-9

k_cutpoints = [-0.2, -0.1, 0.1, 0.15, 0.2, 0.25, 0.3]
k_stocks_only = '.SH'

def run_model(month, data, args):
    print(f'month: {month} data: {data}')
    m = model.get_model('baseline')
    m.fit(data.x_train, data.y_train_cls)
    return analysis.test_model(m, data, args)

def main(args):
    ds_args = dataset.Args(
        data_path   = args.data_path,
        train_dates = args.train_dates,
        test_dates  = args.test_dates,
        classifier_cut_points = k_cutpoints,
        stocks_only = k_stocks_only,
        debug       = args.verbose,
    )

    ds = dataset.DataSet.create(ds_args)

    task_params = []
    for month, data in ds.items():
        task_params.append((month, data, ds_args))

    import multiprocessing as mp
    pool = mp.Pool(args.jobs)
    result = pool.starmap(run_model, task_params)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',      type=str, default='data')
    parser.add_argument('--train-dates',    type=int, default=k_train_dates, nargs='+')
    parser.add_argument('--test-dates',     type=str, default=k_test_dates, nargs='+')
    parser.add_argument('--predict-mode',   type=str, default='predict_changerate_price')
    parser.add_argument('--jobs', '-j',     type=int, default=1)
    parser.add_argument('--verbose', '-v',  action='store_true', default=False)

    args = parser.parse_args()

    main(args)

