import pandas as pd
import numpy as np

from sklearn import linear_model
import os
from csv import writer
from data_preprocess import DataPreprocess
import argparse
from result_analysis import drop_small_change_stock, perf_measure, perf_measure_per_stock


def get_args():
    parser = argparse.ArgumentParser(description='Goldenspoon Regression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",
                        default='regress_data/past_quater_4/',
                        help="Provide regression data path.")
    parser.add_argument("--save_path",
                        default='regress_result/past_quater_4/',
                        help="Provide regression result path.")
    parser.add_argument(
        "--indicator_use_list",
        nargs="+",
        default=[])
    parser.add_argument(
        "--train_date_list",
        nargs="+",
        default=['2021-06-30'])
    parser.add_argument(
        "--test_date_list",
        nargs="+",
        default=['2021-09-30'])
    parser.add_argument(
        "--n_month_predict",
        type=int,
        default=3,
        help="N month to predict.")
    parser.add_argument("--predict_mode", default="predict_changerate_price",
                        choices=['predict_changerate_price', 'predict_absvalue_price'])
    return parser.parse_args()


class Regress():
    def __init__(self, regress_model, save_path_permonth, i_month_label, indicator_list):
        self.model = regress_model
        self.save_path_permonth = save_path_permonth
        # '1_predict_changerate_price' or '1_predict_absvalue_price'
        self.i_month_label = i_month_label
        self.indicator_list = indicator_list

    # Linear regression
    def run(self, data_path, train_date_list, test_date_list, n_month_predict):
        # 数据预处理
        x_train, x_test, y_train, y_test, test_ID_df = DataPreprocess(
            data_path, train_date_list, test_date_list, n_month_predict, self.i_month_label, self.indicator_list).run()
        print("-----x_train.shape:{}\n".format(x_train.shape))
        print("-----x_test.shape:{}\n".format(x_test.shape))
        print("-----y_train.shape:{}\n".format(y_train.shape))
        print("-----y_test.shape:{}\n".format(y_test.shape))

        # 获取训练集的label、测试集的label、测试集的label对应的股票ID
        cur_stock_price_test = y_test['0_label']
        y_train = y_train[self.i_month_label]
        y_test = y_test[self.i_month_label]
        y_testID = test_ID_df['id']

        # 模型训练
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        # R2_y_pred_y_test = r2_score(y_test, y_pred)
        indicator_weight = list(
            zip(self.indicator_list, list(self.model.coef_)))

        # 打印回归模型权重参数
        with open(self.save_path_permonth + 'indicator_weight.log', 'w') as f:
            for i_weight in indicator_weight:
                f.write(str(i_weight))
                f.write('\n')
            f.close()

        # 结果分析
        valid_stock_list, y_pred_valid, y_true_valid, y_stock_id_valid = drop_small_change_stock(
            y_pred=y_pred, y_true=y_test, drop_ponit=0.15, y_stock_id=y_testID)
        full_stock_list = y_test.shape[0]
        # TP, FP, TN, FN = perf_measure(y_pred, y_test, cur_stock_price_test)
        TP, FP, TN, FN = perf_measure(
            y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label, self.save_path_permonth)
        stock_pred_correctness = perf_measure_per_stock(
            full_stock_list, valid_stock_list, y_pred_valid, y_true_valid, y_stock_id_valid)
        with open(self.save_path_permonth + 'stock_pred_correctness.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(stock_pred_correctness)
            f_object.close()
        print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        # self.draw(y_pred, y_test, cur_stock_price_test, R2_y_pred_y_test, self.i_month_label, self.save_path_permonth, show_interval=50)
        return


if __name__ == '__main__':
    # base parameter
    args = get_args()
    data_path = args.data_path
    save_path = args.save_path
    n_month_predict = args.n_month_predict
    indicator_use_list = args.indicator_use_list
    train_date_list = args.train_date_list
    test_date_list = args.test_date_list
    # 'predict_changerate_price' or 'predict_absvalue_price'
    predict_mode = args.predict_mode
    regress_model = linear_model.LinearRegression()

    # 如果没有指定使用的indicator列表，则使用全部的indicator训练
    if indicator_use_list == []:
        indicator_pickle = args.data_path + \
            'indicators.' + train_date_list[0] + '.pickle'
        df_indicator = pd.read_pickle(indicator_pickle)
        all_indicator_list = list(df_indicator.columns.values)[
            1:]  # exclude 'id'
        indicator_list = all_indicator_list
        print("-----indicator_list:{}\n".format(indicator_list))
    else:
        indicator_list = indicator_use_list

    for i in range(n_month_predict):
        i_month_predict = i+1
        print("-----i_month_predict:{}\n".format(i_month_predict))
        i_month_label = str(i_month_predict) + '_' + predict_mode
        print("-----i_month_label:{}\n".format(i_month_label))

        save_path_permonth = save_path + '/' + str(i_month_predict) + '_month/'
        if not os.path.exists(save_path_permonth):
            os.makedirs(save_path_permonth)

        r = Regress(regress_model, save_path_permonth,
                    i_month_label, indicator_list)
        r.run(data_path, train_date_list, test_date_list, n_month_predict)
