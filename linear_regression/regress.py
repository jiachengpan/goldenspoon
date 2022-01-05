import pandas as pd
import numpy as np

from sklearn import linear_model
import os
from csv import writer
from data_preprocess import DataPreprocess
import argparse
from result_analysis import drop_small_change_stock_fntrain, drop_small_change_stock_fntest, perf_measure, perf_measure_per_stock


def get_args():
    parser = argparse.ArgumentParser(description='Goldenspoon Regression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_path",
        default='regress_data/past_quater_4/',
        help="Provide regression data path.")
    parser.add_argument(
        "--save_path",
        default='regress_result/past_quater_4/',
        help="Provide regression result path.")

    parser.add_argument(
        "--indicator_use_list",
        nargs="+",
        help="Set the indicators you want to train.",
        default=[])

    parser.add_argument(
        "--train_date_list",
        help="Set the enddate data for the training set. e.g. ['2021-03-31','2021-06-30'].",
        nargs="+",
        default=['2021-06-30'])
    parser.add_argument(
        "--test_date_list",
        nargs="+",
        help="Set the enddate data for the test set.",
        default=['2021-09-30'])

    parser.add_argument(
        "--n_month_predict",
        type=int,
        default=3,
        help="N month to predict.")
    parser.add_argument(
        "--predict_mode", 
        default="predict_changerate_price",
        help="Using 'predict_changerate_price' is currently the best approach.",
        choices=['predict_changerate_price', 'predict_absvalue_price'])

    parser.add_argument(
        "--drop_small_change_stock_fortrain",
        type=bool,
        default=True,
        help="Drop small change stock for train data.")
    parser.add_argument(
        "--train_drop_ponit",
        type=float,
        default=0.15,
        help="drop ponit for task 'drop_small_change_stock_fortrain'.")
    parser.add_argument(
        "--drop_small_change_stock_fortest",
        type=bool,
        default=True,
        help="Drop small change stock for test data.")
    parser.add_argument(
        "--test_drop_ponit",
        type=float,
        default=0.15,
        help="drop ponit for task 'drop_small_change_stock_fortest'.")
    return parser.parse_args()


class Regress():
    def __init__(self, regress_model, save_path_permonth, i_month_label, indicator_list, drop_small_change_stock_fortrain, drop_small_change_stock_fortest, train_drop_ponit, test_drop_ponit):
        self.model = regress_model
        self.save_path_permonth = save_path_permonth
        # '1_predict_changerate_price' or '1_predict_absvalue_price'
        self.i_month_label = i_month_label
        self.indicator_list = indicator_list
        self.drop_small_change_stock_fortrain = drop_small_change_stock_fortrain
        self.drop_small_change_stock_fortest = drop_small_change_stock_fortest
        self.train_drop_ponit = train_drop_ponit
        self.test_drop_ponit = test_drop_ponit

    # Linear regression
    def run(self, data_path, train_date_list, test_date_list, n_month_predict):
        # 1.数据预处理
        x_train_, x_test_, y_train_, y_test_, train_ID_df, test_ID_df = DataPreprocess(
            data_path, train_date_list, test_date_list, n_month_predict, self.i_month_label, self.indicator_list).run()

        # 2.获取训练集的label、测试集的label、测试集的label对应的股票ID
        ## 训练集重映射
        if self.drop_small_change_stock_fortrain:
            ## 需要按照y_train大于drop_ponit做判断, 保留x_train和y_train中涨跌幅绝对值大于drop_ponit的数据
            y_train_ = y_train_[self.i_month_label]
            y_trainID_ = train_ID_df['id']
            valid_train_stock_list, y_train, x_train, train_stock_id_valid = drop_small_change_stock_fntrain(
                y_train=y_train_, x_train=x_train_, drop_ponit=self.train_drop_ponit, train_stock_id=y_trainID_)
        else:
            x_train = x_train_
            y_train = y_train_[self.i_month_label]
        ## 测试集重映射
        cur_stock_price_test = y_test_['0_label']
        x_test = x_test_
        y_test = y_test_[self.i_month_label]
        y_testID = test_ID_df['id']

        # 3.模型训练
        print("-----x_train.shape:{}\n".format(x_train.shape))
        print("-----x_test.shape:{}\n".format(x_test.shape))
        print("-----y_train.shape:{}\n".format(y_train.shape))
        print("-----y_test.shape:{}\n".format(y_test.shape))
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        # R2_y_pred_y_test = r2_score(y_test, y_pred)
        indicator_weight = list(
            zip(self.indicator_list, list(self.model.coef_)))

        ## 打印回归模型权重参数
        with open(self.save_path_permonth + 'indicator_weight.log', 'w') as f:
            for i_weight in indicator_weight:
                f.write(str(i_weight))
                f.write('\n')
            f.close()

        # 4.结果分析
        if self.drop_small_change_stock_fortest:
            valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                y_pred=y_pred, y_true=y_test, drop_ponit=self.test_drop_ponit, test_stock_id=y_testID)
            full_stock_list = y_test.shape[0]
            
            TP, FP, TN, FN = perf_measure(
                y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label, self.save_path_permonth)
            stock_pred_correctness = perf_measure_per_stock(
                full_stock_list, valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid)
            with open(self.save_path_permonth + 'stock_pred_correctness.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(stock_pred_correctness)
                f_object.close()
            print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        else:
            assert(0)
            TP, FP, TN, FN = perf_measure(y_pred, y_test, cur_stock_price_test, self.i_month_label, self.save_path_permonth)
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
    drop_small_change_stock_fortrain = args.drop_small_change_stock_fortrain
    drop_small_change_stock_fortest = args.drop_small_change_stock_fortest
    train_drop_ponit = args.train_drop_ponit
    test_drop_ponit = args.test_drop_ponit

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
                    i_month_label, indicator_list,
                    drop_small_change_stock_fortrain, drop_small_change_stock_fortest, 
                    train_drop_ponit, test_drop_ponit)
        r.run(data_path, train_date_list, test_date_list, n_month_predict)
