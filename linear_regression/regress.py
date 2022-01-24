import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
import os
import csv
from data_preprocess import DataPreprocess
import argparse
from result_analysis import drop_small_change_stock_fntrain, drop_small_change_stock_fntest, perf_measure, perf_measure_per_stock, draw_confusion_matrix
from pandas.testing import assert_frame_equal


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
        "--indicator_use_type",
        help="Set the type of indicators you want to train.",
        default='all',
        choices=['all', 'all_with_premonth_label', 'industrial_static', 'industrial_dynamic', 'stock_dynamic', 'static_stock_dynamic','all_dynamic'])

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

    parser.add_argument(
        "--training_model",
        default="linear",
        help="The model for training.",
        choices=['linear', 'ridge', 'lasso', 'randomforest', 'randomforestclssifier', 'adaboostregressor', 'adaboostclassifier'])
    parser.add_argument(
        "--label_type",
        default="class",
        help="Label type.",
        choices=['regress','class'])
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

    def result_analysis(self, y_pred, y_test, y_testID, label_type, cur_stock_price_test=None, data_type='test'):
        if label_type == 'class':
            acc_small, acc_mid_positive, acc_big_positive, acc_mid_negative, acc_big_negative, acc_positive, acc_negative = perf_measure(
                y_pred, y_test, cur_stock_price_test, self.i_month_label)
            print("label type is class!")
            print("acc_small:{}, acc_mid_positive:{}, acc_big_positive:{}, acc_mid_negative:{}, acc_big_negative:{}".format(acc_small, acc_mid_positive, acc_big_positive, acc_mid_negative, acc_big_negative))
            print("acc_positive:{}, acc_negative:{}".format(acc_positive, acc_negative))
            return
        
        print("label type is regress!")
        if data_type == 'test':
            print("\n-----test data analysis:")
            if self.drop_small_change_stock_fortest:
                valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                    y_pred=y_pred, y_true=y_test, drop_ponit=self.test_drop_ponit, test_stock_id=y_testID)
                full_stock_list = y_test.shape[0]

                TP, FP, TN, FN = perf_measure(
                    y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label)
                draw_confusion_matrix(TP, FP, TN, FN, self.save_path_permonth)
                stock_pred_correctness = perf_measure_per_stock(
                    full_stock_list, valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid)

                stock_pred_correctness_df = pd.DataFrame(stock_pred_correctness)
                stock_pred_correctness_df.to_csv(
                    os.path.join(self.save_path_permonth, 'stock_pred_correctness.tsv'),
                    sep='\t',
                    quoting=csv.QUOTE_NONE,
                    index=False)

                if not stock_pred_correctness_df.empty:
                    stock_is_SH = stock_pred_correctness_df['id'].str.endswith('SH')
                    stock_pred_topN_df = stock_pred_correctness_df[stock_is_SH].sort_values('pred', ascending=False)[:10]
                    stock_pred_top10P_df = stock_pred_topN_df[stock_pred_topN_df.pred > 0 ]

                    stock_pred_topN_df.to_csv(
                        os.path.join(self.save_path_permonth, 'stock_pred_topN.tsv'),
                        sep='\t',
                        quoting=csv.QUOTE_NONE,
                        index=False)
                    print("top10\n", stock_pred_topN_df)
                    # print("-----top10-describe, include pred N\n", stock_pred_topN_df.describe())
                    print("top10-describe, exclude Negative Pred!\n", stock_pred_top10P_df.describe())
                else:
                    print("stock_pred_correctness_df is a empty dataframe.")
                print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        elif data_type == 'train':
            print("\n-----train data analysis:")
            valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                y_pred=y_pred, y_true=y_test, drop_ponit=0.0, test_stock_id=y_testID)

            TP, FP, TN, FN = perf_measure(
                y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label)
            print("no drop:")
            print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
            
            valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                y_pred=y_pred, y_true=y_test, drop_ponit=0.15, test_stock_id=y_testID)

            TP, FP, TN, FN = perf_measure(
                y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label)
            print("drop 0.15:")
            print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        return
    
    def data_processing_for_run(self,x_train_,x_test_,y_train_,y_test_,train_ID_df,test_ID_df):
        global pretrain_month_label
        global pretest_month_label
        if indicator_use_type == 'all_with_premonth_label' and i_month_predict > 1:
            pre_month_label = str(i_month_predict-1) + '_predict_changerate_price'

            ## 获取1月的label作为2月的indicator
            assert_frame_equal(x_train_[['id']], pretrain_month_label[['id']])
            x_train_ = pd.concat([x_train_, pretrain_month_label[[pre_month_label]]], axis=1)
            assert_frame_equal(x_test_[['id']], pretest_month_label[['id']])
            x_test_ = pd.concat([x_test_, pretest_month_label[[pre_month_label]]], axis=1)

            new_train_ID_df = x_train_.iloc[:,x_train_.columns.str.endswith('id')]
            new_test_ID_df = x_test_.iloc[:,x_test_.columns.str.endswith('id')]

            assert_frame_equal(new_train_ID_df, train_ID_df)
            assert_frame_equal(new_test_ID_df, test_ID_df)

            x_train_ = x_train_.drop(columns=['id'])
            x_test_ = x_test_.drop(columns=['id'])

            pretrain_month_label = y_train_regress_label
            pretest_month_label = y_test_regress_label
        else:
            pretrain_month_label = y_train_regress_label
            pretest_month_label = y_test_regress_label
            new_train_ID_df = x_train_.iloc[:,x_train_.columns.str.endswith('id')]
            new_test_ID_df = x_test_.iloc[:,x_test_.columns.str.endswith('id')]

            assert_frame_equal(new_train_ID_df, train_ID_df)
            assert_frame_equal(new_test_ID_df, test_ID_df)

            x_train_ = x_train_.drop(columns=['id'])
            x_test_ = x_test_.drop(columns=['id'])

        # 2.获取训练集的label、测试集的label、测试集的label对应的股票ID
        # - 训练集重映射
        if (self.drop_small_change_stock_fortrain or (self.train_drop_ponit==0.0)) and (label_type != 'class'):
            ## 需要按照y_train大于drop_ponit做判断, 保留x_train和y_train中涨跌幅绝对值大于drop_ponit的数据
            y_train_ = y_train_[self.i_month_label]
            y_trainID = train_ID_df['id']
            valid_train_stock_list, y_train, x_train, train_stock_id_valid = drop_small_change_stock_fntrain(
                y_train=y_train_, x_train=x_train_, drop_ponit=self.train_drop_ponit, train_stock_id=y_trainID)
        else:
            x_train = x_train_
            y_train = y_train_[self.i_month_label]
            y_trainID = train_ID_df['id']
        # - 测试集重映射
        x_test = x_test_
        y_test = y_test_[self.i_month_label]
        y_testID = test_ID_df['id']
        return x_train,x_test,y_train,y_test,y_trainID,y_testID

    # Linear regression
    def run(self, data_path, train_date_list, test_date_list, n_month_predict):
        # 1.数据预处理
        print("---------------------- data preprocess ----------------------")
        global y_train_regress_label
        global y_test_regress_label
        x_train_, x_test_, y_train_, y_test_, train_ID_df, test_ID_df, y_train_regress_label, y_test_regress_label = DataPreprocess(
            data_path, train_date_list, test_date_list, n_month_predict,
            self.i_month_label, self.indicator_list,
            label_type).run()

        if label_type == 'regress':
            assert(y_train_regress_label==None)
            y_train_regress_label = y_train_
            y_test_regress_label = y_test_
        
        x_train, x_test, y_train, y_test, y_trainID, y_testID = self.data_processing_for_run(x_train_,x_test_,y_train_,y_test_,train_ID_df,test_ID_df)


        # 3.模型训练
        print("---------------------- model runing ----------------------")
        print("-----x_train.shape:{}\n".format(x_train.shape))
        print("-----x_train columns:{}\n".format(x_train.columns.values.tolist()))
        print("-----x_test.shape:{}\n".format(x_test.shape))
        print("-----x_test columns:{}\n".format(x_test.columns.values.tolist()))

        print("-----y_train.shape:{}\n".format(y_train.shape))
        # print("-----y_train columns:{}\n".format(y_train.columns.values.tolist()))
        print("-----y_test.shape:{}\n".format(y_test.shape))
        # print("-----y_test columns:{}\n".format(y_test.columns.values.tolist()))
        print("x_train:",x_train)
        print("y_train:",y_train)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        if args.training_model in ['linear', 'ridge', 'lasso']:
            indicator_weight = list(
                zip(self.indicator_list, list(self.model.coef_)))
        elif args.training_model in ['randomforest', 'randomforestclssifier', 'adaboostregressor', 'adaboostclassifier']:
            indicator_weight = list(
                zip(self.indicator_list, list(self.model.feature_importances_)))
        else:
            assert("indicator_weight error!")

        # - 打印回归模型权重参数
        with open(self.save_path_permonth + 'indicator_weight.log', 'w') as f:
            for i_weight in indicator_weight:
                f.write(str(i_weight))
                f.write('\n')
            f.close()

        # 4.结果分析
        print("---------------------- result analysis ----------------------")
        # - test data analysis
        self.result_analysis(y_pred, y_test, y_testID, label_type, cur_stock_price_test=y_test_['0_label'], data_type='test')
        # - train data analysis
        y_pred_fortrain = self.model.predict(x_train)
        y_test_fortrain = y_train
        self.result_analysis(y_pred=y_pred_fortrain, y_test=y_test_fortrain, y_testID=y_trainID, label_type=label_type, cur_stock_price_test=None, data_type='train')

if __name__ == '__main__':
    # base parameter
    print("***********************************************************")
    args = get_args()
    for arg in vars(args):
        print(format(arg, '<40'), format(" -----> " + str(getattr(args, arg)), '<'))
    print('\n')
    data_path = args.data_path
    save_path = args.save_path

    n_month_predict = args.n_month_predict
    indicator_use_list = args.indicator_use_list
    global indicator_use_type
    indicator_use_type = args.indicator_use_type

    train_date_list = args.train_date_list
    test_date_list = args.test_date_list

    # 'predict_changerate_price' or 'predict_absvalue_price'
    predict_mode = args.predict_mode
    drop_small_change_stock_fortrain = args.drop_small_change_stock_fortrain
    drop_small_change_stock_fortest = args.drop_small_change_stock_fortest
    train_drop_ponit = args.train_drop_ponit
    test_drop_ponit = args.test_drop_ponit

    label_type = args.label_type
    if args.training_model == 'linear':
        regress_model = linear_model.LinearRegression()
    elif args.training_model == 'ridge':
        regress_model = linear_model.Ridge()
    elif args.training_model == 'lasso':
        regress_model = linear_model.Lasso()
    elif args.training_model == 'randomforest':
        regress_model = ensemble.RandomForestRegressor()
        label_type = 'regress'
    elif args.training_model == 'randomforestclssifier':
        regress_model = ensemble.RandomForestClassifier()
        label_type = 'class'
    elif args.training_model == 'adaboostregressor':
        base_estimator = ensemble.RandomForestRegressor(n_estimators=10, max_depth=10)
        regress_model = ensemble.AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, learning_rate=0.8)
        label_type = 'regress'
    elif args.training_model == 'adaboostclassifier':
        base_estimator = ensemble.RandomForestClassifier(n_estimators=10, max_depth=10)
        # base_estimator = tree.DecisionTreeClassifier(max_features=None, max_depth=30)
        regress_model = ensemble.AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, learning_rate=0.8)#algorithm="SAMME"
        label_type = 'class'
    else:
        assert("The training model was not implemented.")

    if (indicator_use_type == 'all') or (indicator_use_type == 'all_with_premonth_label'):
        # 如果没有指定type,同时没有使用的indicator列表，则使用全部的indicator训练
        if indicator_use_list == []:
            indicator_pickle = args.data_path + \
                'indicators.' + train_date_list[0] + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            all_indicator_list = list(df_indicator.columns.values)[
                1:]  # exclude 'id'
            indicator_list = all_indicator_list
            print("-----indicator_list:{}\n".format(indicator_list))
            # assert(0)
        else:
            indicator_list = indicator_use_list
    elif indicator_use_type == 'industrial_static':
        indicator_list = ['是', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股']
    elif indicator_use_type == 'industrial_dynamic':
        indicator_list = ['market_value_mean', 'market_value_std', 'fund_shareholding_mean', 'fund_shareholding_std', 'fund_number_mean', 'fund_number_std']
    elif indicator_use_type == 'stock_dynamic':
        indicator_list = ['close_price_mean', 'close_price_std', 'avg_price_mean', 'avg_price_std', \
                        'turnover_rate_mean', 'turnover_rate_std', 'amplitutde_mean', 'amplitutde_std',\
                        'margin_diff_mean', 'margin_diff_std', 'share_ratio_of_funds_mean', 'share_ratio_of_funds_std', 'num_of_funds_mean',\
                        'num_of_funds_std', 'fund_owner_affinity_mean', 'fund_owner_affinity_std', 'cyclical_industry']
    elif indicator_use_type == 'static_stock_dynamic':
        indicator_list = ['是', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股', \
                        'close_price_mean', 'close_price_std', 'avg_price_mean', 'avg_price_std', \
                        'turnover_rate_mean', 'turnover_rate_std', 'amplitutde_mean', 'amplitutde_std',\
                        'margin_diff_mean', 'margin_diff_std', 'share_ratio_of_funds_mean', 'share_ratio_of_funds_std', 'num_of_funds_mean',\
                        'num_of_funds_std', 'fund_owner_affinity_mean', 'fund_owner_affinity_std', 'cyclical_industry']
    elif indicator_use_type == 'all_dynamic':
        indicator_list = ['market_value_mean', 'market_value_std', 'fund_shareholding_mean', 'fund_shareholding_std', 'fund_number_mean', 'fund_number_std', \
                        'close_price_mean', 'close_price_std', 'avg_price_mean', 'avg_price_std', \
                        'turnover_rate_mean', 'turnover_rate_std', 'amplitutde_mean', 'amplitutde_std',\
                        'margin_diff_mean', 'margin_diff_std', 'share_ratio_of_funds_mean', 'share_ratio_of_funds_std', 'num_of_funds_mean',\
                        'num_of_funds_std', 'fund_owner_affinity_mean', 'fund_owner_affinity_std', 'cyclical_industry']
    else:
        assert("Wrong indicator list!")

    train_month_label = pd.DataFrame()
    print("-----train_month_label:", train_month_label)
    test_month_label = pd.DataFrame()
    print("-----test_month_label:", test_month_label)
    global i_month_predict
    for i in range(n_month_predict):
        i_month_predict = i+1
        i_month_label = str(i_month_predict) + '_' + predict_mode
        print("************************** {} : {} **************************".format(i_month_predict, i_month_label))

        save_path_permonth = save_path + '/' + str(i_month_predict) + '_month/'
        if not os.path.exists(save_path_permonth):
            os.makedirs(save_path_permonth)

        r = Regress(regress_model, save_path_permonth,
                    i_month_label, indicator_list,
                    drop_small_change_stock_fortrain, drop_small_change_stock_fortest,
                    train_drop_ponit, test_drop_ponit)
        r.run(data_path, train_date_list, test_date_list, n_month_predict)
