import pandas as pd
import numpy as np
import json
import pickle

import sklearn
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import os
import csv
from data_preprocess import DataPreprocess
import argparse
from result_analysis import drop_small_change_stock_fntrain, drop_small_change_stock_fntest, perf_measure, perf_measure_per_stock, draw_confusion_matrix
from pandas.testing import assert_frame_equal
import heapq
from utils.logger import G_LOGGER

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

G_LOGGER.severity=G_LOGGER.INFO

indicator_map = {
    '是':'a_hushen',
    '成长型':'a_grow',
    '混合型':'a_mix',
    '价值型':'a_value',
    '小盘股':'a_small',
    '中盘股':'a_mid',
    '大盘股':'a_big',

    'market_value_mean':'b_value_m',
    'market_value_std':'b_value_s',
    'fund_shareholding_mean':'b_rfund_m',
    'fund_shareholding_full':'b_rfund_f',
    'fund_shareholding_std':'b_rfund_s',
    'fund_shareholding_partial':'b_rfund_p',
    'fund_number_mean':'b_nfund_m',
    'fund_number_std':'b_nfund_s',

    'close_price_mean':'c_close_m',
    'close_price_std':'c_close_s',
    'avg_price_mean':'c_avg_m',
    'avg_price_std':'c_avg_s',
    'turnover_rate_mean':'c_turn_m',
    'turnover_rate_std':'c_turn_s',
    'amplitutde_mean':'c_amp_m',
    'amplitutde_std':'c_amp_s',
    'margin_diff_mean':'c_mar_m',
    'margin_diff_std':'c_mar_s',
    'share_ratio_of_funds_mean':'c_rfund_m',
    'share_ratio_of_funds_std':'c_rfund_s',
    'num_of_funds_mean':'c_nfund_m',
    'num_of_funds_std':'c_nfund_s',
    'fund_owner_affinity_mean':'c_fund_m',
    'fund_owner_affinity_std':'c_fund_s',
    'cyclical_industry':'c_cyc_m'
}


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
        #choices=['linear', 'ridge', 'lasso', 'decisiontreeclassifier', 'randomforest', 'randomforestclssifier', 'adaboostregressor', 'adaboostclassifier','gbdtclassifier','votingclassifier'],
        )
    parser.add_argument(
        "--label_type",
        default="class",
        help="Label type.",
        choices=['regress','class'])

    parser.add_argument(
        "--data_standardscaler",
        action="store_true",
        help="Data standardscaler.")

    parser.add_argument(
        "--sample_number",
        type=int,
        default=1000,
        help="Sample number to sample class.")

    parser.add_argument(
        "--label_norm",
        action="store_true",
        help="Data standardscaler.")

    parser.add_argument(
        "--repeat_run",
        type=int,
        default=1,
        help="repeat run times.")
    return parser.parse_args()

class Regress():
    def __init__(self, regress_model, save_path_permonth, i_month_label, indicator_list, \
                    drop_small_change_stock_fortrain, drop_small_change_stock_fortest, \
                    train_drop_ponit, test_drop_ponit, sample_number, label_norm):
        self.model = regress_model
        self.save_path_permonth = save_path_permonth
        self.i_month_label = i_month_label # '1_predict_changerate_price' or '1_predict_absvalue_price'
        self.indicator_list = indicator_list
        self.drop_small_change_stock_fortrain = drop_small_change_stock_fortrain
        self.drop_small_change_stock_fortest = drop_small_change_stock_fortest
        self.train_drop_ponit = train_drop_ponit
        self.test_drop_ponit = test_drop_ponit
        self.sample_number = sample_number
        self.label_norm = label_norm

    def result_analysis(self, y_pred, y_test, y_testID, label_type, cur_stock_price_test=None, data_type='test',y_pred_prob=None):
        """
        1.label_type == 'class', 进行结果分析
        """
        global total_big_positive
        print('DBG: label_type:', label_type)
        if label_type == 'class':
            stats = {}
            if data_type == 'test':
                pre_month_label = str(i_month_predict) + '_predict_changerate_price'
                y_test_regress_value = y_test_regress_label[[pre_month_label]].values
                stock_id = y_test_regress_label[['id']].values
                df_big_positive = perf_measure(y_pred, y_test, cur_stock_price_test, self.i_month_label,
                                                y_test_regress_value,
                                                stock_id=stock_id,
                                                y_pred_prob=y_pred_prob,
                                                votingclassifier_y_pred_prob=self.votingclassifier_y_pred_prob,
                                                stats=stats)

                # deprecated, 观察big_positive在不同月份连续出现的频率
                df_big_positive.columns = ['id',pre_month_label]
                if total_big_positive.empty:
                    total_big_positive = df_big_positive
                else:
                    total_big_positive = pd.merge(total_big_positive,df_big_positive,how='outer')
            else:
                pre_month_label = str(i_month_predict) + '_predict_changerate_price'
                y_train_regress_value = y_train_regress_label[[pre_month_label]].values
                stock_id = y_train_regress_label[['id']].values
                df_big_positive = perf_measure(y_pred, y_test, cur_stock_price_test, self.i_month_label,
                                               y_true_regress_value=None,
                                               stock_id=None,
                                               stats=stats)

            with open(self.save_path_permonth + f'stats.{data_type}.pkl', 'wb') as f:
                pickle.dump(stats, f)

            return

        """
        1.label_type == 'regress', 进行结果分析
        """
        if data_type == 'test':
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
                    G_LOGGER.info("top10\n", stock_pred_topN_df)
                    # print("-----top10-describe, include pred N\n", stock_pred_topN_df.describe())
                    G_LOGGER.info("top10-describe, exclude Negative Pred!\n", stock_pred_top10P_df.describe())
                else:
                    G_LOGGER.info("stock_pred_correctness_df is a empty dataframe.")
                G_LOGGER.info("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        elif data_type == 'train':
            valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                y_pred=y_pred, y_true=y_test, drop_ponit=0.0, test_stock_id=y_testID)

            TP, FP, TN, FN = perf_measure(
                y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label)
            G_LOGGER.info("no drop:")
            G_LOGGER.info("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))

            valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid = drop_small_change_stock_fntest(
                y_pred=y_pred, y_true=y_test, drop_ponit=0.15, test_stock_id=y_testID)

            TP, FP, TN, FN = perf_measure(
                y_pred_valid, y_true_valid, cur_stock_price_test, self.i_month_label)
            G_LOGGER.info("drop 0.15:")
            G_LOGGER.info("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        return

    def data_processing_for_run(self,x_train_,x_test_,y_train_,y_test_,train_ID_df,test_ID_df):
        """
        1.如果indicator_use_type == 'all_with_premonth_label', 获取1月的label作为2月的indicator
        """
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

        """
        2.获取训练集的label、测试集的label、测试集的label对应的股票ID
        """
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

    def graphviz(self, clf, features, class_names, model=[],  fig_path='claasifier_tree.png'):
        clf.classes_ = class_names
        import pybaobabdt
        ax = pybaobabdt.drawTree(clf, model=model, size=10, dpi=72, features=list(features), colormap='Spectral',classes=class_names)
        ax.get_figure().savefig(fig_path, format='png', dpi=300, transparent=True)

    def graphviz_detail(self, clf, features,class_names, fig_path='claasifier_tree.pdf', ):
        import pydotplus
        dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=features,
                                class_names=class_names,
                                filled=True, rounded=True,
                                special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(fig_path)

    def visual(self, training_model, model, features):
        fig_path = self.save_path_permonth + 'graphviz/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        if training_model in ['decisiontreeclassifier']:
            png_path_ = fig_path + 'tree.png'
            clf = model
            self.graphviz(clf, features, fig_path=png_path_)
        elif training_model in ['randomforestclssifier']:
            for idx, clf in enumerate(model.estimators_):
                png_path_ = fig_path + 'tree' + str(idx+1)+'.png'
                self.graphviz(clf, features, model=model, fig_path=png_path_)

        elif training_model in ['gbdtclassifier','adaboostclassifier']:
            top5_estimators_score = heapq.nlargest(5, range(len(model.estimator_weights_)), model.estimator_weights_.__getitem__)

            class_names = model.classes_
            for idx in (top5_estimators_score):
            # for idx, base_model in enumerate(model.estimators_):
                estimators_score = "%.2f" % model.estimator_weights_[idx]
                base_model = model.estimators_[idx]
                if isinstance(base_model, sklearn.tree._classes.DecisionTreeClassifier):
                    clf = base_model
                    png_path_ = fig_path + str(estimators_score)+'_tree' + str(idx+1)+'.png'
                    pdf_path_ = fig_path + str(estimators_score)+'_tree' + str(idx+1)+'.pdf'
                    self.graphviz(clf, features,class_names, model=model, fig_path=png_path_)
                    self.graphviz_detail(clf, features,class_names, fig_path=pdf_path_)
                else:
                    for idy, clf in enumerate(base_model.estimators_):
                        fig_path_ = fig_path + str(estimators_score)+'_random_foreset_'+str(idx+1)+'/'
                        if not os.path.exists(fig_path_):
                            os.makedirs(fig_path_)

                        png_path_ =  fig_path_ + 'tree' + str(idy+1)+'.png'
                        print("png_path_:",png_path_)
                        self.graphviz(clf, features,class_names, model=model, fig_path=png_path_)
                        pdf_path_ =  fig_path_ + 'tree' + str(idy+1)+'.pdf'
                        self.graphviz_detail(clf, features,class_names, fig_path=pdf_path_)

    def sample_class(self,x_train, y_train,y_trainID=None):
        temp = pd.concat([x_train, y_train, y_trainID], axis=1)
        temp = x_train.copy()
        temp['_class_'] = y_train
        temp['_id_'] = y_train

        typicalNDict = {}
        class_num = dict(temp['_class_'].value_counts())
        for key in class_num.keys():
            if class_num[key] >= self.sample_number:
                typicalNDict[key] = self.sample_number
            else:
                typicalNDict[key] = class_num[key]

        def typicalSampling(group, typicalNDict):
            name = group.name
            n = typicalNDict[name]
            tempresult = group.sample(n=n)
            return tempresult

        temp_sample = temp.groupby('_class_', group_keys=False).apply(typicalSampling, typicalNDict)

        y_train_new = temp_sample['_class_']
        y_trainID_new = temp_sample['_id_']
        x_train_new = temp_sample.drop('_class_', axis=1)
        x_train_new = x_train_new.drop('_id_', axis=1)
        return x_train_new, y_train_new, y_trainID_new

    # Linear regression
    def run(self, data_path, train_date_list, test_date_list, n_month_predict):
        """
        1.数据预处理
        """
        G_LOGGER.info("---------------------- data preprocess ----------------------")
        global y_train_regress_label
        global y_test_regress_label
        if args.data_standardscaler:
            x_train_, x_test_, y_train_, y_test_, train_ID_df, test_ID_df, y_train_regress_label, y_test_regress_label = DataPreprocess(
                data_path, train_date_list, test_date_list, n_month_predict,
                self.i_month_label, self.indicator_list,
                label_type).run_with_standardscaler()
        else:
            x_train_, x_test_, y_train_, y_test_, train_ID_df, test_ID_df, y_train_regress_label, y_test_regress_label = DataPreprocess(
                data_path, train_date_list, test_date_list, n_month_predict,
                self.i_month_label, self.indicator_list,
                label_type).run(self.label_norm)

        if label_type == 'regress':
            assert(y_train_regress_label==None)
            y_train_regress_label = y_train_
            y_test_regress_label = y_test_

        x_train, x_test, y_train, y_test, y_trainID, y_testID = self.data_processing_for_run(x_train_,x_test_,y_train_,y_test_,train_ID_df,test_ID_df)

        """
        2.数据采样
        """
        G_LOGGER.info("---------------------- train data sample----------------------")
        G_LOGGER.info(" == before sample == ")
        G_LOGGER.info("-----x_train.shape:{}\n".format(x_train.shape))
        # G_LOGGER.info("-----x_train columns:{}\n".format(x_train.columns.values.tolist()))
        G_LOGGER.info("-----x_test.shape:{}\n".format(x_test.shape))
        # G_LOGGER.info("-----x_test columns:{}\n".format(x_test.columns.values.tolist()))
        G_LOGGER.info("-----y_train.shape:{}\n".format(y_train.shape))
        G_LOGGER.info("-----y_test.shape:{}\n".format(y_test.shape))
        G_LOGGER.info("-----y_train.class:\n{}".format(y_train.value_counts()))
        G_LOGGER.info("-----y_test.class:\n{}".format(y_test.value_counts()))

        # TODO 类别不均衡问题
        x_train, y_train, y_trainID = self.sample_class(x_train,y_train,y_trainID)
        G_LOGGER.info(" == after sample == ")
        G_LOGGER.info("-----x_train.shape:{}\n".format(x_train.shape))
        G_LOGGER.info("-----x_test.shape:{}\n".format(x_test.shape))
        G_LOGGER.info("-----y_train.shape:{}\n".format(y_train.shape))
        G_LOGGER.info("-----y_test.shape:{}\n".format(y_test.shape))
        G_LOGGER.info("-----y_train.class:\n{}".format(y_train.value_counts()))
        G_LOGGER.info("-----y_test.class:\n{}".format(y_test.value_counts()))

        """
        3.模型训练
        """
        G_LOGGER.info("---------------------- model runing ----------------------")
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        y_pred_prob = self.model.predict_proba(x_test).max(axis=-1)
        G_LOGGER.info("model.classes : {}".format(self.model.classes_))
        for i in range(len(self.model.classes_)):
            if "big_positive" == self.model.classes_[i]:
                G_LOGGER.info("big_positive in class-seq-[0,1,2,3,5] number is : {} !!!".format(i))

        """
        3.1 graphviz
            可视化树结构
        """
        # # TODO graphviz
        # graphviz_ = False
        # if graphviz_:
        #     features = []
        #     for feature in list(x_train.columns):
        #         features.append(indicator_map[feature])
        #     G_LOGGER.info("features:",features)
        #     self.visual(args.training_model, self.model, features)

        """
        3.2 indicator_weight
            打印回归模型权重参数
            当indicator_use_type == 'all_with_premonth_label', label为相对前一个月的涨跌幅(而非month0), 同时当月会有前一个月的涨跌幅作为indicator
            indicator_weight为字典,存储每个model的feature_importances_,votingclassifier含有多个model
        """
        indicator_weight = {}
        if indicator_use_type == 'all_with_premonth_label' and i_month_predict > 1:
            self.indicator_list.append('premonth_change_rate')

        if args.training_model in ['linear', 'ridge', 'lasso']:
            indicator_weight[args.training_model] = list(
                zip(self.indicator_list, list(self.model.coef_)))
        elif args.training_model in ['decisiontreeclassifier', 'randomforest', 'randomforestclssifier', 'adaboostregressor', 'adaboostclassifier', 'gbdtclassifier']:
            indicator_weight[args.training_model] = list(
                zip(self.indicator_list, list(self.model.feature_importances_)))
        elif args.training_model in ['votingclassifier']:
            self.votingclassifier_y_pred_prob = {}
            votingclassifierlist = list(self.model.named_estimators_.keys())

            for sub_classifier_name in votingclassifierlist:
                sub_classifier_weight = self.model.named_estimators_[sub_classifier_name].feature_importances_
                indicator_weight[sub_classifier_name] = list(zip(self.indicator_list, list(sub_classifier_weight)))

                # sub_classifier_predict_proba = self.model.named_estimators_[sub_classifier_name].predict_proba(x_test).max(axis=-1)
                sub_classifier_predict_proba = self.model.named_estimators_[sub_classifier_name].predict_proba(x_test)
                self.votingclassifier_y_pred_prob[sub_classifier_name] = sub_classifier_predict_proba
        elif os.path.isfile(args.training_model):
            self.votingclassifier_y_pred_prob = {}
            # os.makedirs(self.save_path_permonth, exist_ok=True)
            # with open(os.path.join(self.save_path_permonth, 'model.pkl'), 'wb') as f:
            #     pickle.dump(self.model, f)
        else:
            assert("indicator_weight error!")

        with open(self.save_path_permonth + 'indicator_weight.log', 'w') as f:
            for key in indicator_weight.keys():
                f.write(str(key)+'\n')
                for i_weight in indicator_weight[key]:
                    f.write(str(i_weight))
                    f.write('\n')
            f.close()

        if indicator_use_type == 'all_with_premonth_label' and i_month_predict > 1:
            self.indicator_list.remove('premonth_change_rate')

        """
        4.结果分析
        """
        G_LOGGER.info("---------------------- result analysis ----------------------")
        """
        4.1 test data analysis
        """
        G_LOGGER.info("\n-----test data analysis for {}:".format(label_type))
        self.result_analysis(y_pred, y_test, y_testID, label_type, cur_stock_price_test=y_test_['0_label'], data_type='test', y_pred_prob=y_pred_prob)
        """
        4.2 train data analysis
        """
        y_pred_fortrain = self.model.predict(x_train)
        y_test_fortrain = y_train
        G_LOGGER.info("\n-----train data analysis for {}:".format(label_type))
        self.result_analysis(y_pred=y_pred_fortrain, y_test=y_test_fortrain, y_testID=y_trainID, label_type=label_type, cur_stock_price_test=None, data_type='train')


if __name__ == '__main__':
    """
        base parameter
    """
    G_LOGGER.info("***********************************************************")
    args = get_args()
    for arg in vars(args):
        G_LOGGER.info(format(arg, '<40')+format(" -----> " + str(getattr(args, arg)), '<'))
    G_LOGGER.info('\n')
    data_path = args.data_path
    save_path_base = args.save_path
    repeat_run = args.repeat_run

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
    sample_number = args.sample_number
    label_norm = args.label_norm
    label_type = args.label_type

    regress_model_params = {}

    """
        regress/class model
    """
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
        regress_model = ensemble.RandomForestClassifier(n_estimators=700, max_depth=10,bootstrap=True, min_samples_leaf=30, min_samples_split=30 )
        label_type = 'class'
    elif args.training_model == 'adaboostregressor':
        base_estimator = ensemble.RandomForestRegressor(n_estimators=10, max_depth=10)
        regress_model = ensemble.AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, learning_rate=0.8,)
        label_type = 'regress'
    elif args.training_model == 'decisiontreeclassifier':
        regress_model = tree.DecisionTreeClassifier(max_depth=5)
        label_type = 'class'
    elif args.training_model == 'adaboostclassifier':
        base_n_estimators = 10
        max_depth = 10
        min_impurity_decrease = 0.002
        n_estimators = 200
        learning_rate = 0.8

        print("base_n_estimators:",base_n_estimators)
        print("max_depth:",max_depth)
        print("n_estimators:",n_estimators)
        print("learning_rate:",learning_rate)
        print("min_impurity_decrease:",min_impurity_decrease)

        base_estimator = ensemble.RandomForestClassifier(n_estimators=200, max_depth=10,bootstrap=True, min_samples_leaf=30, min_samples_split=30,  min_impurity_decrease=min_impurity_decrease )
        # base_estimator = ensemble.RandomForestClassifier(n_estimators=base_n_estimators, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease)
        regress_model = ensemble.AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate)
        label_type = 'class'
    elif args.training_model == 'gbdtclassifier':
        # regress_model = ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.01) ## 学习率设置不宜过大

        n_estimators = 300
        learning_rate = 0.2
        print("n_estimators:",n_estimators)
        print("learning_rate:",learning_rate)
        #regress_model = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate) ## debug
        regress_model = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=60, min_samples_split=60,learning_rate=learning_rate) ## debug
        label_type = 'class'
    elif args.training_model == 'votingclassifier':

        # ADA-RandomForestClassifier
        G_LOGGER.info(" ================================ ")
        G_LOGGER.info(" == ADA RandomForestClassifier == ")
        ada_rf_n_estimators = 200 # change
        ada_rf_max_depth = 10 # change
        ada_rf_bootstrap=True
        ada_rf_min_samples_leaf=50
        ada_rf_min_samples_split=50
        ada_rf_random_state=None
        G_LOGGER.info("ada_rf_n_estimators:{}".format(ada_rf_n_estimators))
        G_LOGGER.info("ada_rf_max_depth:{}".format(ada_rf_max_depth))
        G_LOGGER.info("ada_rf_bootstrap:{}".format(ada_rf_bootstrap))
        G_LOGGER.info("ada_rf_min_samples_leaf:{}".format(ada_rf_min_samples_leaf))
        G_LOGGER.info("ada_rf_min_samples_split:{}".format(ada_rf_min_samples_split))
        G_LOGGER.info("ada_rf_random_state:{}".format(ada_rf_random_state))

        # ADA
        G_LOGGER.info(" == ADA == ")
        ada_n_estimators = 100 # change
        ada_learning_rate = 0.8 # change
        G_LOGGER.info("ada_n_estimators:{}".format(ada_n_estimators))
        G_LOGGER.info("ada_learning_rate:{}".format(ada_learning_rate))

        ada_base_estimator = ensemble.RandomForestClassifier(n_estimators=ada_rf_n_estimators, max_depth=ada_rf_max_depth, bootstrap=ada_rf_bootstrap, min_samples_leaf=ada_rf_min_samples_leaf, min_samples_split=ada_rf_min_samples_split, random_state=ada_rf_random_state)
        adaboost_randomforest = ensemble.AdaBoostClassifier(base_estimator=ada_base_estimator, n_estimators=ada_n_estimators, learning_rate=ada_learning_rate)
        G_LOGGER.info(" ================================ ")

        gbdt_n_estimators = 300 # change
        gbdt_learning_rate = 0.1 # change
        gbdt_max_depth=10
        gbdt_min_samples_leaf=50
        gbdt_min_samples_split=50
        gbdt_random_state=None
        G_LOGGER.info(" == GDBTClassifier == ")
        G_LOGGER.info("gbdt_n_estimators:{}".format(gbdt_n_estimators))
        G_LOGGER.info("gbdt_learning_rate:{}".format(gbdt_learning_rate))
        G_LOGGER.info("gbdt_max_depth:{}".format(gbdt_max_depth))
        G_LOGGER.info("gbdt_min_samples_leaf:{}".format(gbdt_min_samples_leaf))
        G_LOGGER.info("gbdt_min_samples_split:{}".format(gbdt_min_samples_split))
        G_LOGGER.info("gbdt_random_state:{}".format(gbdt_random_state))
        gbdt = ensemble.GradientBoostingClassifier(n_estimators=gbdt_n_estimators, learning_rate=gbdt_learning_rate, max_depth=gbdt_max_depth, min_samples_leaf=gbdt_min_samples_leaf, min_samples_split=gbdt_min_samples_split, random_state=gbdt_random_state)
        G_LOGGER.info(" ================================ ")

        G_LOGGER.info(" == RandomForestClassifier == ")
        rf_n_estimators = 700 # change
        rf_max_depth = 10 # change
        rf_bootstrap=True
        rf_min_samples_leaf=50
        rf_min_samples_split=50
        rf_random_state=None
        randomforest = ensemble.RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, bootstrap=rf_bootstrap, min_samples_leaf=rf_min_samples_leaf, min_samples_split=rf_min_samples_split, random_state=rf_random_state)
        G_LOGGER.info("rf_n_estimators:{}".format(rf_n_estimators))
        G_LOGGER.info("rf_max_depth:{}".format(rf_max_depth))
        G_LOGGER.info("rf_bootstrap:{}".format(rf_bootstrap))
        G_LOGGER.info("rf_min_samples_leaf:{}".format(rf_min_samples_leaf))
        G_LOGGER.info("rf_min_samples_split:{}".format(rf_min_samples_split))
        G_LOGGER.info("rf_random_state:{}".format(rf_random_state))
        G_LOGGER.info(" ================================ ")

        use_estimators = ["adaboost_randomforest","gbdt"]
        regress_model = ensemble.VotingClassifier(estimators=[
            ("adaboost_randomforest", adaboost_randomforest),
            ("gbdt", gbdt),
        ],voting="soft")
        G_LOGGER.info("votingclassifier only use estimators : {}".format(use_estimators))

        # regress_model = ensemble.VotingClassifier(estimators=[
        #     ("randomforest", randomforest),
        #     ("adaboost_randomforest", adaboost_randomforest),
        #     ("gbdt", gbdt),
        # ],voting="soft")
        label_type = 'class'
    elif os.path.isfile(args.training_model):
        import pickle
        data = pickle.load(open(args.training_model, "rb"))
        regress_model        = data['model']
        regress_model_params = data['params']
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
            G_LOGGER.info("-----indicator_list:{}\n".format(indicator_list))
            indicator_list.remove('fund_shareholding_partial')
            G_LOGGER.info("-----remove 'fund_shareholding_partial', indicator_list:{}\n".format(indicator_list))
        else:
            indicator_list = indicator_use_list
    elif indicator_use_type == 'industrial_static':
        indicator_list = ['是', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股']
        feature_id = ['s_hushen', 's_grow', 's_mix', 's_value', 's_small', 's_mid', 's_big']
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

    """
        Run
    """
    for run_time in range(repeat_run):
        G_LOGGER.start("\n\n\nTraining at {} time!\n\n\n".format(run_time))
        save_path = save_path_base + '/' +  "run_" +str(run_time)

        train_month_label = pd.DataFrame()
        test_month_label = pd.DataFrame()
        global i_month_predict
        global total_big_positive
        total_big_positive = pd.DataFrame()

        all_exists = True
        for i in range(n_month_predict):
            i_month_predict = i + 1
            save_path_permonth = save_path + '/' + str(i_month_predict) + '_month/'
            if os.path.exists(os.path.join(save_path_permonth, 'indicator_weight.log')):
                print('file exists:', os.path.join(save_path_permonth, 'indicator_weight.log'))
                continue
            all_exists = False
            break

        if all_exists:
            continue

        for i in range(n_month_predict):
            i_month_predict = i + 1
            i_month_label = str(i_month_predict) + '_' + predict_mode
            G_LOGGER.info("\n\n******************************** {} : {} ********************************".format(i_month_predict, i_month_label))

            save_path_permonth = save_path + '/' + str(i_month_predict) + '_month/'
            if not os.path.exists(save_path_permonth):
                os.makedirs(save_path_permonth)

            with open(save_path_permonth + 'model_params.json', 'w') as f:
                json.dump(regress_model_params, f)

            r = Regress(regress_model, save_path_permonth,
                        i_month_label, indicator_list,
                        drop_small_change_stock_fortrain, drop_small_change_stock_fortest,
                        train_drop_ponit, test_drop_ponit, sample_number, label_norm)
            r.run(data_path, train_date_list, test_date_list, n_month_predict)
