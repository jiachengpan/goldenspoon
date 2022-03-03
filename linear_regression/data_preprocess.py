import pandas as pd
import copy
from sklearn import preprocessing
from utils.logger import G_LOGGER

class DataPreprocess():
    def __init__(self, data_path, train_date_list, test_date_list, n_month_predict, i_month_label, indicator_list, label_type):
        self.data_path = data_path
        self.train_date_list = train_date_list
        self.test_date_list = test_date_list
        self.n_month_predict = n_month_predict
        # '1_predict_changerate_price' or '1_predict_absvalue_price'
        self.i_month_label = i_month_label
        self.indicator_list = indicator_list
        self.label_type = label_type

    def label_classifier_newcol(self,numeric_data):
        numeric_data['class'] = 0
        flag = numeric_data.columns.str.endswith('_predict_changerate_price')
        flag_new = numeric_data.columns.str.endswith('class')
        for i in range(len(flag)):
            if flag[i]:
                label_col = i
            elif flag_new[i]:
                new_col = i
            else:
                assert("flag 'n_predict_changerate_price' not in numeric_data!")
        
        temp_regress_value = numeric_data.iloc[:,label_col].copy()
        numeric_data.insert(label_col + 1, 'regress_value', temp_regress_value, allow_duplicates=False)
        temp_data = numeric_data.iloc[:,label_col].copy()

        for i in range (temp_data.shape[0]):
            if temp_data[i] >= 0.15:
                temp_data[i]='big_positive'
            elif (temp_data[i] < 0.15) and (temp_data[i]>=0.05):
                temp_data[i]='mid_positive'
            elif (temp_data[i] < 0.05) and (temp_data[i]> -0.05):
                temp_data[i]='small'
            elif (temp_data[i] <= -0.05) and (temp_data[i]> -0.15):
                temp_data[i]='mid_negative'
            elif (temp_data[i] <= -0.15):
                temp_data[i]='big_negative'
            # numeric_data.iloc[i,1]=temp_data[i]
            numeric_data.iloc[i,new_col]=temp_data[i]

        classified_data = numeric_data
        return classified_data

    # 1. Data generation
    def merge_df_indicator_label(self, df_indicator, df_label_list):
        df_indicator_label = df_indicator
        for m in range(len(df_label_list)):
            df_label = df_label_list[m]
            df_label = df_label.reset_index()
            if m == 0:
                df_label.columns = ['id', 'date', '0_label']
                df_label = df_label.drop(columns=['date'], axis=1)
            else:
                if '_predict_changerate_price' in self.i_month_label:
                    df_label.columns = ['id', 'date', "predict_absvalue_price", str(
                        m)+'_predict_changerate_price']
                    df_label = df_label.drop(columns=['date'], axis=1)
                    df_label = df_label.drop(
                        columns=['predict_absvalue_price'], axis=1)
                elif '_predict_absvalue_price' in self.i_month_label:
                    df_label.columns = ['id', 'date', str(
                        m)+'_predict_absvalue_price', "predict_changerate_price"]
                    df_label = df_label.drop(columns=['date'], axis=1)
                    df_label = df_label.drop(
                        columns=['predict_changerate_price'], axis=1)
            df_indicator_label = df_indicator_label.merge(
                df_label, on=['id'], how='inner')
        return df_indicator_label

    def merge_indicator_label(self, df_indicator, label_pickle_list):
        df_label_list = []
        for filename in label_pickle_list:
            df_label = pd.read_pickle(filename)
            df_label_list.append(df_label)

        df_indicator_label = self.merge_df_indicator_label(
            df_indicator, df_label_list)
        return df_indicator_label

    def get_train_indicator_label(self, df_indicator_label):
        X_df = df_indicator_label[self.indicator_list]
        self.end_date_stock_price = df_indicator_label.iloc[:,
                                                            df_indicator_label.columns.str.endswith('0_label')]
        Y_df = df_indicator_label.iloc[:, df_indicator_label.columns.str.endswith(
            self.i_month_label)]
        Y_df = pd.concat([self.end_date_stock_price, Y_df], axis=1)
        ID_df = df_indicator_label.iloc[:,
                                        df_indicator_label.columns.str.endswith('id')]
        Y_df = pd.concat([ID_df, Y_df], axis=1)
        X_df = pd.concat([ID_df, X_df], axis=1)
        return X_df, Y_df, ID_df

    def label_classifier(self,numeric_data):
        flag = numeric_data.columns.str.endswith('_predict_changerate_price')
        for i in range(len(flag)):
            if flag[i]:
                label_col = i
            else:
                assert("flag 'n_predict_changerate_price' not in numeric_data!")
        
        temp_regress_value = numeric_data.iloc[:,label_col].copy()
        numeric_data.insert(label_col + 1, 'regress_value', temp_regress_value, allow_duplicates=False)
        temp_data = numeric_data.iloc[:,label_col].copy()

        for i in range (temp_data.shape[0]):
            if temp_data[i] >= 0.15:
                temp_data[i]='big_positive'
            elif (temp_data[i] < 0.15) and (temp_data[i]>=0.05):
                temp_data[i]='mid_positive'
            elif (temp_data[i] < 0.05) and (temp_data[i]> -0.05):
                temp_data[i]='small'
            elif (temp_data[i] <= -0.05) and (temp_data[i]> -0.15):
                temp_data[i]='mid_negative'
            elif (temp_data[i] <= -0.15):
                temp_data[i]='big_negative'
            # numeric_data.iloc[i,1]=temp_data[i]
            numeric_data.iloc[i,label_col]=temp_data[i]

        classified_data = numeric_data
        return classified_data

    def filter(self, database):
        k_filter_columns = database.columns.values.tolist()
        for col in k_filter_columns:
            # print("k_filter_columns:", k_filter_columns)
            # print("database[col]:", database[col])
            if col not in ['id', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股','fund_shareholding_partial']:
                database = database[database[col] <= 10]

        database = database.fillna(0.0)
        return database

    def filter_nan(self, database):
        database = database.fillna(0.0)
        return database

    def fund_standardscaler(self, database, transform_list=[], mode='train', features=['fund_shareholding_mean','fund_shareholding_std','fund_number_mean','fund_number_std',\
                            'share_ratio_of_funds_mean','share_ratio_of_funds_std','num_of_funds_mean','num_of_funds_std'],\
                            ):
        if mode=='train':
            transform_list = []
            for i in range(len(features)):
                feature = database[[features[i]]]
                standar_scaler = preprocessing.StandardScaler()
                scaled_feature = self.standardscaler(standar_scaler, feature)
                # print("-standar_scaler.mean_:",standar_scaler.mean_)
                # print("-standar_scaler.var_:",standar_scaler.var_)
                database[features[i]] = scaled_feature
                transform_list.append(standar_scaler)
            return database,transform_list
        else:
            for i in range(len(features)):
                feature = database[[features[i]]]
                standar_scaler = transform_list[i]
                # print("test-standar_scaler.mean_:",standar_scaler.mean_)
                # print("test-standar_scaler.var_:",standar_scaler.var_)
                
                scaled_feature = self.standardscaler(standar_scaler, feature, mode='test')
                database[features[i]] = scaled_feature

            return database,[]


    def standardscaler(self, standar_scaler, feature, mode='train'):
        if mode=='train':
            feature_scaled = standar_scaler.fit_transform(feature)
        else:
            feature_scaled = standar_scaler.transform(feature)
        return feature_scaled


    def run_with_standardscaler(self):
        assert(0)

        # "2020-09-30 2020-12-31 2021-03-31 2021-06-30"
        train_date_list_3_9 = []
        train_date_list_6_12 = []
        for date in self.train_date_list:
            if (not date.find("09-30") == -1) or (not date.find("03-31") == -1):
                train_date_list_3_9.append(date)
            elif (not date.find("12-31") == -1) or (not date.find("06-30") == -1):
                train_date_list_6_12.append(date)

        if len(self.test_date_list) >1:
            assert(0)
        for date in self.test_date_list:
            if (not date.find("09-30") == -1) or (not date.find("03-31") == -1):
                test_date_flag = '3_9'
            elif (not date.find("12-31") == -1) or (not date.find("06-30") == -1):
                test_date_flag = '6_12'


        ## 处理3 9月份的数据
        totaldate_X_df_3_9 = pd.DataFrame()
        totaldate_Y_df_3_9  = pd.DataFrame()
        totaldate_ID_df_3_9 = pd.DataFrame()
        G_LOGGER.info("train_date_list_3_9:",train_date_list_3_9)
        for date in train_date_list_3_9:
            G_LOGGER.info("3 or 9 train date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            df_indicator = self.filter_nan(df_indicator).copy()
            # df_indicator = self.filter(df_indicator).copy()


            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                # print("!!!!!len df_indicator_label after filter:",len(df_indicator_label))
                # print("df_indicator_label:",df_indicator_label.describe())
                # assert(0)
                X_df, Y_df, train_ID_df_ = self.get_train_indicator_label(
                    df_indicator_label)
                totaldate_X_df_3_9 = pd.concat([totaldate_X_df_3_9, X_df], axis=0)
                totaldate_Y_df_3_9 = pd.concat([totaldate_Y_df_3_9, Y_df], axis=0)
                totaldate_ID_df_3_9 = pd.concat([totaldate_ID_df_3_9, train_ID_df_], axis=0)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_train_3_9, y_train_3_9, train_ID_df_3_9 = totaldate_X_df_3_9, totaldate_Y_df_3_9, totaldate_ID_df_3_9
        G_LOGGER.verbose("-x_train_3_9:",x_train_3_9['fund_shareholding_mean'].describe())
        sacled_x_train_3_9, transform_list_3_9 = self.fund_standardscaler(x_train_3_9)
        G_LOGGER.verbose("-sacled_x_train_3_9:",sacled_x_train_3_9['fund_shareholding_mean'].describe())



        ## 处理6 12月份的数据
        totaldate_X_df_6_12 = pd.DataFrame()
        totaldate_Y_df_6_12  = pd.DataFrame()
        totaldate_ID_df_6_12 = pd.DataFrame()
        print("train_date_list_6_12:",train_date_list_6_12)
        for date in train_date_list_6_12:
            print("3 or 9 train date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            df_indicator = self.filter_nan(df_indicator).copy()


            # df_indicator = self.filter(df_indicator).copy()


            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                # print("!!!!!len df_indicator_label after filter:",len(df_indicator_label))
                # print("df_indicator_label:",df_indicator_label.describe())
                # assert(0)
                X_df, Y_df, train_ID_df_ = self.get_train_indicator_label(
                    df_indicator_label)
                totaldate_X_df_6_12 = pd.concat([totaldate_X_df_6_12, X_df], axis=0)
                totaldate_Y_df_6_12 = pd.concat([totaldate_Y_df_6_12, Y_df], axis=0)
                totaldate_ID_df_6_12 = pd.concat([totaldate_ID_df_6_12, train_ID_df_], axis=0)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_train_6_12, y_train_6_12, train_ID_df_6_12 = totaldate_X_df_6_12, totaldate_Y_df_6_12, totaldate_ID_df_6_12
        G_LOGGER.VERBOSE("-x_train_6_12:",x_train_6_12['fund_shareholding_mean'].describe())
        sacled_x_train_6_12, transform_list_6_12 = self.fund_standardscaler(x_train_6_12)
        G_LOGGER.VERBOSE("-sacled_x_train_6_12:",sacled_x_train_6_12['fund_shareholding_mean'].describe())


        x_train = pd.concat([sacled_x_train_3_9, sacled_x_train_6_12], axis=0)
        y_train = pd.concat([y_train_3_9, y_train_6_12], axis=0)
        train_ID_df = pd.concat([train_ID_df_3_9, train_ID_df_6_12], axis=0)

        if len(self.train_date_list) > 1:
            x_train = x_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            train_ID_df = train_ID_df.reset_index(drop=True)


        for date in self.test_date_list:
            print("test date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            print("!!!!!len df_indicator:",len(df_indicator))
            # print("df_indicator:",df_indicator.describe())
            df_indicator = self.filter(df_indicator).copy()
            print("!!!!!len df_indicator after filter:",len(df_indicator))
            # print("df_indicator:",df_indicator.describe())
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                # print("!!!!!len df_indicator_label after filter:",len(df_indicator_label))
                X_df, Y_df, test_ID_df = self.get_train_indicator_label(
                    df_indicator_label)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_test, y_test, test_ID_df = X_df, Y_df, test_ID_df
        G_LOGGER.VERBOSE("-x_test:",x_test.describe())
        if test_date_flag == '3_9':
            test_transform_list = transform_list_3_9
        elif test_date_flag == '6_12':
            test_transform_list = transform_list_6_12
        sacled_x_test, _ = self.fund_standardscaler(x_test,transform_list=test_transform_list,mode='test')
        G_LOGGER.VERBOSE("-sacled_x_test:",sacled_x_test.describe())
        x_test = sacled_x_test


        if self.label_type == 'regress':
            return x_train, x_test, y_train, y_test, train_ID_df, test_ID_df,  None, None
        elif self.label_type == 'class':
            y_train_regress = y_train.copy() 
            y_test_regress = y_test.copy() 
            y_train_classified = self.label_classifier(y_train)
            y_test_classified = self.label_classifier(y_test)
            return x_train, x_test, y_train_classified, y_test_classified, train_ID_df, test_ID_df,  y_train_regress, y_test_regress
        else:
            assert("label type error!")



    def run(self):
        totaldate_X_df = pd.DataFrame()
        totaldate_Y_df = pd.DataFrame()
        totaldate_ID_df = pd.DataFrame()
        for date in self.train_date_list:
            G_LOGGER.info("train date:{}".format(date))
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)

            G_LOGGER.info("---- len df_indicator:{} ----".format(len(df_indicator)))
            df_indicator = self.filter(df_indicator).copy()
            G_LOGGER.info("---- len df_indicator after filter:{} ----".format(len(df_indicator)))


            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1): # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)

                X_df, Y_df, train_ID_df_ = self.get_train_indicator_label(
                    df_indicator_label)
                totaldate_X_df = pd.concat([totaldate_X_df, X_df], axis=0)
                totaldate_Y_df = pd.concat([totaldate_Y_df, Y_df], axis=0)
                totaldate_ID_df = pd.concat([totaldate_ID_df, train_ID_df_], axis=0)
            else:
                G_LOGGER.info("{} is a empty dataframe!".format(indicator_pickle))
        x_train, y_train, train_ID_df = totaldate_X_df, totaldate_Y_df, totaldate_ID_df
        # x_train, x_test, y_train, y_test = train_test_split(totaldate_X_df, totaldate_Y_df, random_state=1)
        if len(self.train_date_list) > 1:
            x_train = x_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            train_ID_df = train_ID_df.reset_index(drop=True)


        for date in self.test_date_list:
            G_LOGGER.info("test date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            df_indicator = self.filter(df_indicator).copy()
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                X_df, Y_df, test_ID_df = self.get_train_indicator_label(
                    df_indicator_label)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_test, y_test, test_ID_df = X_df, Y_df, test_ID_df


        if self.label_type == 'regress':
            return x_train, x_test, y_train, y_test, train_ID_df, test_ID_df,  None, None
        elif self.label_type == 'class':
            y_train_regress = y_train.copy() 
            y_test_regress = y_test.copy() 
            y_train_classified = self.label_classifier(y_train)
            y_test_classified = self.label_classifier(y_test)
            return x_train, x_test, y_train_classified, y_test_classified, train_ID_df, test_ID_df,  y_train_regress, y_test_regress
        else:
            assert("label type error!")