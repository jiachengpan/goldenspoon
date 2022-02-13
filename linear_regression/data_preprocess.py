import pandas as pd
import copy
import os

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

    def run(self):
        totaldate_X_df = pd.DataFrame()
        totaldate_Y_df = pd.DataFrame()
        totaldate_ID_df = pd.DataFrame()
        for date in self.train_date_list:
            print("train date:", date)
            indicator_pickle = os.path.join(self.data_path, f'indicators.{date}.pickle')
            df_indicator = pd.read_pickle(indicator_pickle)
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = os.path.join(self.data_path, f'labels.{m}_month.{date}.pickle')
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                X_df, Y_df, train_ID_df_ = self.get_train_indicator_label(
                    df_indicator_label)
                totaldate_X_df = pd.concat([totaldate_X_df, X_df], axis=0)
                totaldate_Y_df = pd.concat([totaldate_Y_df, Y_df], axis=0)
                totaldate_ID_df = pd.concat([totaldate_ID_df, train_ID_df_], axis=0)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_train, y_train, train_ID_df = totaldate_X_df, totaldate_Y_df, totaldate_ID_df
        # x_train, x_test, y_train, y_test = train_test_split(totaldate_X_df, totaldate_Y_df, random_state=1)
        if len(self.train_date_list) > 1:
            x_train = x_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            train_ID_df = train_ID_df.reset_index(drop=True)


        for date in self.test_date_list:
            print("test date:", date)
            indicator_pickle = os.path.join(self.data_path, f'indicators.{date}.pickle')
            df_indicator = pd.read_pickle(indicator_pickle)
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = os.path.join(self.data_path, f'labels.{m}_month.{date}.pickle')
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

            return x_train, x_test, y_train_classified, y_test_classified, train_ID_df, test_ID_df,  y_train_regress,y_test_regress
        else:
            assert("label type error!")

