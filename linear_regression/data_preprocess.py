import pandas as pd


class DataPreprocess():
    def __init__(self, data_path, train_date_list, test_date_list, n_month_predict, i_month_label, indicator_list):
        self.data_path = data_path
        self.train_date_list = train_date_list
        self.test_date_list = test_date_list
        self.n_month_predict = n_month_predict
        # '1_predict_changerate_price' or '1_predict_absvalue_price'
        self.i_month_label = i_month_label
        self.indicator_list = indicator_list

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
        return X_df, Y_df, ID_df

    def run(self):
        totaldate_X_df = pd.DataFrame()
        totaldate_Y_df = pd.DataFrame()
        for date in self.train_date_list:
            print("train date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.n_month_predict+1):  # include 0_month
                    label_pickle = self.data_path + 'labels.' + \
                        str(m) + '_month.' + date + '.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(
                    df_indicator, label_pickle_list)
                X_df, Y_df, train_ID_df = self.get_train_indicator_label(
                    df_indicator_label)
                totaldate_X_df = pd.concat([totaldate_X_df, X_df], axis=0)
                totaldate_Y_df = pd.concat([totaldate_Y_df, Y_df], axis=0)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_train, y_train = totaldate_X_df, totaldate_Y_df
        # x_train, x_test, y_train, y_test = train_test_split(totaldate_X_df, totaldate_Y_df, random_state=1)

        for date in self.test_date_list:
            print("test date:", date)
            indicator_pickle = self.data_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
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
        x_test, y_test = X_df, Y_df
        return x_train, x_test, y_train, y_test, test_ID_df