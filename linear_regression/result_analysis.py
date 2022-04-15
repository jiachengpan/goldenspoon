import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heapq

# 2. result visualization


def draw(y_pred, y_test, cur_stock_price_test, R2, i_month_label, save_path_permonth, show_interval=None):
    if show_interval == None:
        show_interval = len(y_pred)
    start_ = 0
    end_ = show_interval
    while start_ < len(y_pred):
        if end_ > len(y_pred):
            end_ = len(y_pred)
        plt.figure()
        show_len = end_ - start_
        plt.plot(np.arange(show_len),
                 y_test[start_:end_], 'bo-', label='true value')
        plt.plot(np.arange(show_len),
                 y_pred[start_:end_], 'ro-', label='predict value')
        if '_predict_absvalue_price' in i_month_label:
            plt.plot(np.arange(
                show_len), cur_stock_price_test[start_:end_], 'g^-', label='current stock price')
        elif '_predict_changerate_price' in i_month_label:
            plt.axhline(y=0, color='g', linestyle='-')
        plt.title('R2: %f' % R2)
        plt.legend()
        flag = 'demo_index_'+str(start_)+'-'+str(end_)+'.png'
        plt.savefig(save_path_permonth + flag)
        start_ += show_interval
        end_ += show_interval

    # draw the y_pred and y_test scatter diagram
    plt.figure()
    x = np.linspace(0, 1.0, len(y_pred))
    plt.scatter(x, y_pred, c='r', marker='*')
    plt.scatter(x, y_test, c='', marker='o', edgecolors='g')
    plt.savefig(save_path_permonth + 'y_scatter.png')



def draw_confusion_matrix(TP, FP, TN, FN, save_path_permonth):
    # [[TN|FP]
    # [FN|TP]]
    confusion_flag = np.array([['TN', 'FP'], ['FN', 'TP']])
    confusion_matrix = np.array([[TN, FP], [FN, TP]])
    plt.matshow(confusion_matrix, cmap=plt.cm.Greens)
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.annotate(confusion_flag[i, j] + ' : ' + str(confusion_matrix[i, j]), xy=(
                i, j), horizontalalignment='center', verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    flag = 'demo_confusion_matrix.png'
    plt.savefig(save_path_permonth + flag)
    return

# 3. confusion_matrix
def perf_measure(
        y_pred,
        y_true,
        cur_stock_price_test,
        i_month_label,
        y_true_regress_value=None,
        stock_id=None,
        y_pred_prob=None,
        votingclassifier_y_pred_prob=None,
        stats = {},
        ):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    cur_stock_price_test = np.array(cur_stock_price_test).flatten()

    TP, FP, TN, FN = 0, 0, 0, 0

    T_small = 0
    T_mid_positive = 0
    T_big_positive = 0
    T_mid_negative = 0
    T_big_negative = 0

    F_small = 0
    F_mid_positive = 0
    F_big_positive = 0
    F_mid_negative = 0
    F_big_negative = 0
    F_small_exclude_small = 0
    F_mid_positive_exclude_small = 0
    F_big_positive_exclude_small = 0
    F_mid_negative_exclude_small = 0
    F_big_negative_exclude_small = 0

    T_positive = 0
    T_negative = 0
    F_positive = 0
    F_negative = 0
    F_positive_exclude_small = 0
    F_negative_exclude_small = 0

    big_positive_y_pred = []
    big_positive_y_pred_prob = []
    big_positive_votingclassifier_y_pred_prob = []
    big_positive_y_true = []

    big_positive_y_true_regress = []
    big_positive_y_true_id = []
    mid_positive_y_pred = []
    mid_positive_y_true = []
    mid_positive_y_true_regress = []
    mid_positive_y_true_id = []
    mid_positive_y_pred_prob = []
    mid_positive_votingclassifier_y_pred_prob = []

    big_negative_y_true = []
    big_negative_y_true_regress = []
    big_negative_y_true_id = []

    big_negative_y_pred_prob = []

    positive_y_pred = []
    positive_y_true = []
    positive_y_true_regress = []
    positive_y_true_id = []
    for i in range(len(y_true)):
        if '_predict_absvalue_price' in i_month_label:
            assert(0)
            y_true_case = y_true[i]-cur_stock_price_test[i]
            y_pred_case = y_pred[i]-cur_stock_price_test[i]
        elif '_predict_changerate_price' in i_month_label:
            # assert(0)
            y_true_case = y_true[i]
            y_pred_case = y_pred[i]

        if isinstance(y_true_case,str):
            if y_pred_case == 'small':
                if (y_true_case) == (y_pred_case) :
                    T_small += 1
                else:
                    F_small +=1
            elif y_pred_case == 'mid_positive':
                if y_true_regress_value is not None:
                    mid_positive_y_true.append(y_true_case)
                    mid_positive_y_true_regress.append(y_true_regress_value[i])
                    mid_positive_y_true_id.append(stock_id[i][0])
                    mid_positive_y_pred_prob.append(y_pred_prob[i])

                    if votingclassifier_y_pred_prob != None:
                        _y_pred_prob_list = {}
                        for key in votingclassifier_y_pred_prob.keys():
                            _y_pred_prob_list[key] = votingclassifier_y_pred_prob[key][i]
                        mid_positive_votingclassifier_y_pred_prob.append(_y_pred_prob_list)

                if (y_true_case) == (y_pred_case) :
                    T_mid_positive += 1
                else:
                    F_mid_positive +=1
                    if y_true_case not in ['small','big_positive']:
                        F_mid_positive_exclude_small += 1
            elif y_pred_case == 'big_positive':
                if y_true_regress_value is not None:
                    big_positive_y_true.append(y_true_case)
                    big_positive_y_true_regress.append(y_true_regress_value[i])
                    big_positive_y_true_id.append(stock_id[i][0])
                    big_positive_y_pred_prob.append(y_pred_prob[i])

                    if votingclassifier_y_pred_prob != None:
                        _y_pred_prob_list = {}
                        for key in votingclassifier_y_pred_prob.keys():
                            _y_pred_prob_list[key] = votingclassifier_y_pred_prob[key][i]
                        big_positive_votingclassifier_y_pred_prob.append(_y_pred_prob_list)

                if (y_true_case) == (y_pred_case) :
                    T_big_positive += 1
                else:
                    F_big_positive +=1
                    if y_true_case not in ['small','mid_positive']:
                        F_big_positive_exclude_small += 1
            elif y_pred_case == 'mid_negative':
                if (y_true_case) == (y_pred_case) :
                    T_mid_negative += 1
                else:
                    F_mid_negative +=1
                    if y_true_case not in ['small','big_negative']:
                        F_mid_negative_exclude_small += 1
            elif y_pred_case == 'big_negative':
                if y_true_regress_value is not None:
                    big_negative_y_true.append(y_true_case)
                    big_negative_y_true_regress.append(y_true_regress_value[i])
                    big_negative_y_true_id.append(stock_id[i][0])
                    big_negative_y_pred_prob.append(y_pred_prob[i])
                if (y_true_case) == (y_pred_case) :
                    T_big_negative += 1
                else:
                    F_big_negative +=1
                    if y_true_case not in ['small','mid_negative']:
                        F_big_negative_exclude_small += 1

            if 'positive' in str(y_pred_case):
                if y_true_regress_value is not None:
                    positive_y_pred.append(y_pred_case)
                    positive_y_true.append(y_true_case)
                    positive_y_true_regress.append(y_true_regress_value[i])
                    positive_y_true_id.append(stock_id[i][0])
                if 'positive' in str(y_true_case):
                    T_positive += 1
                else:
                    F_positive += 1
                    if y_true_case != 'small':
                        F_positive_exclude_small += 1

            elif 'negative' in str(y_pred_case):
                if 'negative' in str(y_true_case):
                    T_negative += 1
                else:
                    F_negative += 1
                    if y_true_case != 'small':
                        F_negative_exclude_small += 1

        elif isinstance(y_true_case,float):
            # share prices are rising in y_true, also y_pred
            if (y_true_case) >= 0 and (y_pred_case) >= 0:
                TP += 1
            # share prices are rising in y_pred, but falling in y_true
            if (y_true_case) < 0 and (y_pred_case) >= 0:
                FP += 1
            # share prices are falling in y_true, also y_pred
            if (y_true_case) < 0 and (y_pred_case) < 0:
                TN += 1
            # share prices are rising in y_true, but falling in y_pred
            if (y_true_case) >= 0 and (y_pred_case) < 0:
                FN += 1

    def acc(T,F):
        try:
            acc = (T)/(T+F)
        except:
            acc = 0
        return acc

    if isinstance(y_true_case,str):
        acc_small = acc(T_small,F_small)
        print("acc_small:{}, T_num:{}, F_num:{}".format(acc_small,T_small,F_small))

        acc_mid_positive = acc(T_mid_positive,F_mid_positive)
        acc_mid_positive_exclude_small = acc(T_mid_positive,F_mid_positive_exclude_small)
        print("acc_mid_positive:{}, acc_mid_positive_exclude_small_big_positive:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_mid_positive, acc_mid_positive_exclude_small, T_mid_positive, F_mid_positive, F_mid_positive_exclude_small))

        acc_big_positive = acc(T_big_positive,F_big_positive)
        acc_big_positive_exclude_small = acc(T_big_positive,F_big_positive_exclude_small)
        print("acc_big_positive:{}, acc_big_positive_exclude_small_mid_positive:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_big_positive, acc_big_positive_exclude_small, T_big_positive, F_big_positive, F_big_positive_exclude_small))

        acc_mid_negative = acc(T_mid_negative,F_mid_negative)
        acc_mid_negative_exclude_small = acc(T_mid_negative,F_mid_negative_exclude_small)
        print("acc_mid_negative:{}, acc_mid_negative_exclude_small_big_negative:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_mid_negative, acc_mid_negative_exclude_small, T_mid_negative, F_mid_negative, F_mid_negative_exclude_small))

        acc_big_negative = acc(T_big_negative,F_big_negative)
        acc_big_negative_exclude_small = acc(T_big_negative,F_big_negative_exclude_small)
        print("acc_big_negative:{}, acc_big_negative_exclude_small_mid_negative:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_big_negative, acc_big_negative_exclude_small, T_big_negative, F_big_negative, F_big_negative_exclude_small))

        acc_positive = acc(T_positive,F_positive)
        acc_positive_exclude_small = acc(T_positive,F_positive_exclude_small)
        print("acc_positive:{}, acc_positive_exclude_small:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_positive, acc_positive_exclude_small, T_positive, F_positive, F_positive_exclude_small))

        acc_negative = acc(T_negative,F_negative)
        acc_negative_exclude_small = acc(T_negative,F_negative_exclude_small)
        print("acc_negative:{}, acc_negative_exclude_small:{}, T_num:{}, F_num:{}, F_exclude_num:{}"\
            .format(acc_negative, acc_negative_exclude_small, T_negative, F_negative, F_negative_exclude_small))

        stats['accuracy'] = {
            'acc_small': {'acc': acc_small, 'T_num': T_small, 'F_num': F_small},
            'acc_mid': {'acc': acc_mid_positive, 'T_num': T_mid_positive, 'F_num': F_mid_positive,
                        'exclude_small': acc_mid_positive_exclude_small, 'F_exclude_num': F_mid_positive_exclude_small},
            'acc_big': {'acc': acc_big_positive, 'T_num': T_big_positive, 'F_num': F_big_positive,
                        'exclude_small': acc_big_positive_exclude_small, 'F_exclude_num': F_big_positive_exclude_small},
            'acc_mid_neg': {'acc': acc_mid_negative, 'T_num': T_mid_negative, 'F_num': F_mid_negative,
                            'exclude_small': acc_mid_negative_exclude_small, 'F_exclude_num': F_mid_negative_exclude_small},
            'acc_big_neg': {'acc': acc_big_negative, 'T_num': T_big_negative, 'F_num': F_big_negative,
                            'exclude_small': acc_big_negative_exclude_small, 'F_exclude_num': F_big_negative_exclude_small},
            'acc_pos': {'acc': acc_positive, 'T_num': T_positive, 'F_num': F_positive,
                        'exclude_small': acc_positive_exclude_small, 'F_exclude_num': F_positive_exclude_small},
            'acc_neg': {'acc': acc_negative, 'T_num': T_negative, 'F_num': F_negative,
                        'exclude_small': acc_negative_exclude_small, 'F_exclude_num': F_negative_exclude_small},
        }

        df_big_positive = pd.DataFrame()
        if y_true_regress_value is not None:

            stats['profit'] = {}

            print("--raw--")
            ########################################
            tops10 = heapq.nlargest(10, range(len(big_positive_y_pred_prob)), big_positive_y_pred_prob.__getitem__)
            tops10_profit = []
            tops20 = heapq.nlargest(20, range(len(big_positive_y_pred_prob)), big_positive_y_pred_prob.__getitem__)
            tops20_profit = []
            tops20_details = []
            for i in tops20:
                if votingclassifier_y_pred_prob != None:
                    print("id: {}, pred: big_positive, true_c: {}, true_r: {}, pred_prob: {}, votingclassifier_pred_prob: {}".format(
                        big_positive_y_true_id[i],
                        big_positive_y_true[i],
                        big_positive_y_true_regress[i],
                        big_positive_y_pred_prob[i],
                        big_positive_votingclassifier_y_pred_prob[i]))
                else:
                    print("id: {}, pred: big_positive, true_c: {}, true_r: {}, pred_prob: {}".format(
                        big_positive_y_true_id[i],
                        big_positive_y_true[i],
                        big_positive_y_true_regress[i],
                        big_positive_y_pred_prob[i]))
                tops20_profit.append(big_positive_y_true_regress[i])
                tops20_details.append({
                    'id': big_positive_y_true_id[i],
                    'true_c': big_positive_y_true[i],
                    'true_r': big_positive_y_true_regress[i],
                    'pred_prob': big_positive_y_pred_prob[i],
                    'pred_prob_volting': big_positive_votingclassifier_y_pred_prob[i],
                })
                if i in tops10:
                    tops10_profit.append(big_positive_y_true_regress[i])
            print("Buy all big_positive, profit: {}".format(np.array(big_positive_y_true_regress).mean()))
            print("Buy top10 big_positive, profit: {}".format(np.array(tops10_profit).mean()))
            print("Buy top20 big_positive, profit: {}".format(np.array(tops20_profit).mean()))

            stats['profit']['big_positive'] = {
                'profit': np.array(big_positive_y_true_regress).mean(),
                'tops10_profit': np.array(tops10_profit).mean(),
                'tops20_profit': np.array(tops20_profit).mean(),
                'tops20_details': tops20_details,
                }

            ########################################
            tops10 = heapq.nlargest(10, range(len(mid_positive_y_pred_prob)), mid_positive_y_pred_prob.__getitem__)
            tops10_profit = []
            tops20 = heapq.nlargest(20, range(len(mid_positive_y_pred_prob)), mid_positive_y_pred_prob.__getitem__)
            tops20_profit = []
            tops20_details = []
            for i in tops20:
                #print("id: {}, pred: mid_positive, true_c: {}, true_r: {}, pred_prob: {}".format(mid_positive_y_true_id[i], mid_positive_y_true[i],mid_positive_y_true_regress[i],mid_positive_y_pred_prob[i]))
                tops20_profit.append(mid_positive_y_true_regress[i])
                tops20_details.append({
                    'id': mid_positive_y_true_id[i],
                    'true_c': mid_positive_y_true[i],
                    'true_r': mid_positive_y_true_regress[i],
                    'pred_prob': mid_positive_y_pred_prob[i],
                    'pred_prob_volting': mid_positive_votingclassifier_y_pred_prob[i],
                })

                if i in tops10:
                    tops10_profit.append(mid_positive_y_true_regress[i])
            print("Buy all mid_positive, profit: {}".format(np.array(mid_positive_y_true_regress).mean()))
            print("Buy top10 mid_positive, profit: {}".format(np.array(tops10_profit).mean()))
            print("Buy top20 mid_positive, profit: {}".format(np.array(tops20_profit).mean()))

            print('YES!')
            stats['profit']['mid_positive'] = {
                'profit': np.array(mid_positive_y_true_regress).mean(),
                'tops10_profit': np.array(tops10_profit).mean(),
                'tops20_profit': np.array(tops20_profit).mean(),
                'tops20_details': tops20_details,
                }

            ########################################
            # tops10 = heapq.nlargest(10, range(len(big_negative_y_pred_prob)), big_negative_y_pred_prob.__getitem__)
            # tops10_profit = []
            # tops20 = heapq.nlargest(20, range(len(big_negative_y_pred_prob)), big_negative_y_pred_prob.__getitem__)
            # tops20_profit = []
            # for i in tops20:
            #     print("id: {}, pred: big_negative, true_c: {}, true_r: {}, pred_prob: {}".format(big_negative_y_true_id[i], big_negative_y_true[i],big_negative_y_true_regress[i],big_negative_y_pred_prob[i]))
            #     tops20_profit.append(big_negative_y_true_regress[i])
            #     if i in tops10:
            #         tops10_profit.append(big_negative_y_true_regress[i])
            # print("Buy all big_negative, profit: {}".format(np.array(big_negative_y_true_regress).mean()))
            # print("Buy top10 big_negative, profit: {}".format(np.array(tops10_profit).mean()))
            # print("Buy top20 big_negative, profit: {}".format(np.array(tops20_profit).mean()))
            ########################################
            df_big_positive['id'] = big_positive_y_true_id
            df_big_positive['positive_y_true_regress'] = big_positive_y_true_regress
        return df_big_positive

    return TP, FP, TN, FN

# 4. drop stocks with small changes

def drop_small_change_stock_fntrain(y_train, x_train, drop_ponit, train_stock_id):
    valid_stock_list = np.where(np.absolute(y_train) > drop_ponit)
    valid_stock_list = np.asarray(valid_stock_list)
    valid_stock_list = valid_stock_list.transpose()
    valid_stock_list = np.array(valid_stock_list).flatten()
    y_train_valid = y_train.loc[valid_stock_list]
    x_train_valid = x_train.loc[valid_stock_list]
    train_stock_id_valid = train_stock_id.loc[valid_stock_list]
    return valid_stock_list, y_train_valid, x_train_valid, train_stock_id_valid


def drop_small_change_stock_fntest(y_pred, y_true, drop_ponit, test_stock_id):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    test_stock_id = np.array(test_stock_id).flatten()
    valid_stock_list = np.where(np.absolute(y_pred) > drop_ponit)
    valid_stock_list = np.asarray(valid_stock_list)
    valid_stock_list = valid_stock_list.transpose()
    y_pred_valid = y_pred[valid_stock_list]
    y_true_valid = y_true[valid_stock_list]
    test_stock_id_valid = test_stock_id[valid_stock_list]
    return valid_stock_list, y_pred_valid, y_true_valid, test_stock_id_valid

# 5. assess the prediction correcness of each stock


def perf_measure_per_stock(full_stock_list, valid_stock_list, y_pred, y_true, y_ID_valid):
    stock_pred_correctness = []
    valid_stock_list_len = valid_stock_list.shape[0]

    # print((valid_stock_list.shape))
    for i in range(valid_stock_list_len):
        n = valid_stock_list[i]
        id = y_ID_valid[i]
        y_pred_case = y_pred[i]
        y_true_case = y_true[i]

        stock_pred_correctness.append(dict(
            index = n[0],
            id    = id[0],
            pred  = y_pred[i][0],
            true  = y_true[i][0]))

        # share prices are rising in y_true, also y_pred
        if (y_true_case) >= 0 and (y_pred_case) >= 0:
            stock_pred_correctness[-1]['type'] = 'TP'
            # print('stock index', n, 'stock id', id,
            #       'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are rising in y_pred, but falling in y_true
        if (y_true_case) < 0 and (y_pred_case) >= 0:
            stock_pred_correctness[-1]['type'] = 'FP'
            # print('stock index', n, 'stock id', id,
            #     'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are falling in y_true, also y_pred
        if (y_true_case) < 0 and (y_pred_case) < 0:
            stock_pred_correctness[-1]['type'] = 'TN'
            # print('stock index', n, 'stock id', id,
            #     'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are rising in y_true, but falling in y_pred
        if (y_true_case) >= 0 and (y_pred_case) < 0:
            stock_pred_correctness[-1]['type'] = 'FN'
            # print('stock index', n, 'stock id', id,
            #     'prediction', y_pred[i], 'truevalue', y_true[i])
    return stock_pred_correctness
