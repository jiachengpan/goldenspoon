import numpy as np
import matplotlib.pyplot as plt

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

# 3. confusion_matrix


def perf_measure(y_pred, y_true, cur_stock_price_test, i_month_label, save_path_permonth):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    cur_stock_price_test = np.array(cur_stock_price_test).flatten()

    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if '_predict_absvalue_price' in i_month_label:
            assert(0)
            y_true_case = y_true[i]-cur_stock_price_test[i]
            y_pred_case = y_pred[i]-cur_stock_price_test[i]
        elif '_predict_changerate_price' in i_month_label:
            # assert(0)
            y_true_case = y_true[i]
            y_pred_case = y_pred[i]
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
            print('stock index', n, 'stock id', id,
                  'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are rising in y_pred, but falling in y_true
        if (y_true_case) < 0 and (y_pred_case) >= 0:
            stock_pred_correctness[-1]['type'] = 'FP'
            print('stock index', n, 'stock id', id,
                'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are falling in y_true, also y_pred
        if (y_true_case) < 0 and (y_pred_case) < 0:
            stock_pred_correctness[-1]['type'] = 'TN'
            print('stock index', n, 'stock id', id,
                'prediction', y_pred[i], 'truevalue', y_true[i])
        # share prices are rising in y_true, but falling in y_pred
        if (y_true_case) >= 0 and (y_pred_case) < 0:
            stock_pred_correctness[-1]['type'] = 'FN'
            print('stock index', n, 'stock id', id,
                'prediction', y_pred[i], 'truevalue', y_true[i])
    return stock_pred_correctness
