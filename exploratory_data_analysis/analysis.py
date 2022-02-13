import os
import sys
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from dataset import DataSet, Args

def compute_feature_importance(dataset : DataSet):
    model = RandomForestRegressor(max_depth=None, random_state=0)

    model.fit(dataset.x_train, dataset.y_train)
    model.score(dataset.x_test, dataset.y_test)

    importance = pd.DataFrame(zip(dataset.x_train.columns.tolist(), model.feature_importances_))
    importance.columns = ['indicator', 'importance']

    return importance.sort_values('importance')

def test_model(model, dataset : DataSet, args : Args):
    model.fit(dataset.x_train, dataset.y_train_cls)

    y_pred_prob = model.predict_proba(dataset.x_test)
    y_pred_cls  = y_pred_prob.argmax(axis=-1)
    y_pred      = np.array([dataset.cls_values[x] for x in y_pred_cls])
    y_pred_prob = y_pred_prob.max(axis=-1)

    for index, cls in enumerate(dataset.cls_values):
        # offset prob with the preference to more-positive classes
        y_pred_prob[y_pred == cls] += float(index)

    prediction = pd.DataFrame(index=dataset.x_test.index)
    prediction['true']      = dataset.y_test
    prediction['true_cls']  = dataset.y_test_cls.values
    prediction['pred']      = y_pred
    prediction['pred_cls']  = y_pred_cls
    prediction['pred_prob'] = y_pred_prob

    assert not prediction.isnull().values.any()

    accuracy = accuracy_score(prediction.true_cls, prediction.pred_cls)
    print(f'acuracy: {accuracy:.4f}')

    stocks_feasible = prediction.index.str.endswith(args.stocks_only)
    stocks_topN     = prediction[stocks_feasible].sort_values('pred', ascending=False)
    stocks_topN     = stocks_topN.sort_values('pred_prob', ascending=False)
    stocks_topN_all = stocks_topN.head(args.stocks_number)
    stocks_topN_pos = stocks_topN[stocks_topN['pred'] > 0].head(args.stocks_number)

    if args.debug:
        print(f'-- topN all --')
        #print(stocks_topN_all)
        print(stocks_topN_all.describe())
        print(f'-- topN positive --')
        #print(stocks_topN_pos)
        print(stocks_topN_pos.describe())
        print(f'-- all pred --')
        #print(stocks_topN.head(50))

    topN_all_pnl = stocks_topN_all['true'].mean()
    topN_pos_pnl = stocks_topN_pos['true'].mean()
    print(f'topN #stocks all: {len(stocks_topN_all):7} pos: {len(stocks_topN_pos):7}')
    print(f'topN pnl:    all: {topN_all_pnl * 100:6.2f}% pos: {topN_pos_pnl * 100:6.2f}%')
