#!/usr/bin/env python
# coding: utf-8



The difference between train-rmse and eval-rmse was not big. It maybe avoid overfitting. But, of course, I
Just used small variables, the value was not big.

1.  cols = ['full_sq', 'life_sq','tan_sq_0.2',  'floor', 'max_floor', 'floor_label_0.1', 'sub_area_sample'] 
train-rmse:0.452406	eval-rmse:0.509475 

2. cols = ['full_sq', 'life_sq','tan_sq_0.2', 'floor', 'max_floor', 'floor_label_0.1', 'sub_area']
train-rmse:0.391513	eval-rmse:0.481048

3. cols = ['full_sq', 'life_sq','tan_sq', 'floor', 'max_floor', 'floor_label_0.1', 'sub_area']
train-rmse:0.379885	eval-rmse:0.481801

4. cols = ['full_sq', 'life_sq','tan_sq', 'floor', 'max_floor', 'tan_floor', 'sub_area']
train-rmse:0.386476	eval-rmse:0.482583




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.




house_train = pd.read_csv('../input/train.csv', parse_dates = ['timestamp'])
house_test = pd.read_csv('../input/test.csv', parse_dates = ['timestamp'])
num = house_train.shape[0]
target = np.log1p(house_train.price_doc)

house_train.drop(['id', 'price_doc'], axis =1, inplace = True)
house_test.drop(['id'], axis = 1, inplace = True)

x_full = pd.concat([house_train, house_test], axis = 0)
x_full.index = list(range(x_full.shape[0]))

cat_var = x_full.select_dtypes(include = ['object']).columns.tolist()
num_var = x_full.select_dtypes(exclude = ['object']).columns.tolist()




for var in cat_var:
    x_full[var] = pd.factorize(x_full[var])[0]




"""
for num_col in num_var:
    unique_num = x_full[num_col].nunique()
    if unique_num > 200:
        print('Col: {}, Number : {}'.format(num_col, unique_num))
"""




def divide_function_label(full = None, col_list = None, new_col = None, unit_list = None):
    #Make relatvie features and separted into intervals
    full[new_col] = x_full[col_list[0]] / x_full[col_list[1]]
    new_col_list = []
    for unit in unit_list:
        new_col_unit = new_col + '_' + str(unit)
        full[new_col_unit], _ = divmod(x_full[new_col], unit)
        new_col_list.append(new_col_unit)
    
    return full, new_col_list




x_full, new_col_tan = divide_function_label(full = x_full, col_list = ['life_sq', 'full_sq'], new_col = 'tan_sq', unit_list = [0.2])
x_full, new_col_rel = divide_function_label(full = x_full, col_list = ['kitch_sq', 'full_sq'], new_col = 'rel_kitch', unit_list = [0.05])

x_full.max_floor.fillna(1, inplace = True)
x_full.max_floor[x_full.max_floor == 0] = 1
x_full.floor.fillna(-1, inplace = True)
x_full['tan_floor'] = x_full.floor / x_full.max_floor
no_max = (x_full.tan_floor > 1)
no_floor = (x_full.tan_floor < 0)
x_full['floor_label_0.1'],_ = divmod(x_full.tan_floor, 0.1)
x_full['floor_label_0.1'][no_max] = 11
x_full['floor_label_0.1'][no_floor] = -1




def rank_function_part2(train = None, col = None, target = None, rank_unit = None):
    #Make feature sorted by values with dic
    col_df = train[col].copy()
    col_val = pd.concat([col_df, np.expm1(target)], axis = 1)

    col_group = col_val.groupby(col, as_index = False)

    col_mean = col_group.mean().sort_values('price_doc').reset_index()
    col_mean.drop('index', axis = 1, inplace = True)
    rank = list(range(col_mean.shape[0]))
    col_mean['rank'] =  rank

    sam = dict()
    for itr in rank:
        key = col_mean[col][itr]
        value = col_mean['rank'][itr]
        sam[key] = value

    return sam




sam = rank_function_part2(train = house_train, col = 'sub_area', target = target)
x_full['sub_area_sample'] = x_full.sub_area.map(sam)




house_train = x_full[:num]
house_test =  x_full[num:]




import xgboost as xgb
def learn_xgb(train = None, test = None, target =None):
    from sklearn.model_selection import train_test_split
    
    eta = 0.1
    max_depth = 10
    subsample = 0.7
    colsample_bytree = 0.7
    random_state = 10
    params = {
            "objectvie": "reg:logistic",
            "eval_metric": "rmse",
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": random_state
        }
    num_boost_round = 200
    early_stopping_rounds = 10
    test_size = 0.3
    
    y_train, y_valid = train_test_split(target, test_size = test_size, random_state = random_state)
    x_train = train.loc[y_train.index]
    x_valid = train.loc[y_valid.index]
    
    dfull = xgb.DMatrix(train, target)
    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_valid, y_valid)
    dtest = xgb.DMatrix(test)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    gbm = xgb.train(params, dtrain, num_boost_round, evals = watchlist, early_stopping_rounds = early_stopping_rounds, verbose_eval = 30)
    
    return gbm




cols = ['full_sq', 'life_sq','tan_sq_0.2', 
        'floor', 'max_floor', 'floor_label_0.1', 'sub_area_sample']
train = house_train[cols]
test = house_test[cols]
gbm = learn_xgb(train = train, test = test, target = target)




cols = ['full_sq', 'life_sq','tan_sq_0.2',
        'floor', 'max_floor', 'floor_label_0.1', 'sub_area']
train = house_train[cols]
test = house_test[cols]
gbm = learn_xgb(train = train, test = test, target = target)




cols = ['full_sq', 'life_sq','tan_sq',
        'floor', 'max_floor', 'floor_label_0.1', 'sub_area']
train = house_train[cols]
test = house_test[cols]
gbm = learn_xgb(train = train, test = test, target = target)




cols = ['full_sq', 'life_sq','tan_sq',
        'floor', 'max_floor', 'tan_floor', 'sub_area']
train = house_train[cols]
test = house_test[cols]
gbm = learn_xgb(train = train, test = test, target = target)






