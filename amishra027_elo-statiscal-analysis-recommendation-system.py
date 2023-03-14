#!/usr/bin/env python
# coding: utf-8



from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import get_dummies
import lightgbm as lgb
import plotly.graph_objs as go
import plotly.offline as py
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv

get_ipython().run_line_magic('matplotlib', 'inline')




#Loading Train and Test Data
df_train = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"] )
df_test = pd.read_csv('../input/test.csv' ,parse_dates=["first_active_month"] )
df_merchants=pd.read_csv('../input/merchants.csv')
df_new_merchant_transactions=pd.read_csv('../input/new_merchant_transactions.csv')
df_historical_transactions = pd.read_csv("../input/historical_transactions.csv")




sample_submission = pd.read_csv("../input/sample_submission.csv")




sample_submission.head()




print(df_train.info())




print("Shape of train set                 : ",df_train.shape)
print("Shape of test set                  : ",df_test.shape)
print("Shape of historical_transactions   : ",df_historical_transactions.shape)
print("Shape of merchants                 : ",df_merchants.shape)
print("Shape of new_merchant_transactions : ",df_new_merchant_transactions.shape)




#Missing Data Analysis on train set
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
missing_data.head(20)




#Missing Data Analysis on test set
total = df_test.isnull().sum().sort_values(ascending = False)
percent = (df_test.isnull().sum()/df_test.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
missing_data.head(20)




df_train.target.plot.hist()




# Code reference from https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
# SRK - Simple Exploration Notebook

import seaborn as sns

cnt_srs = df_train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = df_test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()




df_train.corr()




df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])
df_test = pd.get_dummies(df_test, columns=['feature_1', 'feature_2'])
df_train.head()




df_card_history  = df_historical_transactions['card_id'].value_counts().head(10)
df_card_history




df_card_history_authorized  = df_historical_transactions['authorized_flag'].value_counts()
df_card_history_authorized




df_non_authorized = df_historical_transactions[df_historical_transactions['authorized_flag'] == "N"]
df_non_authorized




df_card_unauth  = df_non_authorized['card_id'].value_counts()
df_card_unauth.head(10)




df_card_1_unauth= df_non_authorized[df_non_authorized['card_id']=="C_ID_5ea401d358"]
df_card_2_unauth = df_non_authorized[df_non_authorized['card_id']=="C_ID_3d3dfdc692"]




df_card_1_merch = df_card_1_unauth['merchant_id'].value_counts()
df_card_1_merch.head(10)




df_card_1_merch = df_card_2_unauth['merchant_id'].value_counts()
df_card_1_merch.head(10)




df_card_new  = df_new_merchant_transactions['card_id'].value_counts().head(10)
df_card_new




df_non_authorized = df_new_merchant_transactions[df_new_merchant_transactions['authorized_flag'] == "N"]
df_non_authorized




df_new_merchant_transactions.head()




df_new_merchant_transactions['purchase_amount_integer'] = df_new_merchant_transactions.purchase_amount.apply(lambda x: x == np.round(x))
print(df_new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())




df_new_merchant_transactions['purchase_amount_new'] = np.round(df_new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,8)




#rounding off two decimal places
df_new_merchant_transactions['purchase_amount_new'] = np.round(df_new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)

df_new_merchant_transactions['purchase_amount_integer'] = df_new_merchant_transactions.purchase_amount_new.apply(lambda x: x == np.round(x))
print(df_new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())




df_new_merchant_transactions.groupby('purchase_amount_new')['card_id'].count().reset_index(name='count').sort_values('count',ascending=False).head(100)




df_historical_transactions = pd.get_dummies(df_historical_transactions, columns=['category_2', 'category_3'])
df_historical_transactions['authorized_flag'] = df_historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
df_historical_transactions['category_1'] = df_historical_transactions['category_1'].map({'Y': 1, 'N': 0})

df_historical_transactions.head()




df_historical_transactions['purchase_amount_integer'] = df_historical_transactions.purchase_amount.apply(lambda x: x == np.round(x))
print(df_historical_transactions.groupby('purchase_amount_integer')['card_id'].count())




#df_historical_transactions['purchase_amount_new'] = np.round(df_historical_transactions['purchase_amount'] / 0.00150265118 + 497.06,8)
df_historical_transactions['purchase_amount_new'] = np.round(df_historical_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
df_historical_transactions['purchase_amount_integer'] = df_historical_transactions.purchase_amount_new.apply(lambda x: x == np.round(x))
print(df_historical_transactions.groupby('purchase_amount_integer')['card_id'].count())




df_historical_transactions.groupby('purchase_amount_new')['card_id'].count().reset_index(name='count').sort_values('count',ascending=False).head(100)




del df_historical_transactions['purchase_amount_integer']




df_historical_transactions.dtypes




####

def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount_new': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans




###### Aggregate transaction function on the historical transaction ######

import gc
merch_hist = aggregate_transactions(df_historical_transactions, prefix='hist_')

del df_historical_transactions
gc.collect()
df_train = pd.merge(df_train, merch_hist, on='card_id',how='left')
df_test = pd.merge(df_test, merch_hist, on='card_id',how='left')

del merch_hist
gc.collect()
df_train.head()




df_new_merchant_transactions = pd.get_dummies(df_new_merchant_transactions, columns=['category_2', 'category_3'])
df_new_merchant_transactions['authorized_flag'] = df_new_merchant_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
df_new_merchant_transactions['category_1'] = df_new_merchant_transactions['category_1'].map({'Y': 1, 'N': 0})

df_new_merchant_transactions.head()




######Aggregate transaction function on the new merchant transaction ######

merch_new = aggregate_transactions(df_new_merchant_transactions, prefix='new_')
del df_new_merchant_transactions
gc.collect()
df_train = pd.merge(df_train, merch_new, on='card_id',how='left')
df_test = pd.merge(df_test, merch_new, on='card_id',how='left')

del merch_new
gc.collect()
df_train.head()




###### Dropping 3 columns ######

target = df_train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in df_train.columns if c not in drops]
features = list(df_train[use_cols].columns)
df_train[features].head()






print(df_train[features].shape)
print(df_test[features].shape)




df_train.corr()




#### Utilizing KFold Cross Validator ####

from sklearn.model_selection import KFold




###LGB model detals for different parameters referenced from  the below notebook
####https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737

param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

################# 5  fold cross validation #################


### Splitting the dataset into 5 folds for validation
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_5 = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof_5[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits




print('5 Fold Cross Validation - lgb - ', np.sqrt(mean_squared_error(oof_5, target)))




###LGB model detals for different parameters referenced from  the below notebook
####https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737

param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}



################# 10  fold cross validation #################



### Splitting the dataset into 10 folds for validation
folds = KFold(n_splits=10, shuffle=True, random_state=15)
oof_10 = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof_10[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits




print('10 Fold Cross Validation - lgb - ', np.sqrt(mean_squared_error(oof_10, target)))




sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] =  predictions #0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df.to_csv("submission.csv", index=False)




sub_df.tail(5)

