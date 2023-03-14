#!/usr/bin/env python
# coding: utf-8



#!kaggle competitions download -c kobe-bryant-shot-selection




from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics




df_raw = pd.read_csv('../input/data.csv', low_memory=False, 
                     parse_dates=["game_date"])




df_raw.dtypes




df_raw.info()




df_raw.drop('lat', axis=1, inplace=True)
df_raw.drop('lon', axis=1, inplace=True)
df_raw.drop('playoffs', axis=1, inplace=True)
df_raw.drop('team_id', axis=1, inplace=True)
df_raw.drop('team_name', axis=1, inplace=True)




df_raw.shape




add_datepart(df_raw, 'game_date')




def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)




display_all(df_raw.tail().T)




get_ipython().run_line_magic('pinfo2', 'train_cats')




display_all(df_raw.isnull().sum().sort_index()/len(df_raw))




train_cats(df_raw)




df_raw.head()




df_test_w_shot_made_flag = df_raw[df_raw['shot_made_flag'].isnull()]




df_test_w_shot_made_flag.shape




df_test_w_shot_made_flag.head()




df_test, y, nas = proc_df(df_test_w_shot_made_flag, 'shot_made_flag')




df_test.shape




df_w_shot_made_flag = df_raw[df_raw['shot_made_flag'].notnull()]




df_w_shot_made_flag.shape




df, y, nas = proc_df(df_w_shot_made_flag, 'shot_made_flag')




df.shape




def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 5000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_w_shot_made_flag, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape




def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)




m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




shot_made_flag_predictions = m.predict(df_test)




shot_ids = df_test['shot_id']




df_result = pd.DataFrame({'shot_id': shot_ids, 'shot_made_flag': shot_made_flag_predictions})
df_result.to_csv('submission.csv', index=False)




df_result.tail()

