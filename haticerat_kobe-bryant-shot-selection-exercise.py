#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!kaggle competitions download -c kobe-bryant-shot-selection


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics


# In[ ]:


df_raw = pd.read_csv('../input/data.csv', low_memory=False, 
                     parse_dates=["game_date"])


# In[ ]:


df_raw.dtypes


# In[ ]:


df_raw.info()


# In[ ]:


df_raw.drop('lat', axis=1, inplace=True)
df_raw.drop('lon', axis=1, inplace=True)
df_raw.drop('playoffs', axis=1, inplace=True)
df_raw.drop('team_id', axis=1, inplace=True)
df_raw.drop('team_name', axis=1, inplace=True)


# In[ ]:


df_raw.shape


# In[ ]:


add_datepart(df_raw, 'game_date')


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_raw.tail().T)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'train_cats')


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.head()


# In[ ]:


df_test_w_shot_made_flag = df_raw[df_raw['shot_made_flag'].isnull()]


# In[ ]:


df_test_w_shot_made_flag.shape


# In[ ]:


df_test_w_shot_made_flag.head()


# In[ ]:


df_test, y, nas = proc_df(df_test_w_shot_made_flag, 'shot_made_flag')


# In[ ]:


df_test.shape


# In[ ]:


df_w_shot_made_flag = df_raw[df_raw['shot_made_flag'].notnull()]


# In[ ]:


df_w_shot_made_flag.shape


# In[ ]:


df, y, nas = proc_df(df_w_shot_made_flag, 'shot_made_flag')


# In[ ]:


df.shape


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 5000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_w_shot_made_flag, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


shot_made_flag_predictions = m.predict(df_test)


# In[ ]:


shot_ids = df_test['shot_id']


# In[ ]:


df_result = pd.DataFrame({'shot_id': shot_ids, 'shot_made_flag': shot_made_flag_predictions})
df_result.to_csv('submission.csv', index=False)


# In[ ]:


df_result.tail()

