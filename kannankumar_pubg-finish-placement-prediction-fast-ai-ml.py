#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from IPython.display import display
from pandas_summary import DataFrameSummary


# In[2]:


PATH = '/kaggle/input/pubg-finish-placement-prediction/'
get_ipython().system('ls -l {PATH}')


# In[3]:


def display_all(df):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        display(df)    


# In[4]:


df_raw = pd.read_csv(f'{PATH}/train_V2.csv')
display_all(df_raw.head().T)


# In[5]:


display_all(df_raw.describe(include='all').T)


# In[6]:


df_raw.dtypes


# In[7]:


# total memory usage of Dataframe
df_raw.memory_usage(deep=True).sum() * 1e-6


# In[8]:


# column wise memory usage of Dataframe
df_raw.memory_usage(deep=True) * 1e-6


# In[9]:


df_raw.drop(['Id'], axis=1, inplace=True)


# In[10]:


# total memory usage of Dataframe after dropping Id column
df_raw.memory_usage(deep=True).sum() * 1e-6


# In[11]:


from pandas.api.types import is_string_dtype,                                 is_categorical_dtype,                                 is_numeric_dtype


# In[12]:


def show_string_cols(df):
    """Print names of all string columns of a dataframe"""
    for name, col in df.items():
        if is_string_dtype(col):
            print(name, end=' , ')
            
def show_categorical_cols(df, show_categories=False, ignore_cols=None):
    """Print names [and categories] of all string columns of dataframe"""
    if not ignore_cols: ignore_cols=[]
    for name, col in df.items():
        if is_categorical_dtype(col) and (name not in ignore_cols):
            if show_categories:
                print(f'{name} : {len(df[name].cat.categories)} categories')
            else:
                print(name, end=' , ')
                
def train_cats(df, ignore_cols=None):
    """Convert string columns of dataframe to categorical inplace"""
    if not ignore_cols: ignore_cols=[]
    for name, col in df.items():
        if is_string_dtype(col) and (name not in ignore_cols):
            df[name] = col.astype('category').cat.as_ordered()                


# In[13]:


show_string_cols(df_raw)


# In[14]:


train_cats(df_raw, ignore_cols=['Id'])
show_categorical_cols(df_raw, show_categories=True,  ignore_cols=['Id'])


# In[15]:


def numericalize(df, ignore_cols=None):
    if not ignore_cols: ignore_cols=[]
    for name, col in df.items():
        if not is_numeric_dtype(col) and (name not in ignore_cols):
            df[name] = pd.Categorical(col).codes+1            


# In[16]:


numericalize(df_raw, ignore_cols=['Id'])


# In[17]:


def show_missing(df):
    display_all(df.isnull().sum().sort_values(ascending=False)/len(df)*100)
    
def fix_missing(df):
    for name, col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                filler = col.median()
                df[name] = col.fillna(filler)


# In[18]:


# Missing values in numeric columns
show_missing(df_raw)


# In[19]:


fix_missing(df_raw)
# After filling missing values in all numeric columns
show_missing(df_raw)


# In[20]:


def split_xy(df, y_fld):
    """Split dataframe into X(features) and y(target)"""
    y = df[y_fld].values
    df_new = df.drop(y_fld, axis=1)    
    return df_new, y

def dummify(df):
    df_new = pd.get_dummies(df)
    return df_new


# In[21]:


df, y = split_xy(df_raw, 'winPlacePerc')
# df = dummify(df)
display_all(df.head().T)


# In[22]:


len(df_raw)


# In[23]:


n_valid = 50000
print(f'Train Rows: {len(df_raw) - n_valid}')
print(f'Valid Rows: {n_valid}')


# In[24]:


def split_vals(arr, split_index):
    """Split a list like object into two at the split_index"""
    return arr[:split_index].copy(), arr[split_index:].copy()


# In[25]:


n_train = len(df) - n_valid

raw_train, raw_valid = split_vals(df_raw, n_train)
X_train, X_valid = split_vals(df, n_train)
y_train, y_valid = split_vals(y, n_train)

print(f'X train: {X_train.shape}, y train: {y_train.shape} \nX valid: {X_valid.shape}, y valid: {y_valid.shape}')


# In[26]:


import math
def mae(y_hat, y): return abs(y - y_hat).mean()
def mae_sk(y_hat, y): return metrics.mean_absolute_error(y, y_hat)

def print_score(m):
    res = [mae_sk(m.predict(X_train), y_train), mae_sk(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[27]:


X_train_copy = X_train.copy()


# In[28]:


m = RandomForestRegressor(max_samples=500000, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[29]:


df_test = pd.read_csv(f'{PATH}/test_V2.csv')
display_all(df_test.head().T)


# In[30]:


display_all(df_test.describe(include='all').T)


# In[31]:


train_cats(df_test, ignore_cols=['Id'])
numericalize(df_test, ignore_cols=['Id'])
fix_missing(df_test)
display_all(df_test.head().T)


# In[32]:


len(df_test)


# In[33]:


# Create Baseline model predictions
get_ipython().run_line_magic('time', "sub = m.predict(df_test.drop(['Id'], axis=1))")
sub


# In[34]:


# Refer Sample Submission File
sample_sub_df = pd.read_csv(f'{PATH}/sample_submission_V2.csv')
sample_sub_df.head(10)


# In[35]:


# Create submission structure
sub_df = pd.DataFrame({'Id' : df_test['Id'], 
                      'winPlacePerc':m.predict(df_test.drop('Id', axis=1))})
sub_df.head()


# In[36]:


# Write submission file
sub_df.to_csv('/kaggle/working/submission.csv', sep=',', index=False)

tmp_df = pd.read_csv('/kaggle/working/submission.csv')
tmp_df.head()

