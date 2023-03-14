#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import xgboost as xgb # XGBoost model
import matplotlib.pyplot as plt
import math

Importing Libraries
# In[2]:


def rmsle(pred, train):
	label = train.get_label()
	assert len(pred) == len(label)
	label = label.tolist()
	pred = pred.tolist()
	summ = [(math.log(label[i] + 1) - math.log(max(0, pred[i]) + 1)) ** 2.0 for i, j in enumerate(label)]
	return 'rmsle', (sum(summ) * (1.0 / len(pred))) ** 0.5

RMSLE function to calculate score locally
# In[3]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_macro = pd.read_csv('../input/macro.csv')

Importing data to DataFrame
# In[4]:


y_train = df_train['price_doc'].values

Assigning prices to variable
# In[5]:


df_train = df_train.drop('price_doc', axis=1)

Dropping price_doc which was assigned to y_train, thus we don't need it anymore
# In[6]:


df_train = pd.merge(df_train, df_macro, how='left', on='timestamp')
df_test = pd.merge(df_test, df_macro, how='left', on='timestamp')

Merging training and test set with macro on timestamp column
# In[7]:


print(df_train.dtypes.value_counts())
print(df_test.dtypes.value_counts())

Checking what type of columns we have
# In[8]:


df_train['timestamp'] = pd.to_numeric(pd.to_datetime(df_train['timestamp'])) / 1e18
df_test['timestamp'] = pd.to_numeric(pd.to_datetime(df_test['timestamp'])) / 1e18
print(df_train['timestamp'].head())

We should not delete timestamp because all other variables are dependant of it, so we convert it to float, dividing by 1e18 because 1e17 gives us too high float and dividing by 1e19 gives us too small float
# In[9]:


df_train = df_train.drop(['id'], axis=1)

Dropping id from df_train dataframe, we don't need it
# In[10]:


result = pd.DataFrame()

Creating future result DataFrame
# In[11]:


result['id'] = df_test['id'].values
df_test = df_test.drop(['id'], axis=1)

Assigning id to result dataframe and then dropping it, we don't need it
# In[12]:


train_columns = list(
	set(df_train.select_dtypes(include=['float64', 'int64']).columns))

test_columns = list(
	set(df_test.select_dtypes(include=['float64', 'int64']).columns))

We are training on  flaot64 and int64 columns, although we lose some data from object columns
# In[13]:


x_train = df_train[train_columns].values
x_test = df_test[test_columns].values

Train/Valid split
# In[14]:


#from sklearn.model_selection import train_test_split
#x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size = 0.2, random_state = 64)

Random set of train/validation locally performs worse than manually assigning them
# In[15]:


split = 25000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

We could use Imputer to fill NaNs but XGBoost does it for us, if we were to use Imputer, then filling NaNs with median performs equally good, although mean and most_frequent performs worse locally
# In[16]:


d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

Creating Matrixes in order to train model
# In[17]:


params = {}
params['objective'] = 'reg:linear' # linear regression
params['eta'] = 0.031              # step size shrinkage used in update to prevent overfitting
params['max_depth'] = 6            # maximum depth of a tree
params['silent'] = 1               # silent mode


# In[18]:


evals = [(d_train, 'train'), (d_valid, 'valid')]

Creating evals: list of pairs of items to be evaluated during training
# In[19]:


clf = xgb.train(params, d_train, 800, evals, feval=rmsle, early_stopping_rounds=100)

Training of a model, final result is ~ train-rmsle:0.419234    valid-rmsle:0.408311
# In[20]:


x_test = xgb.DMatrix(x_test)
pred_test = clf.predict(x_test)


# In[21]:


result['price_doc'] = pred_test
result.to_csv('results.csv', index=False)


# In[22]:


Importing to CV
Overall kaggle score is ~0.324

