#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


from fastai.vision import *


# In[3]:


pip freeze


# In[4]:


pip install chainer


# In[5]:


#!pip install joypy --progress-bar off


# In[6]:


pip install cupy --no-cache-dir -vvvv


# In[7]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[8]:


import numpy as np
import pandas as pd
import cudf
import cupy as cp
from cuml.neighbors import KNeighborsRegressor
from cuml import SVR
from cuml.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from cuml.metrics import mean_absolute_error, mean_squared_error


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# In[9]:


fnc_df = cudf.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = cudf.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = cudf.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[10]:


# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/500

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# In[11]:



NUM_FOLDS = 7
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)


features = loading_features + fnc_features

overal_score = 0
for target, c, w, ff in [("age", 60, 0.3, 0.55), ("domain1_var1", 12, 0.175, 0.2), ("domain1_var2", 8, 0.175, 0.2), ("domain2_var1", 9, 0.175, 0.22), ("domain2_var2", 12, 0.175, 0.22)]:    
    y_oof = np.zeros(df.shape[0])
    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))
    
    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
        train_df = train_df[train_df[target].notnull()]

        model_1 = SVR(C=c, cache_size=3000.0)
        model_1.fit(train_df[features].values, train_df[target].values)
        model_2 = Ridge(alpha = 0.0001)
        model_2.fit(train_df[features].values, train_df[target].values)
        val_pred_1 = model_1.predict(val_df[features])
        val_pred_2 = model_2.predict(val_df[features])
        
        test_pred_1 = model_1.predict(test_df[features])
        test_pred_2 = model_2.predict(test_df[features])
        
        val_pred = (1-ff)*val_pred_1+ff*val_pred_2
        val_pred = cp.asnumpy(val_pred.values.flatten())
        test_pred = (1-ff)*test_pred_1+ff*test_pred_2
        test_pred = cp.asnumpy(test_pred.values.flatten())

        y_oof[val_ind] = val_pred
        y_test[:, f] = test_pred
        
    df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)
    
    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
    mae = mean_absolute_error(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
    rmse = np.sqrt(mean_squared_error(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values))
    overal_score += w*score
    print(target, np.round(score, 8))
    print(target, np.round(score, 4))
    print(target, 'mean absolute error:', np.round(mae, 8))
    print(target, ' root mean square error:', np.round(rmse, 8))
    print()
    
print("Overal score:", np.round(overal_score, 8))
print("Overal score:", np.round(overal_score, 4))


# In[12]:


sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[13]:


sub_df.to_csv("submission_rapids_ensemble.csv", index=False)

