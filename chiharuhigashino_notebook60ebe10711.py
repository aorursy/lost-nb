#!/usr/bin/env python
# coding: utf-8

# In[1]:



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


# In[2]:


np.random.seed(151)  # noqa

import numpy as np
import pandas as pd
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# neural network
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import json

# models for classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# models for regression
from sklearn.ensemble import RandomForestRegressor, 

from sklearn.cross_validation import KFold;


# In[3]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[4]:


train.head(10)


# In[5]:


def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_f = scaler.fit_transform(train.values.astype(np.float32))
    return train_f

full = [train, test]
drop_elements = ['Open Date', 'City','City Group', 'Type']
train_cor = train.drop(drop_elements, axis=1)
train_f = scale(train_cor)
train_cor.describe()


# In[6]:


# training set
trainX = train.drop(['Id','revenue'],axis=1)
trainX = pd.get_dummies(trainX)
trainX = trainX.as_matrix()
trainY = train['revenue'].as_matrix()

# check data
print(trainX.shape)
print(trainY.shape)
trainX


# In[7]:


# Random Forest Regressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
rfr = RandomForestRegressor(n_estimators=5, criterion='mse', max_depth=10)
rfr.fit(trainX,trainY)
predictY = rfr.predict(trainX)


# In[8]:


# check difference
compare = pd.DataFrame(trainY, columns=['target'])
compare['predict'] = predictY
compare['diff'] = predictY - trainY

# display
pd.options.display.float_format = '{:,.0f}'.format
compare


# In[9]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of P-Variables', y=1.05, size=15)
sns.heatmap(train_cor.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[10]:





# In[10]:




