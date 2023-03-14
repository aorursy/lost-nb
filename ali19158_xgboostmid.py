#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
import logging
import datetime
import warnings
import lightgbm as lgb
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


dstest = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
dstrain = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")


# In[3]:


dstest


# In[4]:


#dstestw = dstest.drop(['ID_code'], axis=1).astype('float32')
#dstrainw = dstrain.drop(['ID_code', 'target'], axis=1).astype('float32')


# In[5]:


dstest.shape


# In[6]:


dstest.info()
dstrain.info()


# In[7]:


dstest.describe()


# In[8]:


sns.countplot(dstrain['target'], palette='Set3')


# In[9]:


#X = dstrain.values
#y = dstrain.target.astype('uint8').values


# In[10]:


# def plot_feature_distribution(df1, df2, label1, label2, features):
#     i = 0
#     sns.set_style('whitegrid')
#     plt.figure()
#     fig, ax = plt.subplots(10,10,figsize=(18,22))

#     for feature in features:
#         i += 1
#         plt.subplot(10,10,i)
#         sns.distplot(df1[feature], hist=False,label=label1)
#         sns.distplot(df2[feature], hist=False,label=label2)
#         plt.xlabel(feature, fontsize=9)
#         locs, labels = plt.xticks()
#         plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
#         plt.tick_params(axis='y', which='major', labelsize=6)
#     plt.show();


# In[11]:


# t0 = dstrain.loc[dstrain['target'] == 0]
# t1 = dstrain.loc[dstrain['target'] == 1]
# features = dstrain.columns.values[2:102]
# plot_feature_distribution(t0, t1, '0', '1', features)


# In[12]:


plt.figure(figsize=(16,6))
features = dstrain.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(dstrain[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(dstest[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[13]:


# scale numerical features
from sklearn.preprocessing import StandardScaler


# In[14]:


st_sc = StandardScaler()
from sklearn.model_selection import train_test_split


# In[15]:


# split the inputs and outputs
X_train = dstrain.iloc[:, dstrain.columns != 'target']
y_train = dstrain.iloc[:, 1].values
X_test = dstest.iloc[:,dstest.columns != 'ID_code'].values
X_train = X_train.iloc[:,X_train.columns != 'ID_code'].values

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 0)


# In[16]:


# encode categoricals
from sklearn.preprocessing import LabelEncoder


# In[17]:


le = LabelEncoder()


# In[18]:


import xgboost as xgb


# In[19]:


xg_cl = xgb.XGBClassifier(
    objective = 'binary:logistic',
    n_estimators = 1000, seed=123,
    learning_rate=0.25, max_depth=2,
    colsample_bytree=0.35, subsample=0.82,
    min_child_weight=53, gamma = 9.9,tree_method='gpu_hist'
                          )


# In[20]:


# xg_cl.fit(X_train, y_train)


# In[21]:


# y_pred_xg=xg_cl.predict(X_test)


# In[22]:


# dataset_xg = pd.concat((dstest.ID_code, pd.Series(y_pred_xg).rename('target')), axis=1)
# dataset_xg.target.value_counts()


# In[23]:


# dataset_xg.to_csv('xgbost_mid.csv', index=False)


# In[24]:


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()


# In[25]:


# knn.fit(X_train, y_train)


# In[26]:


#  y_preds=knn.predict(X_test)


# In[27]:


# dataset_knn = pd.concat((dstest.ID_code, pd.Series(y_preds).rename('target')), axis=1)
# dataset_knn.target.value_counts()


# In[28]:


#dataset_knn.to_csv('knn_mid.csv', index=False)


# In[29]:


from sklearn import svm
clf = svm.SVC()


# In[ ]:





# In[30]:


clf.fit(X_train, y_train)


# In[31]:


clf_preds = clf.predict(X_test)


# In[32]:


dataset_clf = pd.concat((dstest.ID_code, pd.Series(clf_preds).rename('target')), axis=1)
dataset_clf.target.value_counts()


# In[33]:


dataset_clf.to_csv('clf_mid.csv', index=False)


# In[ ]:




