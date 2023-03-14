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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv", low_memory=False)\ntest = pd.read_csv("../input/test.csv", low_memory=False)')


# In[3]:


# basic models
# from https://www.kaggle.com/ankitdhall97/basic-models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb


# In[4]:


train.shape


# In[5]:


train.head()


# In[6]:


from sklearn.model_selection import *


# In[7]:


train_sub = train.sample(frac=0.2)
X = train_sub.drop(["ID_code", "target"],axis=1)
y = train_sub["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[8]:


[print(x.shape) for x in [X_train, X_test, y_train, y_test]]


# In[9]:


from sklearn.metrics import *


# In[10]:


def model_score(model):
    return {"train":roc_auc_score(y_train, model.predict(X_train)),
            "test":roc_auc_score(y_test, model.predict(X_test))}


# In[11]:


lsvc = LinearSVC(verbose=True)
get_ipython().run_line_magic('time', 'lsvc.fit(X_train, y_train)')
model_score(lsvc)


# In[12]:


xgb = XGBClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'xgb.fit(X_train, y_train)')
model_score(xgb)


# In[13]:


lr = LogisticRegression(n_jobs=-1)
get_ipython().run_line_magic('time', 'lr.fit(X_train, y_train)')
model_score(lr)


# In[14]:


rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=10)
get_ipython().run_line_magic('time', 'rf.fit(X_train, y_train)')
model_score(rf)


# In[15]:


gbc = GradientBoostingClassifier(verbose=1)
get_ipython().run_line_magic('time', 'gbc.fit(X_train, y_train)')
model_score(gbc)


# In[16]:


gnb = GaussianNB()
get_ipython().run_line_magic('time', 'gnb.fit(X_train, y_train)')
get_ipython().run_line_magic('time', 'model_score(gnb)')


# In[17]:


train_sub_true = train.loc[train.target == 1,:].reset_index(drop=True)
train_sub_true.shape


# In[18]:


train_sub_false = train.loc[train.target != 1,:]                    .sample(n=train_sub_true.shape[0]).reset_index(drop=True)
train_sub_2 = pd.concat([train_sub_true, train_sub_false],axis=0)                            .reset_index(drop=True)


# In[19]:


train_sub_2.shape


# In[20]:


X_train_2, X_test_2, y_train_2, y_test_2 =     train_test_split(train_sub_2.drop(["ID_code", "target"],axis=1),
                     train_sub_2["target"])


# In[21]:


def model_score_2(model):
    return {"train":roc_auc_score(y_train_2, model.predict(X_train_2)),
            "test":roc_auc_score(y_test_2, model.predict(X_test_2))}


# In[22]:


gnb2 = GaussianNB()
get_ipython().run_line_magic('time', 'gnb2.fit(X_train_2, y_train_2)')
get_ipython().run_line_magic('time', 'model_score_2(gnb2)')


# In[23]:


lsvc2 = LinearSVC(verbose=True)
get_ipython().run_line_magic('time', 'lsvc2.fit(X_train_2, y_train_2)')
model_score_2(lsvc2)


# In[24]:


xgb2 = XGBClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'xgb2.fit(X_train_2, y_train_2)')
model_score_2(xgb2)


# In[25]:


lr2 = LogisticRegression(n_jobs=-1)
get_ipython().run_line_magic('time', 'lr2.fit(X_train_2, y_train_2)')
model_score_2(lr2)


# In[26]:


rf2 = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=10)
get_ipython().run_line_magic('time', 'rf2.fit(X_train_2, y_train_2)')
model_score_2(rf2)


# In[27]:


gbc2 = GradientBoostingClassifier(verbose=1)
get_ipython().run_line_magic('time', 'gbc2.fit(X_train_2, y_train_2)')
model_score_2(gbc2)


# In[28]:


from IPython.display import display
def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)


# In[29]:


display_all(train.describe().T)


# In[30]:


def make_features(df):
    df = df.copy()
    df["mean"] = df.mean(axis=1)
    df["skew"] = df.skew(axis=1)
    df["std"] = df.std(axis=1)
    df["kurt"] = df.kurt(axis=1)
    df["max"] = df.max(axis=1)
    df["min"] = df.min(axis=1)
    df["max_min"] = df["max"] - df["min"]
    df["mean_std"] = df["mean"] - df["std"]
    return df


# In[31]:


train_sub_3 = make_features(train_sub_2.drop(["ID_code", "target"], axis=1))


# In[32]:


X_train_3, X_test_3, y_train_3, y_test_3 =     train_test_split(train_sub_3,
                     train_sub_2["target"])


# In[33]:


def model_score_3(model):
    return {"train":roc_auc_score(y_train_3, model.predict(X_train_3)),
            "test":roc_auc_score(y_test_3, model.predict(X_test_3))}


# In[34]:


gnb3 = GaussianNB()
get_ipython().run_line_magic('time', 'gnb3.fit(X_train_3, y_train_3)')
get_ipython().run_line_magic('time', 'model_score_3(gnb3)')


# In[35]:


X_for_predict = test.drop(["ID_code"], axis=1)


# In[36]:


submission = pd.DataFrame({"ID_code":test.ID_code, 
                           "target":gnb2.predict_proba(X_for_predict)[:,1]})


# In[37]:


submission.head()


# In[38]:


submission.to_csv("submission.csv", index=False)


# In[39]:




