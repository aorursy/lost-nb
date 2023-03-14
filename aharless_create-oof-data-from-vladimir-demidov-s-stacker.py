#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from numba import jit
from sklearn.linear_model import LogisticRegression

os.listdir('../input')


# In[2]:


oof = pd.read_csv('../input/kaggleportosegurocnoof/stacker_oof_1.csv')
oof.head()


# In[3]:


train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')
train[['id','target']].head()


# In[4]:


df = pd.merge(train[['id','target']], oof, on='id')
df.head(10)


# In[5]:


# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


# In[6]:


# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)
y_valid_pred = 0*df['target']


# In[7]:


stacker = LogisticRegression()


# In[8]:


for i, (train_index, test_index) in enumerate(kf.split(df)):
    
    # Create data for this fold
    y_train, y_valid = df['target'].iloc[train_index].copy(), df['target'].iloc[test_index]
    X_train, X_valid = df[['target0','target1','target2']].iloc[train_index,:].copy(),                        df[['target0','target1','target2']].iloc[test_index,:].copy()
    print( "\nFold ", i)
    
    stacker.fit(X_train, y_train)
    pred = stacker.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    
    y_valid_pred.iloc[test_index] = pred


# In[9]:


print( "\nGini for full training set:" )
eval_gini(df['target'], y_valid_pred)


# In[10]:


val = pd.DataFrame()
val['id'] = df['id'].values
val['target'] = y_valid_pred.values
val.to_csv('stacker_oof_preds_1.csv', float_format='%.6f', index=False)


# In[11]:


val.head()

