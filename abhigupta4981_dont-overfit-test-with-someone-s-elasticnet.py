#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet


# In[2]:


get_ipython().run_line_magic('pinfo2', 'ElasticNet')


# In[3]:


PATH = Path('../input')


# In[4]:


train = pd.read_csv(PATH/'train.csv')
test = pd.read_csv(PATH/'test.csv').drop(columns=['id'])


# In[5]:


train_Y = train['target']
train_X = train.drop(columns=['target', 'id'])


# In[6]:


best_parameters = {
    'alpha': 0.2,
    'l1_ratio': 0.31,
    'precompute': True,
    'selection': 'random',
    'tol': 0.001, 
    'random_state': 2
}


# In[7]:


net = ElasticNet(**best_parameters)
net.fit(train_X, train_Y)


# In[8]:


sub = pd.read_csv(PATH/'sample_submission.csv')
sub['target'] = net.predict(test)


# In[9]:


sub.head()


# In[10]:


sub.to_csv('submission.csv', index=False)


# In[11]:


FileLink('submission.csv')


# In[12]:




