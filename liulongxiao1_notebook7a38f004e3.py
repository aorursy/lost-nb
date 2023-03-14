#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb


# In[2]:


print(np)
user_log=pd.read_csv('../input/user_logs.csv',chunksize=1000000)


# In[3]:


print(user_log)


# In[4]:


user_log1=user_log.__next__()


# In[5]:


print(user_l
      og1.head(5))

