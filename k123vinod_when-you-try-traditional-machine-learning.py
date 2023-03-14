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


train = pd.read_csv("/kaggle/input/data-without-drift/train_clean.csv")


# In[3]:


train.shape


# In[4]:


test = pd.read_csv("/kaggle/input/data-without-drift/test_clean.csv")
test.shape


# In[5]:


test.head()


# In[6]:


sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
sub.head()


# In[7]:


sub.shape


# In[8]:


X = train.drop('open_channels', axis=1).copy()


# In[9]:


y = train['open_channels'].copy()


# In[10]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.95, test_size=0.05,random_state = 0)


# In[11]:


train_X.shape


# In[12]:


import lightgbm as lgb 
lg = lgb.LGBMClassifier()
lg = lg.fit(train_X,train_y)
predict_l = lg.predict(val_X)


# In[13]:


from sklearn.metrics import f1_score
print('F1 Score: %.2f'
     % f1_score(val_y,predict_l,average='macro'))


# In[14]:


# lg = lgb.LGBMClassifier()
# lg = lg.fit(X,y)
predict_l = lg.predict(test)


# In[15]:


df = pd.DataFrame({'time':test.time, 'open_channels':predict_l})


# In[16]:


df.head(50)


# In[17]:


df.to_csv("submission.csv",index=False,float_format='%.4f')


# In[18]:


sub = pd.read_csv('submission.csv')


# In[ ]:




