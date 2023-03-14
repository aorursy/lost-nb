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


train=  pd.read_csv("../input/train.csv")



# In[3]:


train.head()


# In[4]:


import matplotlib.pyplot as plt 
import seaborn as sns
train.scalar_coupling_constant.describe()
sns.distplot( train.scalar_coupling_constant.values)


# In[5]:


sns.countplot(train.type.values)


# In[6]:


train.boxplot(column="scalar_coupling_constant",        # Column to plot
                 by= "type",         # Column to split upon
                 figsize= (8,8)) 


# In[7]:


train.groupby('molecule_name',group_keys=False).size()


# In[ ]:





# In[8]:


test= pd.read_csv("../input/test.csv")


# In[9]:


test.head()


# In[10]:


sns.countplot(test.type.values)


# In[11]:


import sklearn
from sklearn.model_selection import train_test_split

nrow = len(train)
X_train, X_test = sklearn.model_selection.train_test_split(train,test_size=int(nrow*0.1),random_state=1)
X_train, X_val = sklearn.model_selection.train_test_split(X_train,test_size=int(nrow*0.2),random_state=1)


# In[12]:


dt=  X_train['scalar_coupling_constant'].groupby(X_train['type']).mean()


# In[13]:


X_train['Mean'] = X_train.groupby('type')['scalar_coupling_constant'].transform(np.average)  
X_train.head()

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(X_train.Mean.values, X_train.scalar_coupling_constant.values))

print("RMSE for mean by type is "+ str(rms))


# In[14]:


sample_submission=  pd.read_csv("../input/sample_submission.csv")
sample_submission.head()


# In[ ]:





# In[15]:


Poetential_energy= pd.read_csv("../input/potential_energy.csv")


# In[16]:


Poetential_energy.head()


# In[17]:


Mulliken_charges= pd.read_csv("../input/mulliken_charges.csv")


# In[18]:


Mulliken_charges.head()


# In[19]:


scalar_coupling_contributions= pd.read_csv("../input/scalar_coupling_contributions.csv")


# In[20]:


scalar_coupling_contributions.head()

