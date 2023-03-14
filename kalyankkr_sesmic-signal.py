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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)


# In[3]:


train= pd.read_csv("../input/train.csv",dtype={"acoustic_data":np.int16, "time_ro_failure": np.float64},nrows=1500000)
train.head()


# In[4]:


train.isna().sum()


# In[5]:


train.describe()


# In[6]:


plt.figure(figsize=(8,6))
plt.title("Distribution of Acoustic data")
ax= sns.distplot(train.acoustic_data,label="acustic_data")


# In[7]:


This shows most of the signal data centred around mean value of the signals


# In[8]:


plt.figure(figsize=(12,8))
plt.title("Sesmic signal time_to_failure")
plt.plot(train.time_to_failure,train.acoustic_data)

