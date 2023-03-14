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


data = pd.read_csv('../input/train_ver2.csv', usecols=['ncodpers', 'fecha_dato', 'age',' fecha_alta'])


# In[3]:


data.shape


# In[4]:


df = data.loc[data.age.isnull(), 'ncodpers']


# In[5]:


data.age.isnull().sum()


# In[6]:


data.isnull().sum(axis=1).value_counts()


# In[7]:


data.loc[data.ncodpers==]
