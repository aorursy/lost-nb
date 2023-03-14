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


ls ../input


# In[3]:


train_df = pd.read_csv("../input/train_relationships.csv")


# In[4]:


s1 = set((p1, p2) for i,(p1,p2) in train_df.iterrows())


# In[5]:


s2 = set((p2, p1) for i,(p1,p2) in train_df.iterrows())


# In[6]:


s1 == s2


# In[7]:


len(s1), len(s2), len(s1.union(s2))


# In[8]:


3598*2 == 7196
