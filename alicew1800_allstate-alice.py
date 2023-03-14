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


dstrain=pd.read_csv("../input/train.csv")
dstest=pd.read_csv("../input/test.csv")

print(dstrain.head())
print(dstest.head())


# In[3]:


dstrain=dstrain.iloc[:,1:]
testid=dstest['id']
dstest.drop('id',axis=1,inplace=True)

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

print(dstrain.head())
print(dstest.head())


# In[4]:


print(dstrain.shape)


# In[5]:


print(dstrain.describe())


# In[6]:


print(dstrain.skew())


# In[7]:


import seaborn as sb
import matplotlib.pyplot as plt

dscont=dstrain.iloc[:,116:]

for i in range(14)
    fg,ax=plt,subplots(nrow-1,ncols=1,figsize=(12,16))
    for j in range(1)
        sns.violinplot(y=cols[i+j],data=dscont,ax=ax[j])

