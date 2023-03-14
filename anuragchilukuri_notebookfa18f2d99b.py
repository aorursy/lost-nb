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


train = pd.read_json("../input/train.json")
train.head()


# In[3]:


train.columns


# In[4]:


train_bb_dummy=pd.get_dummies(train,columns=['bathrooms','bedrooms'])


# In[5]:


train_bb_dummy['DayOfWeek']=pd.to_datetime(train_bb_dummy['created']).dt.dayofweek
def weekend(x):
    if x==5 or x==6:
        return 'weekend'
    else:
        return 'weekday'
train_bb_dummy['WeekendId']=train_bb_dummy['DayOfWeek'].map(weekend)


# In[6]:


train_bb_dummy['hour']=pd.to_datetime(train_bb_dummy['created']).dt.hour


# In[7]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = np.array([]))


# In[8]:


train_bb_dummy['description'][100004]


# In[9]:





# In[9]:


#possible transformations -

1. Make bathrooms, bedrooms sparse - No need

2. sort by building id

3. sort by date, split date to week day/weekend. Split time to 4 halves

4. Make description sparse by cpnverting the common keywords into a sparse matrix

5. Make the features sparse

6. Transform latitude and longitude to useful features


# In[10]:


train_bb_dummy['features'][10000]


# In[11]:





# In[11]:


from sklearn 
