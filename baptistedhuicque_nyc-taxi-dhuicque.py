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

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime
import math

import os
from pathlib import Path
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[2]:


import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv ('../input/train.csv')
test = pd.read_csv ('../input/test.csv')


# In[4]:


train.head(10)


# In[5]:


train.describe().transpose()


# In[6]:


train['trip_duration'] = np.log(train['trip_duration'].values)


# In[7]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_minute'] = train['pickup_datetime'].dt.minute
train['pickup_time']=train['pickup_hour']*60 +train['pickup_minute']


# In[8]:


test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['pickup_minute'] = test['pickup_datetime'].dt.minute
test['pickup_time']=test['pickup_hour']*60 +test['pickup_minute']


# In[9]:


from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371800  # Earth radius in meters  
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 +         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


# In[10]:


train['distance'] = train.apply(lambda row: haversine(row['pickup_latitude'],row['pickup_longitude'],row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
test['distance']  = test.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'],row['dropoff_latitude'], row['dropoff_longitude']), axis=1)


# In[11]:


train["distance"].describe().transpose()


# In[12]:


X = train[['distance','passenger_count','pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude','pickup_time']
]
y = train['trip_duration']
X.shape, y.shape


# In[13]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import ShuffleSplit


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[15]:


m1 = RandomForestRegressor(n_estimators=30, min_samples_leaf= 3)
m1.fit(X_train, y_train)


# In[16]:


X_test=m1.predict(test[['distance','passenger_count','pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude','pickup_time']])
X_test, len(X_test)


# In[17]:


submit = pd.read_csv('../input/sample_submission.csv')


# In[18]:


my_submission = pd.DataFrame({'id': test.id, 'trip_duration':np.exp(X_test)})


# In[19]:


my_submission.to_csv('submission.csv', index=False)

