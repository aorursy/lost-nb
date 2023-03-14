#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


import os
from pathlib import Path

import datetime as dt
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
sns.set({'figure.figsize':(16,10)})
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(os.listdir("../input"))


# In[3]:


train = pd.read_csv('../input/dataset-de-landry/train.csv')
test = pd.read_csv('../input/dataset-de-landry/test.csv')
sample = pd.read_csv('../input/dataset-de-landry/sample_submission.csv')


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


train.tail()


# In[7]:


train.dtypes


# In[8]:


train.isna().sum()


# In[9]:


train.trip_duration.min()


# In[10]:


train.trip_duration.max()


# In[11]:


fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)
ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.0002, alpha=1)


# In[12]:


fig, ax = plt.subplots(7, sharex=True)
for i,c in enumerate(["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","trip_duration"]):
    sns.boxplot(train[c],ax=ax[i],width=1.5)
    ax[i].set_xscale("log")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(c, fontsize=15,rotation=45)
fig.suptitle('Analyse des outliers', fontsize=20)


# In[13]:


train.loc[train.trip_duration<4000,"trip_duration"].hist(bins=120)


# In[14]:


train = train[train['passenger_count']>0]
train = train[train['passenger_count']<9]


# In[15]:




train = train[(train['trip_duration'] > 60) & (train['trip_duration'] < 3600)]


train['trip_duration'] = np.log(train['trip_duration'].values)

train['hour'] = train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))


test['hour'] = test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))

#outliers coordonnés
train = train.loc[train['pickup_longitude']> -80]
train = train.loc[train['pickup_latitude']< 44]
train = train.loc[train['dropoff_longitude']> -90]
train = train.loc[train['dropoff_latitude']> 34]


# In[16]:


def haversine(lat1, lon1, lat2, lon2):
   R = 6372800  # Earth radius in meters
   phi1, phi2 = math.radians(lat1), math.radians(lat2)
   dphi       = math.radians(lat2 - lat1)
   dlambda    = math.radians(lon2 - lon1)

   a = math.sin(dphi/2)**2 +        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

   return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']

train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))

train['speed'] = 100000*train['dist'] / train['trip_duration']


# In[17]:


train.isnull().sum()


# In[18]:


col_diff = list(set(train.columns).difference(set(test.columns)))

train.head()


# In[19]:


y_train = train["trip_duration"] # <-- target
X_train = train[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","hour"]] # <-- features

X_datatest = test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","hour"]]


# In[20]:


train.drop(['speed','dist','hour']+col_diff, axis=1, inplace=True)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)


# In[22]:


#rfr = RandomForestRegressor(n_estimators=200,min_samples_leaf=5, min_samples_split=15, max_depth=80,verbose=0,max_features="auto",n_jobs=-1)
#rfr.fit(X_train, y_train)


# In[23]:


# Un peu long
# calculer les scores de cross validation du model selon une decoupe du dataset de train
# cv_scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring= 'neg_mean_squared_log_error')


# In[24]:


# cv_scores


# In[25]:


#for i in range(len(cv_scores)):
#    cv_scores[i] = np.sqrt(abs(cv_scores[i]))
#print(np.mean(cv_scores))

## xgb parameters
params = {
    'booster':            'gbtree',
    'objective':          'reg:linear',
    'learning_rate':      0.1,
    'max_depth':          14,
    'subsample':          0.8,
    'colsample_bytree':   0.7,
    'colsample_bylevel':  0.7,
    'silent':             1
}


# In[26]:


nrounds = 1200
dtrain = xgb.DMatrix(X_train, np.log(y_train+1))
gbm = xgb.train(params,
                dtrain,
                num_boost_round = nrounds)


# In[27]:


#train_pred = rfr.predict(X_datatest)
train_pred = np.exp(gbm.predict(xgb.DMatrix(X_datatest))) - 1


# In[28]:


train_pred


# In[29]:


len(train_pred)


# In[30]:


sample.shape[0]


# In[31]:


my_submission = pd.DataFrame({"id": test.id, "trip_duration": np.exp(train_pred)})
my_submission.head()


# In[32]:




my_submission.to_csv('submission.csv', index=False)
my_submission.head()


# In[33]:





# In[33]:





# In[33]:




