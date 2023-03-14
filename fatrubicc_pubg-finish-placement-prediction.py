#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[2]:


Ntrain = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')


# In[3]:


Ntrain.head()


# In[4]:


Ntrain[Ntrain['winPlacePerc'].isnull()]


# In[5]:


Ntrain.drop(2744604,inplace = True)


# In[6]:


Ntrain['playerJoined'] = Ntrain.groupby('matchId')['matchId'].transform('count')


# In[7]:


plt.figure(figsize = (15,10))
sns.countplot(Ntrain[Ntrain['playerJoined']>= 60]['playerJoined'])
plt.show()


# In[8]:


Ntrain['killsNorm'] = Ntrain['kills']*((Ntrain['playerJoined'] - 1)/(100-1))
#maxPlaceNorm
#damageDealtNorm
Ntrain['damageDealtNorm'] = Ntrain['damageDealt'] * ((Ntrain['playerJoined'] - 1)/99)
#matchDurationNorm


# In[9]:


Ntrain[['kills','killsNorm','damageDealt','damageDealtNorm','playerJoined']][:11]


# In[10]:


Ntrain['totalDistance'] = Ntrain['rideDistance'] + Ntrain['swimDistance'] + Ntrain['walkDistance']


# In[11]:


Ntrain['totalDistance'].head(11)


# In[12]:


Ntrain['killWithoutMove'] = ((Ntrain['kills'] > 0) & (Ntrain['totalDistance'] == 0))


# In[13]:


np.shape(Ntrain[Ntrain['killWithoutMove'] == True])


# In[14]:


Ntrain.drop(Ntrain[Ntrain['killWithoutMove'] == True].index,inplace = True)


# In[15]:


plt.figure(figsize=(12,4))
sns.distplot(Ntrain['boosts'], bins=10)
plt.show()


# In[16]:


print(np.shape(Ntrain))
Ntrain.drop(Ntrain[Ntrain['boosts'] >11].index,inplace = True)
print(np.shape(Ntrain))


# In[17]:


plt.figure(figsize=(12,4))
sns.distplot(Ntrain['weaponsAcquired'], bins=10)
plt.show()


# In[18]:


print(np.shape(Ntrain))
Ntrain.drop(Ntrain[Ntrain['weaponsAcquired'] >20].index,inplace = True)
print(np.shape(Ntrain))


# In[19]:


train = Ntrain[:200000]
np.shape(train)


# In[20]:


train.head(10)


# In[21]:


train = train.drop(['Id','groupId','matchId','matchType','killWithoutMove','kills','damageDealt','numGroups','swimDistance','playerJoined'],axis = 1)


# In[22]:


Y_train = train['winPlacePerc']
X_train = train.drop(['winPlacePerc'],axis = 1)
X_train.head(21)


# In[23]:


Y_train.head()


# In[24]:


X_train.shape,Y_train.shape


# In[25]:


# Metric used for the PUBG competition (Mean Absolute Error (MAE))
from sklearn.metrics import mean_absolute_error


# In[26]:


m1 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1)
m1.fit(X_train, Y_train)
mean_absolute_error(m1.predict(X_train), Y_train)


# In[27]:


X_test = test.copy()


# In[28]:


X_test.head(11)


# In[29]:


X_test['playerJoined'] = X_test.groupby('matchId')['matchId'].transform('count')
X_test['killsNorm'] = X_test['kills']*((X_test['playerJoined'] - 1)/(100-1))
#maxPlaceNorm
#damageDealtNorm
X_test['damageDealtNorm'] = X_test['damageDealt'] * ((X_test['playerJoined'] - 1)/99)
X_test['totalDistance'] = X_test['rideDistance'] + X_test['swimDistance'] + X_test['walkDistance']


# In[30]:


X_test.head(11)


# In[31]:


X_test = X_test.drop(['Id','groupId','matchId','matchType','kills','damageDealt','numGroups','swimDistance','playerJoined'],axis = 1)


# In[32]:


X_test.head(11)
np.shape(X_test),np.shape(test["Id"])


# In[33]:


I = np.clip(a = m1.predict(X_test), a_min = 0.0, a_max = 1.0)


# In[34]:


I.shape


# In[35]:


submission = pd.DataFrame({
        "Id": test["Id"],
        "winPlacePerc": I
    })
submission.to_csv('submission.csv', index=False)


# In[36]:


submission.head(10)


# In[37]:





# In[37]:





# In[37]:




