#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians, asin
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# import googlemaps

# gmaps = googlemaps.Client(key='')

# Set Figsize
plt.rcParams['figure.figsize'] = (16,10)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=500000, parse_dates=['pickup_datetime'])


# In[ ]:


# test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


train.describe()


# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(len(train['key']))
print(376/len(train['key']))


# In[ ]:


train = train.dropna()


# In[ ]:


train.dtypes


# In[ ]:


fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['pickup_longitude'].values,
             train['pickup_latitude'].values, color="red", s=1, label="pickup", alpha=0.1)
ax[0].set_xlim((-75, -73))
ax[0].set_ylim((40, 41.5))
ax[1].scatter(train['dropoff_longitude'].values,
             train['dropoff_latitude'].values, color="blue", s=1, label="pickup", alpha=0.1)
ax[1].set_xlim((-75, -73))
ax[1].set_ylim((40, 41.5))
plt.show()


# In[ ]:


train.fare_amount.describe()


# In[ ]:


train = train[train['fare_amount']>0]
print(len(train))
print(train.fare_amount.describe())


# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(train['fare_amount'].values, bins=100)
plt.ylabel('fare_amount')
plt.xlabel('number of training')
plt.show()


# In[ ]:


train['log_fare_amount'] = np.log(train['fare_amount'].values + 1)
train['log_fare_amount'].plot.hist(bins=100, figsize=(8,6))
plt.ylabel('log(fare_amount)')
plt.xlabel('number of training')
plt.show()


# In[ ]:


train.log_fare_amount.describe()


# In[ ]:


plt.figure(figsize=(8,6))
plt.boxplot(train['fare_amount'])
plt.show()


# In[ ]:


train.passenger_count.plot.hist(figsize=(8,6))


# In[ ]:


train.groupby('passenger_count')['fare_amount'].mean()


# In[ ]:


grp = train.groupby('passenger_count')['passenger_count'].count()
print(grp)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(grp)
plt.show()


# In[ ]:


1791/len(train['passenger_count'])


# In[ ]:


train = train[train['passenger_count'] != 0]
print(train.describe())


# In[ ]:


train.passenger_count.plot.hist(figsize=(8,6))


# In[ ]:


grp = train.groupby('passenger_count')['passenger_count', 'fare_amount'].mean()
print(grp)


# In[ ]:


# This is extract of the Stackoverflow but I have problem with this
def calc_distance(lat1,lon1, lat2, lon2):
    # Approximate radius of earht in km
    R = 6373.0
    dlon = radians(lon2) - radians(lon1)
    dlat = radians(lat2) - radians(lat1)
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    # c = 2 * atan2(sqrt(a), sqrt(1 - a))
    c = 2 * asin(sqrt(a))    
    distance = R * c
    return distance

# This is use on  https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration. 
# I don't have problem with this
def dis(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    #  This Return Miles. I want to KMs
    # return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
    miles = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return miles * 1.609344 # Kms


# In[ ]:


train['distance_easy'] = train.apply(lambda x: dis(x['pickup_latitude'], x['pickup_longitude'],
                                                             x['dropoff_latitude'], x['dropoff_longitude']), axis=1)


# In[ ]:


train.head(5)


# In[ ]:


sample_google_distance = ['2.8 km', '10.1 km', '1.4 km', '3.9 km', '2.0 km', '4.4 km', '2.2 km', '6.1 km', '2.3 km', '4.0 km']
sample_google_duration = ['9 mins', '28 mins', '7 mins', '20 mins', '5 mins', '18 mins', '11 mins', '19 mins', '11 mins', '20 mins']

sample_google_distance = [float(d.split(' ')[0]) for d in sample_google_distance]
sample_google_duration = [int(d.split(' ')[0]) for d in sample_google_duration]


# In[ ]:


print(sample_google_distance)
print(sample_google_duration)


# In[ ]:


print("Mean squared error: %.2f"
      % mean_squared_error(sample_google_distance, train.distance_easy[:10]))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(sample_google_distance, train.distance_easy[:10]))


# In[ ]:


x = train.as_matrix(['distance_easy'])
reg = linear_model.LinearRegression()
reg.fit(x[:8], sample_google_distance[:8])
y_pred = reg.predict(x[8:10])
m = reg.coef_[0]
b = reg.intercept_
print(m)
print (b)


# In[ ]:


print(y_pred)
print("Mean squared error: %.2f"
      % mean_squared_error(y_pred, sample_google_distance[8:10]))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_pred, sample_google_distance[8:10]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




