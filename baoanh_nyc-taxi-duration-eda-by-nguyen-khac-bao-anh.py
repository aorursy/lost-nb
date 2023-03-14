#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualiser les données
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
from sklearn.preprocessing import LabelEncoder
# package geo pour traiter les données coordinates
from geopy.geocoders import Nominatim
# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set({'figure.figsize':(16,8)})


# In[3]:


train = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")
test = pd.read_csv("../input/nyc-taxi-trip-duration/test.csv")


# In[4]:


print(f"shape of training set{train.shape}")
print(f"shape of testing set{test.shape}")


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


print(f"La différence de la variable entre data training et data testing:{set(train.columns).difference(set(test.columns))}")


# In[8]:


for i,v in zip(list(train.isnull().sum().index),list(train.isnull().sum().values)):
    print(f"{i} a {v} valeur(s) manquant(s)")


# In[9]:


for i,v in zip(list(test.isnull().sum().index),list(test.isnull().sum().values)):
    print(f"{i} a {v} valeur(s) manquant(s)")


# In[10]:


print(f"Il y a {train.duplicated().sum()} données duplicates dans le train")


# In[11]:


print(f"Il y a {test.duplicated().sum()} données duplicates dans le test")


# In[12]:


train.describe().T


# In[13]:


train.info()


# In[14]:


print(f"vendor_id ont {len(train.vendor_id.unique())} valeur: {list(train.vendor_id.unique())}")


# In[15]:


print(f"store_and_fwd_flag ont {len(train.store_and_fwd_flag.unique())} valeur: {list(train.store_and_fwd_flag.unique())}")


# In[16]:


print(f"passenger_count ont {len(train.passenger_count.unique())} valeur: {sorted(list(train.passenger_count.unique()))}")


# In[17]:


train.trip_duration.hist();


# In[18]:


len(train.trip_duration[train.trip_duration>6000].values)


# In[19]:


train.loc[train.trip_duration<6000,"trip_duration"].hist(bins=100)
plt.title("distribution de trip duration sans les oulieurs");


# In[20]:


plt.hist(np.log(train.trip_duration), bins=1000, edgecolor='red');


# In[21]:


train['log_trip_duration'] = np.log(train['trip_duration'])


# In[22]:


plt.hist(train.loc[train.vendor_id==1, 'log_trip_duration'], bins=100, edgecolor='red')
plt.hist(train.loc[train.vendor_id==2, 'log_trip_duration'], bins=100, edgecolor='violet')
plt.xlabel("trip duration")
plt.ylabel("frequency")
plt.legend(['vendor_id=1', 'vendor_id=2']);


# In[23]:


train.groupby(['vendor_id','passenger_count'])['trip_duration'].agg('mean').unstack(level=0).plot()
plt.ylabel("trip duration average")
plt.xlabel("nombre de passenger")
plt.title("Trip duration by number of passenger on each vendor");


# In[24]:


train[['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']].head()


# In[25]:


fig,ax = plt.subplots(2,1)
sns.scatterplot(x='pickup_longitude', y='pickup_latitude',data=train,ax=ax[0])
plt.ylim([31,53]);
sns.scatterplot(x='dropoff_longitude', y='dropoff_latitude',data=train,ax=ax[1])
plt.ylim([31,53]);


# In[26]:


le = LabelEncoder()
le.fit(train['store_and_fwd_flag'])
train['store_and_fwd_flag'] = le.transform(train['store_and_fwd_flag'])
test['store_and_fwd_flag'] = le.transform(test['store_and_fwd_flag'])


# In[27]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['month'] = train['pickup_datetime'].dt.month
train['day'] = train['pickup_datetime'].dt.day
train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
train['weekday'] = train['pickup_datetime'].dt.weekday
train['hour'] = train['pickup_datetime'].dt.hour
train['minute'] = train['pickup_datetime'].dt.minute
#train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)


# In[28]:


# petite verification
#train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime'])\
#                                    .map(lambda x: x.total_seconds())
#print(f"il y a {sum(train['check_trip_duration'] != train.trip_duration)} ligne(s) ne pas correspondre")


# In[29]:


train.groupby(['vendor_id','dayofweek'])['trip_duration'].agg("mean").unstack(level=0).plot()
plt.xlabel("Journé dans la semaine (lundi=0, dimanche=6)")
plt.ylabel("Trip duration moyenne");


# In[30]:


train.groupby(['vendor_id','hour'])['trip_duration'].agg("mean").unstack(level=0).plot()
plt.xlabel("L'heure dans la journée")
plt.ylabel("Trip duration moyenne");


# In[31]:


train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']

train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))


# In[32]:


#### spatial features: count and speed
train['pickup_longitude_bin'] = np.round(train['pickup_longitude'], 2)
train['pickup_latitude_bin'] = np.round(train['pickup_latitude'], 2)
train['dropoff_longitude_bin'] = np.round(train['dropoff_longitude'], 2)
train['dropoff_latitude_bin'] = np.round(train['dropoff_latitude'], 2)


# In[33]:


test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
test['month'] = test['pickup_datetime'].dt.month
test['day'] = test['pickup_datetime'].dt.day
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
test['weekday'] = test['pickup_datetime'].dt.weekday
test['hour'] = test['pickup_datetime'].dt.hour
test['minute'] = test['pickup_datetime'].dt.minute


# In[34]:


test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']
test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))


# In[35]:


test['pickup_longitude_bin'] = np.round(test['pickup_longitude'], 2)
test['pickup_latitude_bin'] = np.round(test['pickup_latitude'], 2)
test['dropoff_longitude_bin'] = np.round(test['dropoff_longitude'], 2)
test['dropoff_latitude_bin'] = np.round(test['dropoff_latitude'], 2)


# In[36]:


## count features
a = pd.concat([train,test]).groupby(['pickup_longitude_bin', 'pickup_latitude_bin']).size().reset_index()
b = pd.concat([train,test]).groupby(['dropoff_longitude_bin', 'dropoff_latitude_bin']).size().reset_index()

train = pd.merge(train, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')
test = pd.merge(test, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')

train = pd.merge(train, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')
test = pd.merge(test, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')

## speed features
train['speed'] = 100000*train['dist'] / train['trip_duration']

a = train[['speed', 'pickup_longitude_bin', 'pickup_latitude_bin']].groupby(['pickup_longitude_bin', 'pickup_latitude_bin']).mean().reset_index()
a = a.rename(columns = {'speed': 'ave_speed'})
b = train[['speed', 'dropoff_longitude_bin', 'dropoff_latitude_bin']].groupby(['dropoff_longitude_bin', 'dropoff_latitude_bin']).mean().reset_index()
b = b.rename(columns = {'speed': 'ave_speed'})

train = pd.merge(train, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')
test = pd.merge(test, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')

train = pd.merge(train, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')
test = pd.merge(test, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')

## drop bins
train = train.drop(['speed', 'pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin'], axis = 1)
test = test.drop(['pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin'], axis = 1)


# In[37]:


#### weather data
weather = pd.read_csv('../input/knycmetars2016/KNYC_Metars.csv')
weather['Time'] = pd.to_datetime(weather['Time'])
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hour'] = weather['Time'].dt.hour
weather = weather[weather['year'] == 2016]

train = pd.merge(train, weather[['Temp.', 'month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')
test = pd.merge(test, weather[['Temp.', 'month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')


# In[38]:


# export data training and data testing
train.to_csv("training_data.csv", index=False)
test.to_csv("testing_data.csv", index=False)


# In[39]:


col_diff = list(set(train.columns).difference(set(test.columns)))
print(f"La différence de la variable entre data training et data testing:{set(train.columns).difference(set(test.columns))}")


# In[40]:




