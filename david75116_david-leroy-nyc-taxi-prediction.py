#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime
import math

import os
from pathlib import Path
print(os.listdir("../input"))




df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')




df_train.head()




# check data usage
print('Memory usage, Mb: {:.2f}\n'.format(df_train.memory_usage().sum()/2**20))

# overall df info
print('---------------- DataFrame Info -----------------')
print(df_train.info())




print(df_train.isnull().sum())




print('----------------distance Outliers-------------------')
print('Latitude : {} to {}'.format(
    max(df_train.pickup_latitude.min(), df_train.dropoff_latitude.min()),
    max(df_train.pickup_latitude.max(), df_train.dropoff_latitude.max())
))
print('Longitude : {} to {}'.format(
    max(df_train.pickup_longitude.min(), df_train.dropoff_longitude.min()),
    max(df_train.pickup_longitude.max(), df_train.dropoff_longitude.max())
))
print('')
print('------------------Time Outliers---------------------')
print('Trip duration in seconds: {} to {}'.format(
    df_train.trip_duration.min(), df_train.trip_duration.max()))

print('')
print('------------------Date Outliers---------------------')
print('Datetime range: {} to {}'.format(df_train.pickup_datetime.min(), 
                                        df_train.dropoff_datetime.max()))
print('')
print('----------------Passengers Outliers------------------')
print('Passengers: {} to {}'.format(df_train.passenger_count.min(), 
                                        df_train.passenger_count.max()))




print('duplicates IDs: {}'.format(len(df_train) - len(df_train.drop_duplicates(subset='id'))))




def haversine(lat1, lon1, lat2, lon2):
    R = 6371800  # Earth radius in meters  
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 +         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))




#trop LONG et peu précis
#%%time
#df_train['distance'] = df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(haversine_distance, axis=1)
#df_train.head()
#df_test['distance'] = df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(haversine_distance, axis=1)




#rapide mais moins performant
#from math import radians, cos, sin, asin, sqrt

#def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    #lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    #dlon = lon2 - lon1 
    #dlat = lat2 - lat1 
    #a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    #c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    #km = 6371* c
    #return km

#def haversine_distance(x):
    #x1, y1 = np.float64(x['pickup_longitude']), np.float64(x['pickup_latitude'])
    #x2, y2 = np.float64(x['dropoff_longitude']), np.float64(x['dropoff_latitude'])    
    #return haversine(x1, y1, x2, y2)




df_train['distance'] = df_train.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
df_test['distance']  = df_test.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)




df_train.head()




#sns.set(rc={'figure.figsize':(15,10)})
#sns.distplot(df_train['distance'],hist=False)




#RABDOM FOREST REGRESSOR <=> NO CLEAN


#outliers temporels
#duration_Proportion = ((df_train.trip_duration < 60) | # < 1 min 
#            (df_train.trip_duration > 8000)) # > 3 hours
#print('Anomalies in trip duration, %: {:.2f}'.format(
#    df_train[duration_Proportion].shape[0] / df_train.shape[0] * 100))

#outliers passagers
#df_train = df_train[df_train['passenger_count']>0]
#df_train = df_train[df_train['passenger_count']<6]

#outliers coordonnés
#df_train = df_train.loc[df_train['pickup_longitude']> -80]
#df_train = df_train.loc[df_train['pickup_latitude']< 44]
#df_train = df_train.loc[df_train['dropoff_longitude']> -90]
#df_train = df_train.loc[df_train['dropoff_latitude']> 34]

#outliers distances
#df_train = df_train[df_train['distance']>1]
#df_train = df_train[df_train['distance']<120000 




#delete
#df_train = df_train[~duration_Proportion]
#Check
#print('Trip duration in seconds: {} to {}'.format(
#   df_train.trip_duration.min(), df_train.trip_duration.max()
#))




plt.figure(figsize=(8,5))
sns.distplot(df_train['trip_duration']).set_title("Distribution of Trip Duration")
plt.xlabel("Trip Duration")




df_train['trip_duration'] = np.log(df_train['trip_duration'].values)




df_train[pd.isnull(df_train)].sum()




df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')




df_train['hour'] = df_train.loc[:,'pickup_datetime'].dt.hour;
df_train['week'] = df_train.loc[:,'pickup_datetime'].dt.week;
df_train['weekday'] = df_train.loc[:,'pickup_datetime'].dt.weekday;
df_train['hour'] = df_train.loc[:,'pickup_datetime'].dt.hour;
df_train['month'] = df_train.loc[:,'pickup_datetime'].dt.month;

df_test['hour'] = df_test.loc[:,'pickup_datetime'].dt.hour;
df_test['week'] = df_test.loc[:,'pickup_datetime'].dt.week;
df_test['weekday'] = df_test.loc[:,'pickup_datetime'].dt.weekday;
df_test['hour'] = df_test.loc[:,'pickup_datetime'].dt.hour;
df_test['month'] = df_test.loc[:,'pickup_datetime'].dt.month;




cat_vars = ['store_and_fwd_flag']
for col in cat_vars:
    df_train[col] = df_train[col].astype('category').cat.codes
df_train.head()

for col in cat_vars:
    df_test[col] = df_test[col].astype('category').cat.codes
df_test.head()




y_train = df_train["trip_duration"]
X_train = df_train[["vendor_id", "store_and_fwd_flag","passenger_count", "pickup_longitude", "pickup_latitude", "distance", "dropoff_longitude","dropoff_latitude", "hour", "week", "weekday", "month" ]]




get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\nm = RandomForestRegressor(n_estimators=100,min_samples_leaf=3, min_samples_split=15, n_jobs=-1, max_features="auto")\nm.fit(X_train, y_train)')




X_test = df_test[["vendor_id", "store_and_fwd_flag","passenger_count","pickup_longitude", "pickup_latitude", "distance","dropoff_longitude","dropoff_latitude", "hour", "week", "weekday", "month"]]
prediction = m.predict(X_test)
prediction




submit = pd.read_csv('../input/sample_submission.csv')
submit.head()




my_submission = pd.DataFrame({'id': df_test.id, 'trip_duration': np.exp(prediction)})
my_submission.head()




my_submission.to_csv('submission.csv', index=False)

