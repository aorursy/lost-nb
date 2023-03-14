#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from geopy import distance # geo_distance
from math import cos, radians 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train =  pd.read_csv('../input/train.csv', nrows=1000000)
train.dtypes




train['pickup_datetime']= pd.to_datetime(train['pickup_datetime']) # convert to datetime
train.head()




train.isnull().sum()




train.dropna(inplace=True)




train.describe()




train.drop(train[train.fare_amount <= 0].index, inplace=True) # Drop -ve fare values
train.drop(train[(train.pickup_longitude > 180) | (train.dropoff_longitude > 180) | (train.pickup_longitude < -180) | (train.dropoff_longitude < -180)].index, inplace=True) # Drop unrealistic longtiudes
train.drop(train[(train.pickup_latitude > 90) | (train.dropoff_latitude > 90) | (train.pickup_latitude < -90) | (train.dropoff_latitude < -90)].index, inplace=True) # Drop unrealistic longtiudes
train.drop(train[train.passenger_count > 6].index, inplace=True) # Drop unrealistic passenger_count




minimum_fare = train[(train.pickup_longitude == train.dropoff_longitude) & 
                     (train.pickup_latitude == train.dropoff_latitude)].fare_amount.min()

train.drop(train[(train.pickup_longitude == train.dropoff_longitude) & 
                 (train.pickup_latitude == train.dropoff_latitude) & 
                 (train.fare_amount > minimum_fare)].index, inplace=True) # Drop expensive fares with 0 distances




train.describe()




train.drop(train[ (train.pickup_longitude > -71.9) | (train.dropoff_longitude > -71.9) | (train.pickup_longitude < -75.9) | (train.dropoff_longitude < -75.9) ].index, inplace=True) # Drop unrealistic longtiudes
train.drop(train[ (train.pickup_latitude > 42.7) | (train.dropoff_latitude > 42.7) | (train.pickup_latitude < 38.7) | (train.dropoff_latitude < 38.7) ].index, inplace=True) # Drop unrealistic longtiudes
train.describe()




# ALERT - 1M rows were computed in 5 mins on a powerful PC
# We will use this just as a reference later on.
def geodesic_distance (row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return distance.distance(pickup, dropoff).km

get_ipython().run_line_magic('time', "train['geo_distance'] = train.apply(lambda row: geodesic_distance(row), axis=1)")




degree_latitude = 111
degree_longtiude = cos(radians(train.pickup_latitude.mean())) * degree_latitude
print(degree_longtiude)




train['abs_diff_longitude'] = (train.dropoff_longitude - train.pickup_longitude).abs()
train['abs_diff_latitude'] = (train.dropoff_latitude - train.pickup_latitude).abs()

train['abs_diff_longitude_km'] = train.abs_diff_longitude * degree_longtiude
train['abs_diff_latitude_km'] = train.abs_diff_latitude * degree_latitude

train['manhattan_distance'] = train.abs_diff_longitude_km + train.abs_diff_latitude_km




### Angle difference between north, and manhattan roadways
meas_ang = radians(29)

## adding extra features
def add_extra_manh_features(df):
    df['euclidean'] = (df.abs_diff_latitude_km**2 + df.abs_diff_longitude_km**2)**0.5 ### as the crow flies  
    df['delta_manh_long'] = (df.euclidean*np.sin(np.arctan(df.abs_diff_longitude_km / df.abs_diff_latitude_km)-meas_ang)).abs()
    df['delta_manh_lat'] = (df.euclidean*np.cos(np.arctan(df.abs_diff_longitude_km / df.abs_diff_latitude_km)-meas_ang)).abs()
    df.delta_manh_long.fillna(0, inplace=True) # happens if pickup coords = dropoff coords
    df.delta_manh_lat.fillna(0, inplace=True)
    df['manhattan_distance_corrected'] = df.delta_manh_long + df.delta_manh_lat

add_extra_manh_features(train)




# Sanity Check, we will compare the 2 euclidean distances we calculated (one very accurate 'geo_distance' and one rounded by calculating the length of 1 degree of longtiude and latitude)
# and the 2 manhattan distances
train[['geo_distance', 'euclidean', 'manhattan_distance', 'manhattan_distance_corrected']].head()




# Ideally, we should break down the island into smaller pieces due to its orientation but this is only a rough estimation.
def manh_checker(x):
    if 40.7031 < x['dropoff_latitude'] < 40.83 and -74.017 < x['dropoff_longitude'] < -73.909 and        40.7031 < x['pickup_latitude'] < 40.83 and -74.017 < x['pickup_longitude'] < -73.909:
        return 1
    elif 40.7091 < x['dropoff_latitude'] < 40.83 and -74.017 < x['dropoff_longitude'] < -73.909 and         (not 40.7031 < x['pickup_latitude'] < 40.83 or not -74.017 < x['pickup_longitude'] < -73.909):
        return 2
    elif (not 40.7031 < x['dropoff_latitude'] < 40.83 or not -74.017 < x['dropoff_longitude'] < -73.909) and           40.7091 < x['pickup_latitude'] < 40.83 and -74.017 < x['pickup_longitude'] < -73.909:
        return 3
    else:
        return 0
    
train['manh_island'] = train.apply(manh_checker, axis=1) 




# Percentages:
print('Inside Manhattan = {:.2f}, Outside Manhattan = {:.2f}, Mixed = {:.2f}'.format(len(train[train.manh_island == 1])/len(train), len(train[train.manh_island == 0])/len(train), len(train[train.manh_island == 2])/len(train)))




def final_distance_calculator(x):
    if x.manh_island == 1:
        return x.manhattan_distance_corrected
    elif x.manh_island == 2 or x.manh_island == 3:
        return (x.manhattan_distance_corrected + x.manhattan_distance) / 2
    elif x.manh_island == 0:
        return x.manhattan_distance
train['final_distance'] = train.apply(final_distance_calculator, axis=1)




# Let's try to figure out the relation between the fare and the travelled distance
plt.scatter(train.final_distance, train.fare_amount, alpha=0.15)
plt.title('Avg Pricing: {:.3f} $/Km'.format(np.sum(train.fare_amount)/np.sum(train.final_distance)))
plt.xlabel('Distance in KM')
plt.ylabel('Fare in $')
plt.show()




train[(train.fare_amount > 300) & (train.final_distance < 20)][['fare_amount', 'final_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]




# What is this cheap cluster @ 120 to 150km?
train[(train.fare_amount < 100) & (train.final_distance > 100)][['fare_amount', 'final_distance']].head()




train['fare_rate'] = train.fare_amount/train.final_distance
print(train.fare_rate.describe())
print('Avg per KM: {}'.format(np.sum(train.fare_amount)/np.sum(train.final_distance)))
print(np.percentile(train.fare_rate, 99.5)) # Rate which 99.5% of trips fall behind
print(np.percentile(train.fare_rate, 0.3)) # Rate which 99.7% of trips fall after




train.drop(train[(train.fare_rate > np.percentile(train.fare_rate, 99.7)) |                   (train.fare_rate < np.percentile(train.fare_rate, 0.3))].index, 
           inplace=True) # Drop extremely expensive/cheap trips




plt.figure(figsize=(18, 10))
plt.subplot(221)
sns.barplot(train.pickup_datetime.dt.hour, train.fare_rate)
plt.xlabel('Hour')
plt.subplot(222)
sns.barplot(train.pickup_datetime.dt.month, train.fare_rate)
plt.xlabel('Month')
plt.subplot(223)
sns.barplot(train.pickup_datetime.dt.year, train.fare_rate)
plt.xlabel('Year')
plt.subplot(224)
sns.barplot(train.passenger_count, train.fare_rate)
plt.xlabel('Passengers')
plt.show()




train['morning_afternoon'] = train.pickup_datetime.dt.hour.apply(lambda x: 0 if 8 <= x <= 19 else 1)
train['last_4_months'] = train.pickup_datetime.dt.month.apply(lambda x: 0 if x >= 9 else 1)
train['exp_year'] = train.pickup_datetime.dt.year.apply(lambda x: 0 if x >= 2013 else 1)




# Let's see the data one more time after all the cleaning
plt.scatter(train.final_distance, train.fare_amount, alpha=0.15)
plt.title('Avg Pricing: {:.3f} $/Km'.format(np.sum(train.fare_amount)/np.sum(train.final_distance)))
plt.xlabel('Distance in KM')
plt.ylabel('Fare in $')
plt.show()




features = ['final_distance', 'morning_afternoon', 'last_4_months', 'exp_year']
x_train, x_test, y_train, y_test = train_test_split(train[features], train['fare_amount'], train_size = 0.8, test_size = 0.2, random_state=555)
lm = LinearRegression()
model = lm.fit(x_train, y_train)
y_predict= lm.predict(x_test)

print("Train score: ", lm.score(x_train, y_train))
print("Test score: ", lm.score(x_test, y_test))
print("Weights: ", lm.coef_)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.plot(range(300), range(300), alpha=0.25)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual Fare vs Predicted Fare")
plt.show()




test =  pd.read_csv('../input/test.csv')
test['pickup_datetime']= pd.to_datetime(test['pickup_datetime']) # convert to datetime
test.isnull().sum()




test['abs_diff_longitude'] = (test.dropoff_longitude - test.pickup_longitude).abs()
test['abs_diff_latitude'] = (test.dropoff_latitude - test.pickup_latitude).abs()
test['abs_diff_longitude_km'] = test.abs_diff_longitude * degree_longtiude
test['abs_diff_latitude_km'] = test.abs_diff_latitude * degree_latitude
test['manhattan_distance'] = test.abs_diff_longitude_km + test.abs_diff_latitude_km
add_extra_manh_features(test)
test['manh_island'] = test.apply(manh_checker, axis=1)
test['final_distance'] = test.apply(final_distance_calculator, axis=1)
test['morning_afternoon'] = test.pickup_datetime.dt.hour.apply(lambda x: 0 if 8 <= x <= 19 else 1)
test['last_4_months'] = test.pickup_datetime.dt.month.apply(lambda x: 0 if x >= 9 else 1)
test['exp_year'] = test.pickup_datetime.dt.year.apply(lambda x: 0 if x >= 2013 else 1)




lm = LinearRegression()
model = lm.fit(train[features], train['fare_amount'])
prediction = lm.predict(test[features])




submission = pd.DataFrame({"key": test.key, "fare_amount": prediction})
submission.to_csv('submission.csv', index=False)

