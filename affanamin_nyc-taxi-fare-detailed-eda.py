#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




# Importing Libraries for EDA

#Importing Libraries

#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




data_train = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows = 200000,parse_dates=["pickup_datetime"])




print(len(data_train));




#Data exploration
data_train.info()




data_train.describe()




data_train.head()




#Null value exploration code
print(data_train.isnull().sum())




#So we will enter there a median value.

median1 = data_train['dropoff_longitude'].median()
data_train['dropoff_longitude'].fillna(median1, inplace=True)

median2 = data_train['dropoff_latitude'].median()
data_train['dropoff_latitude'].fillna(median2, inplace=True)




print(data_train.isnull().sum())




data_train["pickup_longitude"] = pd.to_numeric(data_train.pickup_longitude, errors='coerce')
data_train["pickup_latitude"] = pd.to_numeric(data_train.pickup_latitude, errors='coerce')
data_train["dropoff_longitude"] = pd.to_numeric(data_train.dropoff_longitude, errors='coerce')
data_train["dropoff_latitude"] = pd.to_numeric(data_train.dropoff_latitude, errors='coerce')




data_train.head()




data_train.dtypes




from math import pi,sqrt,sin,cos,atan2

def haversine(pickUp_lat,pickUp_long,dropOff_lat,dropOff_long):
    lat1 = pd.to_numeric(pickUp_lat, errors='coerce')
    long1 = pd.to_numeric(pickUp_long, errors='coerce')

    lat2 = pd.to_numeric(dropOff_lat, errors='coerce')
    long2 = pd.to_numeric(dropOff_long, errors='coerce')

    #lat1 = pickUp_lat
    #long1 = pickUp_long
    #lat2 = dropOff_lat
    #long2 = dropOff_long


    #degree_to_rad = float(pi / 180.0)
    degree_to_rad =  0.017453292519943295
    d_lat = (lat2 - lat1) * degree_to_rad
    d_long = (long2 - long1) * degree_to_rad

    a = pow(np.sin(d_lat / 2), 2) + np.cos(lat1 * degree_to_rad) * np.cos(lat2 * degree_to_rad) * pow(np.sin(d_long / 2), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km = 6367 * c
    #mi = 3956 * c

    #return km#"km":km, "miles":mi}
    return km




res = haversine(40.721319,-73.844311,40.712278,-73.841610)
print("Distance is {} km".format(round(res)))




# add new column to dataframe with distance in miles
data_train['distance_km'] = haversine(data_train.pickup_latitude, data_train.pickup_longitude,data_train.dropoff_latitude, data_train.dropoff_longitude)




data_train.head()




data_train['year'] = data_train.pickup_datetime.apply(lambda t: t.year)
data_train['month'] = data_train.pickup_datetime.apply(lambda t: t.month)
data_train['weekday'] = data_train.pickup_datetime.apply(lambda t: t.weekday())
data_train['hour'] = data_train.pickup_datetime.apply(lambda t: t.hour)




#Some statisitcs


statistics_of_data = []
for col in data_train.columns:
  statistics_of_data.append((col,
                             data_train[col].nunique(),
                             data_train[col].isnull().sum()*100/data_train.shape[0],
                             data_train[col].value_counts(normalize=True, dropna=False).values[0] * 100, 
                             data_train[col].dtype
                             ))
stats_df = pd.DataFrame(statistics_of_data, columns=['Feature', 'Uniq_val', 'missing_val', 'val_biggest_cat', 'type'])




stats_df.sort_values('val_biggest_cat', ascending=False)




## Now lets explore features one by one

def exploreFeatures(col):
  top_n=10
  top_n = top_n if data_train[col].nunique() > top_n else data_train[col].nunique()
  #print(f"{col} has {data_train[col].nunique()} unique values and type: {data_train[col].dtype}.")
  #txt2 = "My name is {0}, I'am {1}".format("John",36)
  print("col has {0} unique values and type {1}:".format(data_train[col].nunique(),data_train[col].dtype))
  print(data_train[col].value_counts(normalize=True, dropna=False).head(10))




exploreFeatures('passenger_count')




exploreFeatures('fare_amount')




exploreFeatures('weekday')




exploreFeatures('hour')




exploreFeatures('distance_km')




exploreFeatures('pickup_latitude')




exploreFeatures('pickup_longitude')




exploreFeatures('dropoff_longitude')




exploreFeatures('dropoff_latitude')




exploreFeatures('year')




exploreFeatures('month')




data_train[data_train['distance_km'] == 0.00]




## SO, We will remove them

data_train = data_train[data_train['distance_km'] != 0.00]




data_train




##Lets test it
exploreFeatures('distance_km')




## Any coordiantes wrongly planted ??
## Does New York City have a beach?

#New York City has 14 miles of beaches, from beauties in the Bronx, to the historical sands of Brooklyn, to surfing in Queens.

#First of all we need to create a boundary (Bounding Box). NYC boundary is:

data_test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')

mindatapoints = min(data_test.pickup_longitude.min(), data_test.dropoff_longitude.min())
maxdatapoints = max(data_test.pickup_longitude.max(), data_test.dropoff_longitude.max())

print("minimum LONGITUDE data points {0} maximum data points {1} in NYC".format(mindatapoints,maxdatapoints))




mindatapointsLAT = min(data_test.pickup_latitude.min(), data_test.dropoff_latitude.min())
maxdatapointsLAT = max(data_test.pickup_latitude.max(), data_test.dropoff_latitude.max())

print("minimum LATITUDE data points {0} maximum data points {1} in NYC".format(mindatapointsLAT,maxdatapointsLAT))




# Now Creating a Boundary.

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])




#Boundary is :
BB = (-74.5, -72.8, 40.5, 41.8)




print('Old size: %d' % len(data_train))
data_train = data_train[select_within_boundingbox(data_train, BB)]
print('New size: %d' % len(data_train))




#We have successfully removed all data points which are not in the boundary of NYC.Now we need to remove data points in water, as they are Noisy data-points.

def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'),                (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    # define bounding box
    BB = (-74.5, -72.8, 40.5, 41.8)
    
    # read nyc mask and turn into boolean map with
    # land = True, water = False
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    # calculate for each lon,lat coordinate the xy coordinate in the mask map
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    # calculate boolean index
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    # return only datapoints on land
    return df[idx]




print('Old size: %d' % len(data_train))
data_train = remove_datapoints_from_water(data_train)
print('New size: %d' % len(data_train))




#Strat From Airport, Did we find any lead ?
jfkAirport_Corrd = (-73.7822222222, 40.6441666667)
LGAAirport_Corrd = (-73.87, 40.77)
EWRAirport_Coord = (-74.175, 40.69)
nyc = (-74.0063889, 40.7141667)




def absoluteDataPoint(loc, name):
    range=1.5
    idx0 = (haversine(data_train.pickup_latitude, data_train.pickup_longitude, loc[1], loc[0]) < range)
    idx1 = (haversine(data_train.dropoff_latitude, data_train.dropoff_longitude, loc[1], loc[0]) < range)
    fareAmount_Pickup = data_train[idx0].fare_amount
    fareAmount_DropOff = data_train[idx1].fare_amount
    distance_pickup = data_train[idx0].distance_km
    distance_dropoff = data_train[idx1].distance_km
    return idx0,idx1,fareAmount_Pickup,fareAmount_DropOff




idx0,idx1,fareAmount_Pickup,fareAmount_DropOff = absoluteDataPoint(jfkAirport_Corrd,"JFK Airport")




idx0,idx1,fareAmount_Pickup,fareAmount_DropOff = absoluteDataPoint(EWRAirport_Coord,"Newark Airport")




idx0,idx1,fareAmount_Pickup,fareAmount_DropOff = absoluteDataPoint(LGAAirport_Corrd, 'LaGuardia Airport')




data_train_ToAndFro_airport_JFK_Airport = data_train[(data_train.fare_amount == 57.33) | (data_train.fare_amount == 49.80) | (data_train.fare_amount == 49.57)]




data_train_ToAndFro_airport_JFK_Airport.head()






