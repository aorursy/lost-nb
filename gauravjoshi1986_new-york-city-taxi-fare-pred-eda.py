#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load some default Python modules
import numpy as np
import pandas as pd
from datetime import date, datetime
from haversine import haversine
# packages for mapping
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# data path\nTRAIN_PATH = '../input/train.csv'\n\n# Set columns to most suitable type to optimize for memory usage\ndatatypes = {'fare_amount': 'float32',\n              'pickup_datetime': 'str', \n              'pickup_longitude': 'float32',\n              'pickup_latitude': 'float32',\n              'dropoff_longitude': 'float32',\n              'dropoff_latitude': 'float32',\n              'passenger_count': 'uint8'}\n\ncols = list(datatypes.keys())\n\n# read data in pandas dataframe\ndf = pd.read_csv(TRAIN_PATH, usecols=cols, dtype=datatypes, nrows = 2000000)\n\ndf['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)\ndf['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')")


# In[ ]:


# list first few rows (datapoints)
df.tail()


# In[ ]:


# there are negative values in fare amount which we need to remove
print('Old size: %d' % len(df))
df = df[df.fare_amount>=0]
print('New size: %d' % len(df))


# In[ ]:


# plot histogram of fare - 
df[df.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');


# In[ ]:


# check if we have missing entries in dataframe
df.isnull().sum()


# In[ ]:


# drop observation with missing entries
print('Old size: %d' % len(df))
df = df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df))


# In[ ]:


# plot histogram of pessanger - 
df[df.passenger_count<10].passenger_count.hist(bins=10, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');


# In[ ]:


def distance(lat1, lon1, lat2, lon2):
    """
    calculates the Manhattan distance between 2 points using their coordinates   
    Returns
    -------
    d: float
        The Manhattan distance between the two points in kilometers
    """
    d = haversine((lat1, lon1), (lat2, lon1)) + haversine((lat2, lon1), (lat2, lon2))
    return d

# The distance is calculated in kilometers
df["distance"] = df.apply(lambda row: distance(row["pickup_latitude"], 
                                               row["pickup_longitude"], 
                                               row["dropoff_latitude"], 
                                               row["dropoff_longitude"]), axis=1)

# date time features
df["pickup_month"] = df["pickup_datetime"].apply(lambda x: x.month)
df["pickup_day"] = df["pickup_datetime"].apply(lambda x: x.day)
df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: x.weekday())
df["pickup_hour"] = df["pickup_datetime"].apply(lambda x: x.hour)
df["pickup_minute"] = df["pickup_datetime"].apply(lambda x: x.minute)
df["pickup_time"] = df["pickup_hour"] + (df["pickup_minute"] / 60)


# In[ ]:


def plot_bar(df, col):
    plt.figure(figsize=(12,8))
    sns.countplot(x=col, data=df)
    plt.show()


# In[ ]:


plot_bar(df, "pickup_hour")


# In[ ]:


plot_bar(df, "pickup_month")


# In[ ]:


def plot_line(df, col1, col2):
    df_agg = df.groupby(col1)[col2].aggregate(np.median).reset_index()

    plt.figure(figsize=(12,8))
    sns.pointplot(df_agg[col1], df_agg[col2])
    plt.show()


# In[ ]:


plot_line(df, 'pickup_hour', 'distance')


# In[ ]:


plot_line(df, 'pickup_weekday', 'distance')


# In[ ]:


df = df[(df['pickup_longitude'] >= -90) & (df['pickup_longitude'] <= 90)]
df = df[(df['pickup_latitude'] >= -90) & (df['pickup_latitude'] <= 90)]

df = df[(df['dropoff_longitude'] >= -90) & (df['dropoff_longitude'] <= 90)]
df = df[(df['dropoff_latitude'] >= -90) & (df['dropoff_latitude'] <= 90)]


# In[ ]:


plt.figure(figsize=(20,20))

# Set the limits of the map to the minimum and maximum coordinates
#lon_min = min(df.pickup_longitude.min(), df.dropoff_longitude.min()) - .2
#lon_max = max(df.pickup_longitude.max(), df.dropoff_longitude.max()) + .2
#lat_min = min(df.pickup_latitude.min(), df.dropoff_latitude.min()) - .2
#lat_max = max(df.pickup_latitude.max(), df.dropoff_latitude.max()) + .2

lon_min = -74.05
lon_max = -73.75
lat_min = 40.6
lat_max = 40.9


print ("min lon {} max lon {}".format(lon_min, lon_max))
print ("min lat {} max lat {}".format(lat_min, lat_max))

# Set the center of the map
cent_lat = (lat_min + lat_max) / 2
cent_lon = (lon_min + lon_max) / 2

map = Basemap(projection='tmerc', resolution='l', 
              llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max, 
              lat_0 = cent_lat, lon_0 = cent_lon)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgray', lake_color='aqua')
map.drawcountries(linewidth=2)
map.drawstates(color='b')

long = np.array(df["pickup_longitude"])
lat = np.array(df["pickup_latitude"])

x, y = map(long, lat)
map.plot(x, y, 'ro', markersize=3, alpha=1)

plt.show()

