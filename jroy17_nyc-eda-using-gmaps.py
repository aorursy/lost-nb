#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import codecs
import calendar
import datetime
from time import *
import seaborn as sns
import matplotlib.animation as animation
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from IPython.display import Image
from geopy.distance import vincenty
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor


# In[2]:


def distance(source_lat, source_long, dest_lat, dest_long):
    #source = [source_lat, source_long]
    #dest = [dest_lang, dest_long]
    #dist = vincenty(source,dest).miles
    radius = 6371 # km
    dlat = math.radians(dest_lat-source_lat)
    dlon = math.radians(dest_long-source_long)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(source_lat))         * math.cos(math.radians(dest_lat)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist = radius * c

    return dist


# In[3]:


direc = "../input/nyc-taxi-trip-duration/"


# In[4]:


train = pd.read_csv(direc+"train.csv")
test = pd.read_csv(direc+"test.csv")


# In[5]:


dist = []
for i in range(len(train)):
    dist.append(distance(train.pickup_latitude[i],train.pickup_longitude[i],
                                    train.dropoff_latitude[i],train.dropoff_longitude[i]))

train["distance"] = dist


# In[6]:


dist_test = []
for i in range(len(test)):
    dist_test.append(distance(test.pickup_latitude[i],test.pickup_longitude[i],
                                    test.dropoff_latitude[i],test.dropoff_longitude[i]))

test["distance"] = dist_test


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


unique=test.id.unique()


# In[10]:


len(unique)


# In[11]:


pickup_date = []
pickup_time = []
pickup_day = []
pickup_hr = []
pickup_month = []


# In[12]:


pickup_datetime = train["pickup_datetime"]

for i in range(len(train)):
    split_row = pickup_datetime[i].split()

    # date
    pickup_date.append(split_row[0])
    # time
    pickup_time.append(split_row[1])
    # day of week
    pickup_day.append(datetime.datetime.strptime(pickup_date[i], "%Y-%m-%d").strftime("%A"))
    # month of the year
    split_dt = pickup_date[i].split('-')
    pickup_month.append(calendar.month_name[int(split_dt[1])])
    # hour of day
    split_hr = pickup_time[i].split(':')
    if split_row[0] == '00':
        pickup_hr.append(24)
    else:
        pickup_hr.append(int(split_hr[0]))


# In[13]:


# Adding the columns to train dataset
train["pickup_hr"] = pickup_hr
train["pickup_day"] = pickup_day
train["pickup_month"] = pickup_month


# In[14]:


# Plots a heatmap showing the maximum duration of trips grouped by days of the week
day = train.loc[:,["pickup_day","trip_duration"]]
day_grp = day.groupby('pickup_day').sum()
sns.heatmap(day_grp, annot=True,cmap="YlOrRd")
plt.show()


# In[15]:


sunday = train.query("pickup_day == 'Sunday'")
monday = train.query("pickup_day == 'Monday'")
tuesday = train.query("pickup_day == 'Tuesday'")
wednesday = train.query("pickup_day == 'Wednesday'")
thursday = train.query("pickup_day == 'Thursday'")
friday = train.query("pickup_day == 'Friday'")
saturday = train.query("pickup_day == 'Saturday'")


# In[16]:


plt.figure(figsize=(12,8))
ax = sns.kdeplot(sunday.pickup_hr,label="Sunday")
ax = sns.kdeplot(monday.pickup_hr,label="Monday")
ax = sns.kdeplot(tuesday.pickup_hr,label="Tuesday")
ax = sns.kdeplot(wednesday.pickup_hr,label="Wednesday")
ax = sns.kdeplot(thursday.pickup_hr,label="Thursday")
ax = sns.kdeplot(friday.pickup_hr,label="Friday")
ax = sns.kdeplot(saturday.pickup_hr,label="Saturday")
ax.set_xlabel("Hours of the day")
ax.set_ylabel("Frequency of pickups")
plt.show()


# In[17]:


train["pickup_hr"] =  train["pickup_hr"].convert_objects(convert_numeric=True)


# In[18]:


plt.figure(figsize=(12,8))
axe = sns.violinplot(x="pickup_day", y="pickup_hr", hue="vendor_id", data=train, split=True,palette='Set2',scale="count")
plt.show()


# In[19]:


# Creating a pivot of train to create heat map between Pickup Months and Pickup Days
pvt = train.pivot_table(values='pickup_hr',index='pickup_month',columns='pickup_day')
# Re-ordering columns
pvt = pvt[["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]

# Re-ordering indexes
pvt = pvt.reindex(["January","February","March","April","May"])
sns.heatmap(pvt,cmap='YlOrRd')
plt.show()


# In[20]:


train["duration_hrs"] = train["trip_duration"]/3600


# In[21]:


grp_sum = train.groupby(['pickup_day','pickup_month'])['duration_hrs'].transform(max) == train['duration_hrs']
# Creating a pivot of train to create heat map between Pickup Months and Pickup Days
pvt = train[grp_sum].pivot_table(values='duration_hrs',index='pickup_month',columns='pickup_day')
# Re-ordering columns
pvt = pvt[["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]

# Re-ordering indexes
pvt = pvt.reindex(["January","February","March","April","May"])
sns.heatmap(pvt,cmap='YlOrRd')
plt.show()


# In[22]:


sns.set(style="whitegrid", color_codes=True)

fig = plt.figure(figsize=(15,15))
# Frequency distribution of passengers
plt.subplot(221)

ax = sns.countplot(x="passenger_count", data=train,palette=sns.cubehelix_palette(8))
plt.ylabel('Frequency')
plt.xlabel('Number of Passengers')
plt.title('Frequency distribution of passengers')

# Frequency distribution of days
plt.subplot(222)
ax = sns.countplot(x="pickup_day", data=train,palette="GnBu_d")
plt.ylabel('Frequency')
plt.xlabel('Days')
plt.title('Frequency distribution of days')

# Frequency distribution of months
plt.subplot(223)
ax = sns.countplot(x="pickup_month", data=train,palette="BuGn_r")
plt.ylabel('Frequency')
plt.xlabel('Months')
plt.title('Frequency distribution of months')

# Frequency distribution of hours
plt.subplot(224)
ax = sns.countplot(x="pickup_hr", data=train,palette="YlOrRd")
plt.ylabel('Frequency')
plt.xlabel('Hours')
plt.title('Frequency distribution of hours')


# In[23]:


plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.show()


# In[24]:


long_dist = []
for i in range(len(train)):
    if train.duration_hrs[i] > 24:
        long_dist.append(i)


# In[25]:


def airportTrips(lat1,lat2, long1,long2):
    PU = []
    DO = []

    for i in range(len(train)):
        if(lat1 <= train.pickup_latitude[i] <= lat2 and long1 <= train.pickup_longitude[i] <= long2): 
            PU.append(i)
        if(lat1 <= train.dropoff_latitude[i] <= lat2 and long1 <= train.dropoff_longitude[i] <= long2):
            DO.append(i)
            
    pickup = []
    dropoff = []
    for i in range(len(PU)):
        pickup.append(train.loc[PU[i],["pickup_latitude","pickup_longitude"]])
    for i in range(len(DO)):
        dropoff.append(train.loc[DO[i],["dropoff_latitude","dropoff_longitude"]] )
    PU = pd.DataFrame(pickup)
    DO = pd.DataFrame(dropoff)
    
    heatlayer1 = gmaps.heatmap_layer(PU)
    heatlayer2 = gmaps.heatmap_layer(DO)
    return heatlayer1,heatlayer1
    


# In[26]:


weather = pd.read_csv("../input/nycweather/nyc_Jan_Jun_2016_weat.csv")


# In[27]:


weather.head(10)


# In[28]:


weather.loc[weather.SNOW == -9999, 'SNOW'] = NaN
weather.loc[weather.PRCP == -9999.0, 'PRCP'] = NaN
weather.loc[weather.SNWD == -9999.0, 'SNWD'] = NaN
weather.loc[weather.TAVG == -9999, 'TAVG'] = NaN
weather.loc[weather.TMAX == -9999, 'TMAX'] = NaN
weather.loc[weather.TMIN == -9999, 'TMIN'] = NaN


# In[29]:


weather.head()


# In[30]:


pickup_date = weather["DATE"].apply(str)
weather_date = []

for i in range(len(weather)):
    
    # day of week
    weather_date.append(datetime.datetime.strptime(pickup_date[i], "%Y%m%d").strftime("%A"))
    
weather["DAY"] = weather_date


# In[31]:


weather.dtypes


# In[32]:


meanSnow = weather.groupby('DAY')['SNOW'].transform('mean')
meanPrec = weather.groupby('DAY')['PRCP'].transform('mean')
meanSnwd = weather.groupby('DAY')['SNWD'].transform('mean')
meanTavg = weather.groupby('DAY')['TAVG'].transform('mean').astype(int)
meanTmin = weather.groupby('DAY')['TMIN'].transform('mean').astype(int)
meanTmax = weather.groupby('DAY')['TMAX'].transform('mean').astype(int)


# In[33]:


weather["SNOW"] = weather["SNOW"].fillna(meanSnow)
weather["PRCP"] = weather["PRCP"].fillna(meanPrec)
weather["SNWD"] = weather["SNWD"].fillna(meanSnwd)
weather["TAVG"] = weather["TAVG"].fillna(meanTavg)
weather["TMIN"] = weather["TMIN"].fillna(meanTmin)
weather["TMAX"] = weather["TMAX"].fillna(meanTmax)


# In[34]:


# Extracting month from Date
month_weather = []
for i in range(len(weather)):
    date = weather["DATE"][i]/100
    month = int(date%100)
    month_weather.append(calendar.month_name[month])
    
weather["MONTH"] = month_weather


# In[35]:


# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(weather,col="MONTH", hue="MONTH", col_wrap=4, size=3)

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "DATE", "PRCP", marker="o")

# Adjust the tick positions and labels
grid.set(yticks=[0, 35])

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
plt.show()


# In[36]:


# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(weather,col="MONTH", hue="MONTH", col_wrap=4, size=3)

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "DATE", "SNWD", marker="o")

# Adjust the tick positions and labels
grid.set(yticks=[0, 35])

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
plt.show()


# In[37]:


from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error

X = train.loc[:,["distance"]].values
y = train.duration_hrs

err = []
kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=0)


# In[38]:


kf


# In[39]:


for train_index, test_index in kf:
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    xtest = X[test_index]
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    err.append(mean_squared_error(actuals, predictions))
err


# In[40]:


sns.set(style="whitegrid", color_codes=True)

fig = plt.figure(figsize=(12,8))
# Frequency distribution of passengers
plt.subplot(121)

plot(xtest,predictions,color='red',marker='o',label="predicted")
plt.xlabel("Distance in km")
plt.ylabel("Time in hrs")
plt.legend()


plt.subplot(122)

plot(xtest,actuals,color='green',marker='o',label="actuals")
plt.xlabel("Distance in km")
plt.ylabel("Time in hrs")
plt.legend()
plt.ylim(-40,950)
plt.show()

