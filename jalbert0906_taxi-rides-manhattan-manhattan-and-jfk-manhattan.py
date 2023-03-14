#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import math
from math import radians
import warnings
warnings.filterwarnings('ignore')

#Visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
sns.set_style("darkgrid")
import folium
import folium.plugins
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot




def readData(path, types, chunksize, chunks):

    df_list = []
    counter = 1
    
    for df_chunk in tqdm(pd.read_csv(path, usecols=list(types.keys()), dtype=types, chunksize=chunksize)):

        # The counter helps us stop whenever we want instead of reading the entire data
        if counter == chunks+1:
            break
        counter = counter+1

        # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
        # Using parse_dates would be much slower!
        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
        df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'])

        # Process the datetime and get hour of day and day of week
        # After Price Reform - Before Price Reform ('newRate')
        df_chunk['hour'] = df_chunk['pickup_datetime'].apply(lambda x: x.hour)
        df_chunk['weekday'] = df_chunk['pickup_datetime'].apply(lambda x: x.weekday())
        df_chunk['newRate'] = df_chunk['pickup_datetime'].apply(lambda x: True if x > pd.Timestamp(2012, 9, 30, 10) else False)
        df_chunk.drop(columns=['pickup_datetime'], inplace=True)   
        
        # Aappend the chunk to list
        df_list.append(df_chunk) 

    # Merge all dataframes into one dataframe
    df = pd.concat(df_list)

    # Delete the dataframe list to release memory
    del df_list
    
    return df




# The path where the Training set is
TRAIN_PATH = '../input/train.csv'

# The datatypes we want to pass the reading function
traintypes = {'fare_amount': 'float32',
             'pickup_datetime': 'str', 
             'pickup_longitude': 'float32',
             'pickup_latitude': 'float32',
             'dropoff_longitude': 'float32',
             'dropoff_latitude': 'float32',
             'passenger_count': 'float32'}

# The size of the chunk for each iteration
chunksizeTrain = 1_000_000

# The number of chunks we want to read
chunksnumberTrain = 5

df_train = readData(TRAIN_PATH, traintypes, chunksizeTrain, chunksnumberTrain)




# The path where the Test set is
TEST_PATH = '../input/test.csv'

# The datatypes we want to pass the reading function
testtypes = {'key': 'str',
             'pickup_datetime': 'str', 
             'pickup_longitude': 'float32',
             'pickup_latitude': 'float32',
             'dropoff_longitude': 'float32',
             'dropoff_latitude': 'float32',
             'passenger_count': 'float32'}

# The size of the chunk for each iteration
chunksizeTest = 1_000_000

# The number of chunks we want to read
chunksnumberTest = 1

df_test = readData(TEST_PATH, testtypes, chunksizeTest, chunksnumberTest)




df_train.head()




df_train.describe(include='all')




df_test.head()




df_test.describe(include='all')




# Function that cleans data
def cleanData(df, isTrain):

    # 1) Drop NaN
    df.dropna(how = 'any', axis = 'rows', inplace = True)
    
    # 2) 3) Drop fares below 2.5 USD or above 400 USD in case the dataset is the Training set
    if isTrain:
        df = df[df['fare_amount']>=2.5]
        df = df[df['fare_amount']<400]
    
    # 4) Drop passenger count below 1 or above 10
    df = df[(df['passenger_count']>=1) & (df['passenger_count']<10)] # Drop lines out of bound passenger count   
    
    # 5) Drop rides outside NYC
    minLon = -74.3
    maxLon = -73.7
    minLat = 40.5
    maxLat = 41

    df = df[df['pickup_latitude'] < maxLat]
    df = df[df['pickup_latitude'] > minLat]
    df = df[df['pickup_longitude'] < maxLon]
    df = df[df['pickup_longitude'] > minLon]

    df = df[df['dropoff_latitude'] < maxLat]
    df = df[df['dropoff_latitude'] > minLat]
    df = df[df['dropoff_longitude'] < maxLon]
    df = df[df['dropoff_longitude'] > minLon]

    # Reset Index
    df.reset_index(inplace=True, drop=True)
    
    return df

# Apply cleaning function to both datasets
df_train = cleanData(df_train, True)




df_train.describe(include='all')




trace = go.Pie(values = [df_train.shape[0],chunksizeTrain*chunksnumberTrain - df_train.shape[0]],
               labels = ["Useful data" , "Data loss due to missing values or other reasons"],
               marker = dict(colors = ['skyblue' ,'yellow'], line = dict(color = "black", width =  1.5)),
               rotation  = 60,
               hoverinfo = 'label+percent',
              )

layout = go.Layout(dict(title = 'Data Cleaning (percentage of data loss)',
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        showlegend=False
                       )
                  )

fig = go.Figure(data=[trace],layout=layout)
py.iplot(fig)
fig = go.Figure(data=[trace],layout=layout)




# Using datashader
#Import Libraries
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure, output_notebook, show
import datashader as ds
from datashader.bokeh_ext import InteractiveImage
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Hot, inferno, Elevation
from datashader import transfer_functions as tf
output_notebook()

# Define plotting function using Datashader
def plot_data_points(longitude,latitude,data_frame) :
    export  = partial(export_image, export_path="export", background="black")
    fig = figure(background_fill_color = "black")    
    cvs = ds.Canvas(plot_width=800, 
                    plot_height=600,
                    x_range=(-74.15,-73.75), 
                    y_range=(40.6,40.9))
    agg = cvs.points(data_frame,longitude,latitude)
    #img = tf.shade(agg, cmap=Hot, how='eq_hist')
    img = tf.shade(agg)   
    image_xpt = tf.dynspread(img, threshold=0.5, max_px=4)
    return export(image_xpt,'map')

# Call function and plot
plot_data_points('pickup_longitude', 'pickup_latitude', df_train)




# Let's look at some clusters with Folium (20000 points)
samples = df_train.sample(n=min(20000,df_train.shape[0]))
m = folium.Map(location=[np.mean(samples['pickup_latitude']), np.mean(samples['pickup_longitude'])], zoom_start=11)
FastMarkerCluster(data=list(zip(samples['pickup_latitude'], samples['pickup_longitude']))).add_to(m)
folium.LayerControl().add_to(m)
m




# Define how we want to split the data into sections based on 'fare_amount'
a = 40
b = 70
c = 400

# Plot normalized histogram for each section
plt.figure(figsize = (25,7))
plt.subplot(1,3,1)
plt.title('Below ' + str(a) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_train[df_train['fare_amount']<=a]['fare_amount'], norm_hist=True, bins=np.arange(0,a))
plt.subplot(1,3,2)
plt.title('From ' + str(a) + ' USD to ' + str(b) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_train[(df_train['fare_amount']>a)&(df_train['fare_amount']<=b)]['fare_amount'], norm_hist=True, bins=np.arange(a,b))
plt.subplot(1,3,3)
plt.title('From ' + str(b) + ' USD to ' + str(c) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_train[(df_train['fare_amount']>b)&(df_train['fare_amount']<=c)]['fare_amount'], norm_hist=True, bins=np.arange(b,c));




# Split df_train into a dataset of the rides before the fare rules change and after the fare rules change
df_before = df_train[df_train['newRate']==False]
df_after = df_train[df_train['newRate']==True]
print ('Number of data points from before fare rule change: ' + str(df_before.shape[0]))
print ('Number of data points from after fare rule change: ' + str(df_after.shape[0]))

# Plot the sections for the rides before fare rules change
plt.figure(figsize = (25,14))
plt.subplot(2,3,1)
plt.title('Old rate, below ' + str(a) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_before[df_before['fare_amount']<=a]['fare_amount'], norm_hist=True, bins=np.arange(0,a))
plt.subplot(2,3,2)
plt.title('Old rate, from ' + str(a) + ' USD to ' + str(b) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_before[(df_before['fare_amount']>a)&(df_before['fare_amount']<=b)]['fare_amount'], norm_hist=True, bins=np.arange(a,b))
plt.subplot(2,3,3)
plt.title('Old rate, from ' + str(b) + ' USD to ' + str(c) + ' USD',color = "b")
plt.ylabel('Normalized Density')
sns.distplot(df_before[(df_before['fare_amount']>b)&(df_before['fare_amount']<=c)]['fare_amount'], norm_hist=True, bins=np.arange(b,c)) 

# Plot the sections for the rides after fare rules change
plt.figure(figsize = (25,14))
plt.subplot(2,3,4)
plt.title('New rate, below ' + str(a) + ' USD',color = "g")
plt.ylabel('Normalized Density')
sns.distplot(df_after[df_after['fare_amount']<=a]['fare_amount'], norm_hist=True, color='green', bins=np.arange(0,a))
plt.subplot(2,3,5)
plt.title('New rate, from ' + str(a) + ' USD to ' + str(b) + ' USD',color = "g")
plt.ylabel('Normalized Density')
sns.distplot(df_after[(df_after['fare_amount']>a)&(df_after['fare_amount']<=b)]['fare_amount'], norm_hist=True, color='green', bins=np.arange(a,b))
plt.subplot(2,3,6)
plt.title('New rate, from ' + str(b) + ' USD to ' + str(c) + ' USD',color = "g")
plt.ylabel('Normalized Density')
sns.distplot(df_after[(df_after['fare_amount']>b)&(df_after['fare_amount']<=c)]['fare_amount'], norm_hist=True, color='green', bins=np.arange(b,c)); 




print('Mean fare BEFORE rate change: ' + str(np.around(df_before[df_before['fare_amount']<=a]['fare_amount'].mean(),2)))
print('Mean fare AFTER rate change: ' + str(np.around(df_after[df_after['fare_amount']<=a]['fare_amount'].mean(),2)))
print('Median fare BEFORE rate change: ' + str(np.around(df_before[df_before['fare_amount']<=a]['fare_amount'].median(),2)))
print('Median fare AFTER rate change: ' + str(np.around(df_after[df_after['fare_amount']<=a]['fare_amount'].median(),2)))




# Using mod function to extract the "cents value" of each ride
plt.figure(figsize = (20,7))
plt.subplot(1,2,1)
plt.title('Old Rate, below ' + str(a) + ' USD, "cents value"',color = "b")
sns.distplot(np.mod(df_before[df_before['fare_amount']<=a]['fare_amount'],1)) 
plt.subplot(1,2,2)
plt.title('New Rate, below ' + str(a) + ' USD, "cents value"',color = "g")
sns.distplot(np.mod(df_after[df_after['fare_amount']<=a]['fare_amount'],1), color='green'); 




from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Define polygon using coordinates (Just took them from Google Maps by clicking on the map)
lats_vect = [40.851638, 40.763022, 40.691262, 40.713380, 40.743944, 40.794344, 40.846332]
lons_vect = [-73.952423, -74.010418, -74.026685, -73.972200, -73.962051, -73.924073, -73.926454]
lons_lats_vect = np.column_stack((lons_vect, lats_vect))
polygon = Polygon(lons_lats_vect)

# Plot the polygon using Folium
man_map = folium.Map(location=[40.7631, -73.9712], zoom_start=12)
for i in range(0,6):
    folium.PolyLine(locations=[[lats_vect[i],lons_vect[i]], [lats_vect[i+1],lons_vect[i+1]]], color='blue').add_to(man_map)
folium.PolyLine(locations=[[lats_vect[6],lons_vect[6]], [lats_vect[0],lons_vect[0]]], color='blue').add_to(man_map)
man_map




# Check for every point on df_train if it belongs to polygon or not
manhattanRides = df_train[df_train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']]
                          .apply(lambda row: ((polygon.contains(Point(row['pickup_longitude'],row['pickup_latitude']))) &
                                              (polygon.contains(Point(row['dropoff_longitude'],row['dropoff_latitude'])))), axis=1)]

# Plot the remaining dataset 'manhattanRides'
plot_data_points('pickup_longitude', 'pickup_latitude', manhattanRides)




# Simple Euclidean Distance calculator 
def quickDist(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    R = 6371
    x = (lng2 - lng1) * np.cos(0.5*(lat2+lat1))
    y = lat2 - lat1
    d = R * np.sqrt(x*x + y*y)
    return d

# Longitude distance (use same Euclidean distance function with fixed latitude)
def latDist(lat1, lng1, lat2, lng2):
    uno = quickDist((lat1+lat2)/2, lng1, (lat1+lat2)/2, lng2)
    return uno

# Calculate real distance (Manhattan distance with 29 degrees to north)
def realManDist(lat1, lng1, lat2, lng2):
    flightDist = quickDist(lat1, lng1, lat2, lng2)
    latDistance = latDist(lat1, lng1, lat2, lng2)
    if flightDist == 0:
        ret = np.nan
    else:
        th = np.arccos(latDistance/flightDist)
        ata = flightDist*np.cos(th-0.506) + flightDist*np.sin(th-0.506)
        bta = flightDist*np.cos(th+0.506) + flightDist*np.sin(th+0.506)
        ret = max(ata,bta)
    return ret




# Calculate distance for every ride on manhattanRides
manhattanRides['distance'] = manhattanRides.apply(lambda row: realManDist(row['pickup_latitude'], 
                                                                          row['pickup_longitude'], 
                                                                          row['dropoff_latitude'], 
                                                                          row['dropoff_longitude']), axis=1)




print('Percentage of trips that happen inside Manhattan: ' + str(np.around(100*(manhattanRides.shape[0])/df_train.shape[0],2)))




# Split train/test
from sklearn.model_selection import train_test_split
manhattanRides_train, manhattanRides_test = train_test_split(manhattanRides, test_size=0.2, random_state=42)




# Plot the 'fare_amount' against the distance of the trip
plt.figure(figsize = (20,15))
plt.title('Manhattan Rides', color = "b")
plt.ylabel('Fare in USD')
plt.xlabel('Distance in Km')
plt.scatter(manhattanRides_train['distance'], manhattanRides_train['fare_amount'], alpha=0.5);




manhattanRides_shortDist = manhattanRides_train[manhattanRides_train['distance']<=0.3];
manhattanRides_highValue = manhattanRides_train[manhattanRides_train['fare_amount']>75];
manhattanRides_lowValue = manhattanRides_train[manhattanRides_train['fare_amount']<=3];

manhattanRides_train = manhattanRides_train[(manhattanRides_train['distance']>0.3)&(manhattanRides_train['fare_amount']<75)&(manhattanRides_train['fare_amount']>3)];

# Filter fare range
straight_lines = manhattanRides_train[(manhattanRides_train['fare_amount']>44)&(manhattanRides_train['fare_amount']<60)]
# Group by fare and count frequency
freq = straight_lines.groupby('fare_amount').count().sort_values('pickup_longitude', ascending=False)

# Keep the fare value of the top 8
fares_straight_lines = freq.index[0:7].values
# Extract from training dataframe
manhattanRides_straightLines = manhattanRides_train[manhattanRides_train['fare_amount'].isin(fares_straight_lines)]
manhattanRides_train = manhattanRides_train[~manhattanRides_train['fare_amount'].isin(fares_straight_lines)]
# Plot the 'fare_amount' against the distance of the trip
plt.figure(figsize = (20,15))
plt.title('Manhattan Rides', color = "b")
plt.ylabel('Fare in USD')
plt.xlabel('Distance in Km')
plt.scatter(manhattanRides_train['distance'], manhattanRides_train['fare_amount'], alpha=0.5);




# Fitting a linear model and measuring the MSE
import statsmodels.api as sm

# Define function that makes regression and returns params
def measureMSE(df):
    regression = sm.OLS(df['fare_amount'], sm.add_constant(df['distance'])).fit()
    farepred = regression.predict(sm.add_constant(df['distance'])) 
    mse = np.around(np.sqrt((((df['fare_amount']-farepred)**2).sum())/(df.shape[0])),4)
    return [regression.params[1], regression.params[0], mse]

# Apply function on manhattanRides_train
reg = measureMSE(manhattanRides_train)

print ('Slope: ' + str(np.around(reg[0],2)))
print ('Intercept: ' + str(np.around(reg[1],2)))
print ('MSE: ' + str(np.around(reg[2],4)))




# Split the train set into before the fare rule change and after the fare rule change
manhattanRidesBefore = manhattanRides_train[manhattanRides_train['newRate']==False]
manhattanRidesAfter = manhattanRides_train[manhattanRides_train['newRate']==True]

# Fitting a linear model and measuring the MSE
reg = measureMSE(manhattanRidesBefore)
print ('BEFORE - Slope: ' + str(np.around(reg[0],2)))
print ('BEFORE - Intercept: ' + str(np.around(reg[1],2)))
print ('BEFORE - MSE: ' + str(np.around(reg[2],4)))

reg = measureMSE(manhattanRidesAfter)
print ('AFTER - Slope: ' + str(np.around(reg[0],2)))
print ('AFTER - Intercept: ' + str(np.around(reg[1],2)))
print ('AFTER - MSE: ' + str(np.around(reg[2],4)))

# Plot the 'fare_amount' against the distance of the trip
plt.figure(figsize = (20,15))
plt.subplot(2,1,1)
plt.title('Manhattan Rides before fare change', color = "b")
plt.ylabel('Fare')
plt.xlabel('Distance in Km')
plt.scatter(manhattanRidesBefore['distance'], manhattanRidesBefore['fare_amount'], alpha=0.5)
plt.subplot(2,1,2)
plt.title('Manhattan Rides after fare change', color = "g")
plt.ylabel('Fare')
plt.xlabel('Distance in Km')
plt.scatter(manhattanRidesAfter['distance'], manhattanRidesAfter['fare_amount'], color='green', alpha=0.5);




series1 = manhattanRidesBefore[(manhattanRidesBefore['weekday'] == 3) & (manhattanRidesBefore['hour'] == 20)]
series2 = manhattanRidesAfter[(manhattanRidesAfter['weekday'] == 3) & (manhattanRidesAfter['hour'] == 20)]

# Fitting a linear model and measuring the MSE
reg = measureMSE(series1)
print ('BEFORE - Specifc daytime - Slope: ' + str(np.around(reg[0],2)))
print ('BEFORE - Specifc daytime - Intercept: ' + str(np.around(reg[1],2)))
print ('BEFORE - Specifc daytime - MSE: ' + str(np.around(reg[2],4)))

reg = measureMSE(series2)
print ('AFTER - Specifc daytime - Slope: ' + str(np.around(reg[0],2)))
print ('AFTER - Specifc daytime - Intercept: ' + str(np.around(reg[1],2)))
print ('AFTER - Specifc daytime - MSE: ' + str(np.around(reg[2],4)))

plt.figure(figsize = (20,15))
plt.subplot(2,1,1)
plt.title('Manhattan Rides before fare change, Tuesday at 14pm', color = "b")
plt.ylabel('Fare')
plt.xlabel('Distance in Km')
plt.scatter(series1['distance'], series1['fare_amount'], alpha=0.5)
plt.subplot(2,1,2)
plt.title('Manhattan Rides after fare change, Tuesday at 14pm', color = "g")
plt.ylabel('Fare')
plt.xlabel('Distance in Km')
plt.scatter(series2['distance'], series2['fare_amount'], color='green', alpha=0.5);




weekdaysOpt = [0,1,2,3,4,5,6]
hoursOpt = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

slopeMatBefore = np.zeros((7, 24))
interMatBefore = np.zeros((7, 24))
mseMatBefore = np.zeros((7, 24))

slopeMatAfter = np.zeros((7, 24))
interMatAfter = np.zeros((7, 24))
mseMatAfter = np.zeros((7, 24))

for i in weekdaysOpt:
    for j in hoursOpt:
        series1 = manhattanRidesBefore[(manhattanRidesBefore['weekday'] == i) & (manhattanRidesBefore['hour'] == j)]
        reg = measureMSE(series1)
        slopeMatBefore[i,j] = reg[0]
        interMatBefore[i,j] = reg[1]        
        mseMatBefore[i,j] = reg[2] 
        
        series2 = manhattanRidesAfter[(manhattanRidesAfter['weekday'] == i) & (manhattanRidesAfter['hour'] == j)]
        reg = measureMSE(series2)
        slopeMatAfter[i,j] = reg[0]
        interMatAfter[i,j] = reg[1]        
        mseMatAfter[i,j] = reg[2] 




fig = plt.figure(figsize=(30,13))
ax1 = fig.add_subplot(211)
ax1.set_title('MSE by day and hour: Before Fare Rule Change')
sns.heatmap(mseMatBefore, xticklabels = hoursOpt, yticklabels = weekdaysOpt, annot = True, ax=ax1, cmap='YlOrRd')
ax2 = fig.add_subplot(212)
ax2.set_title('MSE by day and hour: After Fare Rule Change')
sns.heatmap(mseMatAfter, xticklabels = hoursOpt, yticklabels = weekdaysOpt, annot = True, ax=ax2, cmap='YlOrRd');




def predictManhattan(hour, day, new, distance, slopeMatBefore, interMatBefore, slopeMatAfter, interMatAfter):
    
    if new:
        slope = slopeMatAfter[day, hour]
        inter = interMatAfter[day, hour]
    else:
        slope = slopeMatBefore[day, hour]
        inter = interMatBefore[day, hour]  
        
    fare = inter + slope*distance
    
    return fare




predictions = manhattanRides_train.apply(lambda row: predictManhattan(row['hour'], row['weekday'], row['newRate'], row['distance'], slopeMatBefore, interMatBefore, slopeMatAfter, interMatAfter), axis=1)
print('The in-sample Average MSE is: ' + str(np.around(np.sqrt((((manhattanRides_train['fare_amount']-predictions)**2).sum())/manhattanRides_train.shape[0]),3)))




predictions = manhattanRides_test.apply(lambda row: predictManhattan(row['hour'], row['weekday'], row['newRate'], row['distance'], slopeMatBefore, interMatBefore, slopeMatAfter, interMatAfter), axis=1)
print('The out-of-sample average MSE is: ' + str(np.around(np.sqrt((((manhattanRides_test['fare_amount']-predictions)**2).sum())/manhattanRides_test.shape[0]),3)))




def addMarkerPick(df, color, m1, m2):
    samples = df.sample(n=min(300,df.shape[0]))
    for lt, ln in zip(samples['pickup_latitude'], samples['pickup_longitude']):
            folium.Circle(location = [lt,ln] ,radius = 2, color = color).add_to(m1)
    for lt, ln in zip(samples['dropoff_latitude'], samples['dropoff_longitude']):
            folium.Circle(location = [lt,ln] ,radius = 2, color = color).add_to(m2)
        
m1 = folium.Map(location=[40.7631, -73.9712], zoom_start=13)
m2 = folium.Map(location=[40.7631, -73.9712], zoom_start=13)

addMarkerPick(manhattanRides_shortDist, 'blue', m1, m2)
addMarkerPick(manhattanRides_highValue, 'red', m1, m2)
addMarkerPick(manhattanRides_lowValue, 'green', m1, m2)
addMarkerPick(manhattanRides_straightLines, 'yellow', m1, m2)




m1




m2




# One end has to be JFK
jfk_lat_min = 40.626777
jfk_lat_max = 40.665599
jfk_lon_min = -73.823964
jfk_lon_max = -73.743085

# Filter trips originating on JFK
df_fromJFK = df_train[(df_train['pickup_latitude']<jfk_lat_max)&
                      (df_train['pickup_latitude']>jfk_lat_min)&
                      (df_train['pickup_longitude']<jfk_lon_max)&
                      (df_train['pickup_longitude']>jfk_lon_min)]

# Filter trips ending on JFK
df_toJFK = df_train[(df_train['dropoff_latitude']<jfk_lat_max)&
                    (df_train['dropoff_latitude']>jfk_lat_min)&
                    (df_train['dropoff_longitude']<jfk_lon_max)&
                    (df_train['dropoff_longitude']>jfk_lon_min)]




# The other end has to be Manhattan
df_fromJFK = df_fromJFK[df_fromJFK[['dropoff_latitude', 'dropoff_longitude']].apply(lambda row: polygon.contains(Point(row['dropoff_longitude'],row['dropoff_latitude'])), axis=1)]
df_toJFK = df_toJFK[df_toJFK[['pickup_latitude', 'pickup_longitude']].apply(lambda row: polygon.contains(Point(row['pickup_longitude'],row['pickup_latitude'])), axis=1)]




print('Percentage of trips between JFK and Manhattan: ' + str(np.around(100*(df_fromJFK.shape[0] + df_toJFK.shape[0])/df_train.shape[0],2)))




m1 = folium.Map(location=[40.645580, -73.785115], zoom_start=16)
samples = df_fromJFK.sample(n=min(500,df_fromJFK.shape[0]))
for lt, ln in zip(samples['pickup_latitude'], samples['pickup_longitude']):
            folium.Circle(location = [lt,ln] ,radius = 2, color = 'blue').add_to(m1)
            
samples = df_toJFK.sample(n=min(500,df_toJFK.shape[0]))
for lt, ln in zip(samples['dropoff_latitude'], samples['dropoff_longitude']):
            folium.Circle(location = [lt,ln] ,radius = 2, color = 'red').add_to(m1)
        
m1




# Plot a histogram of the fares
plt.figure(figsize = (25,10))
plt.subplot(2,2,1)
plt.title('Old rate, from JFK to Manhattan',color = "b")
sns.distplot(df_fromJFK[df_fromJFK['newRate']==False]['fare_amount'], norm_hist=True, bins=np.arange(a,b))
plt.subplot(2,2,2)
plt.title('Old rate, from Manhattan to JFK',color = "b")
sns.distplot(df_toJFK[df_toJFK['newRate']==False]['fare_amount'], norm_hist=True, bins=np.arange(a,b))
plt.subplot(2,2,3)
plt.title('New rate, from JFK to Manhattan',color = "b")
sns.distplot(df_fromJFK[df_fromJFK['newRate']==True]['fare_amount'], norm_hist=True, bins=np.arange(a,b), color='green')
plt.subplot(2,2,4)
plt.title('New rate, from Manhattan to JFK',color = "b")
sns.distplot(df_toJFK[df_toJFK['newRate']==True]['fare_amount'], norm_hist=True, bins=np.arange(a,b), color='green');




# Add a weekend column
df_fromJFK['weekend'] = df_fromJFK['weekday'].isin([5,6])==True
df_toJFK['weekend'] = df_toJFK['weekday'].isin([5,6])==True




def plotBars(df, newRate, weekend):
   frfr = df[(df['newRate']==newRate)&(df['weekend']==weekend)]
   sns.barplot(x='hour',y='fare_amount', data = frfr, edgecolor=".1", errcolor = 'red')
      
# Plot every combination
plt.figure(figsize = (23,25))
plt.subplot(4,2,1)
plt.title('Old rate, from JFK to Manhattan, not weekend',color = "b")
plotBars(df_fromJFK, False, False)
plt.subplot(4,2,2)
plt.title('Old rate, from JFK to Manhattan, weekend',color = "b")
plotBars(df_fromJFK, False, True)
plt.subplot(4,2,3)
plt.title('New rate, from JFK to Manhattan, not weekend',color = "b")
plotBars(df_fromJFK, True, False)
plt.subplot(4,2,4)
plt.title('New rate, from JFK to Manhattan, weekend',color = "b")
plotBars(df_fromJFK, True, True)
plt.subplot(4,2,5)
plt.title('Old rate, from Manhattan to JFK, not weekend',color = "b")
plotBars(df_toJFK, False, False)
plt.subplot(4,2,6)
plt.title('Old rate, from Manhattan to JFK, weekend',color = "b")
plotBars(df_toJFK, False, True)
plt.subplot(4,2,7)
plt.title('New rate, from Manhattan to JFK, not weekend',color = "b")
plotBars(df_toJFK, True, False)
plt.subplot(4,2,8)
plt.title('New rate, from Manhattan to JFK, weekend',color = "b")
plotBars(df_toJFK, True, True)




# Define options
weekendOpt = [0, 1]
hoursOpt = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# Initialize matrices
oldFromJFKtoMAN = np.zeros((2, 24))
newFromJFKtoMAN = np.zeros((2, 24))
oldFromMANtoJFK = np.zeros((2, 24))
newFromMANtoJFK = np.zeros((2, 24))

# Run combiantions
for i in weekendOpt:
    for j in hoursOpt:
        
        mean = df_fromJFK[(df_fromJFK['newRate']==False)&(df_fromJFK['weekend']==i)&(df_fromJFK['hour']==j)]['fare_amount'].mean()
        oldFromJFKtoMAN[i,j] = mean
        
        mean = df_fromJFK[(df_fromJFK['newRate']==True)&(df_fromJFK['weekend']==i)&(df_fromJFK['hour']==j)]['fare_amount'].mean()
        newFromJFKtoMAN[i,j] = mean
        
        mean = df_toJFK[(df_toJFK['newRate']==False)&(df_toJFK['weekend']==i)&(df_toJFK['hour']==j)]['fare_amount'].mean()
        oldFromMANtoJFK[i,j] = mean
        
        mean = df_toJFK[(df_toJFK['newRate']==True)&(df_toJFK['weekend']==i)&(df_toJFK['hour']==j)]['fare_amount'].mean()
        newFromMANtoJFK[i,j] = mean




# This are the prediction matrices
oldFromJFKtoMAN;
newFromJFKtoMAN;
oldFromMANtoJFK;
newFromMANtoJFK;




# Prediction function
def predictAirport(isfrom, new, weekend, hour, oldFromJFKtoMAN, newFromJFKtoMAN, oldFromMANtoJFK, newFromMANtoJFK):
    if isfrom:
        if new:
            fare = newFromJFKtoMAN[weekend, hour]
        else:
            fare = oldFromJFKtoMAN[weekend, hour]
    else:
        if new:
            fare = newFromMANtoJFK[weekend, hour]
        else:
            fare = oldFromMANtoJFK[weekend, hour]            
    return fare




# From JFK
predictions = df_fromJFK.apply(lambda row: predictAirport(True, row['newRate'], row['weekend']*1, row['hour'], oldFromJFKtoMAN, newFromJFKtoMAN, oldFromMANtoJFK, newFromMANtoJFK), axis=1)
print('From JFK to Manhattan the MSE is: ' + str(np.around(np.sqrt((((df_fromJFK['fare_amount']-predictions)**2).sum())/df_fromJFK.shape[0]),3)))

# To JFK
predictions = df_toJFK.apply(lambda row: predictAirport(False, row['newRate'], row['weekend']*1, row['hour'], oldFromJFKtoMAN, newFromJFKtoMAN, oldFromMANtoJFK, newFromMANtoJFK), axis=1)
print('From Manhattan to JFK the MSE is: ' + str(np.around(np.sqrt((((df_toJFK['fare_amount']-predictions)**2).sum())/df_toJFK.shape[0]),3)))

