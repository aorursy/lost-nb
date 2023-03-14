#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
import json, requests
from os import environ
get_ipython().run_line_magic('matplotlib', 'inline')
import geocoder




df = pd.read_csv('../input/train.csv')
df.head()




df.dtypes




df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])




df['store_and_fwd_flag'].unique()




df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y':1,'N':0})




len(df['id'].unique()) 




len(df['id'].unique()) == df.shape[0]




df['id'].str.startswith('id').sum() # every one starts with 'id'




df['id'] = df['id'].apply(lambda x: int(x[2:]))




df['id'].count() == len(df['id'].unique())




df['id'].min()




df['id'].max()




df['vendor_id'].unique()




df['vendor_id'][df['vendor_id'] == 1].count() # How many vendor 1?




df['vendor_id'].count() - df['vendor_id'][df['vendor_id'] == 1].count() # How many vendor 2?




sns.barplot(x = [1,2],y = [678342,780302])




sns.boxplot(df['passenger_count'])




from collections import Counter




passenger_counts = Counter(df['passenger_count'])
passenger_counts




pd.DataFrame({'Count':passenger_counts}).plot(kind='bar',title='Number of passengers count frequency')




df['trip_duration_delta'] = df['dropoff_datetime'] - df['pickup_datetime']




trip_delta = df['trip_duration_delta']




trip_delta.sort_values().head(20)




trip_delta.sort_values().tail()




trip_delta[trip_delta > '1 days 00:00:00']




df.iloc[355003]




df.iloc[trip_delta[trip_delta > '1 days 00:00:00'].index.values]




delta_seconds = df['trip_duration_delta'].dt.total_seconds()




delta_seconds.head()




df['trip_duration'].head()




df[delta_seconds != df['trip_duration']]




delta_seconds = df['trip_duration_delta'].dt.total_seconds().apply(int) # Round the floats to ints




df[delta_seconds != df['trip_duration']]




df.loc[0,['pickup_latitude','pickup_longitude']].values # First pickup for testing




g = geocoder.google([40.767936706542969, -73.982154846191392],method='reverse')
g.address




pulat,pulng,dolat,dolng = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]




print(pulat,pulng,dolat,dolng)




google = "http://maps.googleapis.com/maps/api/distancematrix/json?"
geo = "origins={a},{b}&destinations={c},{d}"
reply_type = "&mode=driving&language=en-EN&sensor=false"




# The geo piece requires the coordinates for pickup and destination

q = google+geo.format(a=pulat,b=pulng,c=dolat,d=dolng)+reply_type




result= json.loads(requests.get(q).text)
result




result['rows'][0]['elements'][0]['duration']['value']




df.loc[0,'trip_duration']




def get_google_estimate_now(pulat,pulng,dolat,dolng):
    google = "http://maps.googleapis.com/maps/api/distancematrix/json?"
    geo = "origins={a},{b}&destinations={c},{d}"
    reply_type = "&mode=driving&language=en-EN&sensor=false"
    q = google+geo.format(a=pulat,b=pulng,c=dolat,d=dolng)+reply_type
    result= json.loads(requests.get(q).text)
    return result 




x = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]




get_google_estimate_now(*x)




small_df = df.loc[0:3,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]




for vals in small_df.values:
    print(get_google_estimate_now(*vals))




df.loc[0:3,['trip_duration']]




google_estimate = []
df_coordinates = df.loc[:1000,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]

for vals in df_coordinates.values:
    google_estimate.append(get_google_estimate_now(*vals))
    print(google_estimate[-1:])




len(google_estimate)




google_estimate[1]




estimates = [est['rows'][0]['elements'][0]['duration']['value'] for est in google_estimate]




estimates[:10]




df['trip_duration'][:10]




first_thousand = pd.DataFrame({'trip_time':df['trip_duration'][:len(estimates)],'est_time':pd.Series(estimates)})




first_thousand.head()




first_thousand.plot(x = 'trip_time',
                    y= 'est_time',
                    kind='scatter',
                    title='correlation between trip time and google estimate')




df['trip_duration'][:len(estimates)][df['trip_duration'][:len(estimates)]> 80000]




first_thousand.iloc[531]




first_thousand.drop(531).plot(x = 'est_time',
                              y='trip_time',
                              kind='scatter',
                              title = 'correlation between trip time and google estimate')




import time
import datetime

# google maps API will take a departure time in unix time format
# unix datetime is seconds since 1st Jan 1970

def dt2ut(dt): 
    
    epoch = pd.to_datetime('1970-01-01')
    
    return (dt - epoch).total_seconds()

def format_query(pulat,pulng,dolat,dolng,unixtime,api_key):
    
    google = "http://maps.googleapis.com/maps/api/distancematrix/json?"
    geo = "origins={a},{b}&destinations={c},{d}"
    time = "&departure_time={e}"
    reply_type = "&mode=driving&language=en-EN&sensor=false"
    key = 'key={f}'
    q = google+geo.format(a=pulat,b=pulng,c=dolat,d=dolng)+    time.format(e=int(unixtime))+reply_type+key.format(f=api_key)
    
    return q

def get_google_estimate_future(pulat,pulng,dolat,dolng,deptime,api_key):
    
    unixtime = dt2ut(deptime)
    variables = [pulat,pulng,dolat,dolng,unixtime,api_key] 
    q = format_query(*variables)
    result= json.loads(requests.get(q).text)
    driving_results = result
    
    return driving_results




test_dt = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'pickup_datetime']]




get_google_estimate_future(*test_dt, api_key=environ["GOOGLE_API_KEY"])




pd.DatetimeIndex




df['day_of_week'] = df['pickup_datetime'].dt.dayofweek




pd.DataFrame(list(Counter(df['day_of_week']).values()),
             index=['Mon','Tue','Wed','Thur','Fri','Sat','Sun']).plot(title='Number of trips by weekday',
                                                                      figsize=(10,8))




ax = df['trip_duration'].groupby(df['day_of_week']).median().plot(title='Median length of trip by weekday',
                                                                  figsize=(10,8))
ax.set_xticklabels(['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])

