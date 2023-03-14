#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import json, requests
from os import environ
get_ipython().run_line_magic('matplotlib', 'inline')
import geocoder


# In[2]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[3]:


df.dtypes


# In[4]:


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])


# In[5]:


df['store_and_fwd_flag'].unique()


# In[6]:


df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y':1,'N':0})


# In[7]:


len(df['id'].unique()) 


# In[8]:


len(df['id'].unique()) == df.shape[0]


# In[9]:


df['id'].str.startswith('id').sum() # every one starts with 'id'


# In[10]:


df['id'] = df['id'].apply(lambda x: int(x[2:]))


# In[11]:


df['id'].count() == len(df['id'].unique())


# In[12]:


df['id'].min()


# In[13]:


df['id'].max()


# In[14]:


df['vendor_id'].unique()


# In[15]:


df['vendor_id'][df['vendor_id'] == 1].count() # How many vendor 1?


# In[16]:


df['vendor_id'].count() - df['vendor_id'][df['vendor_id'] == 1].count() # How many vendor 2?


# In[17]:


sns.barplot(x = [1,2],y = [678342,780302])


# In[18]:


sns.boxplot(df['passenger_count'])


# In[19]:


from collections import Counter


# In[20]:


passenger_counts = Counter(df['passenger_count'])
passenger_counts


# In[21]:


pd.DataFrame({'Count':passenger_counts}).plot(kind='bar',title='Number of passengers count frequency')


# In[22]:


df['trip_duration_delta'] = df['dropoff_datetime'] - df['pickup_datetime']


# In[23]:


trip_delta = df['trip_duration_delta']


# In[24]:


trip_delta.sort_values().head(20)


# In[25]:


trip_delta.sort_values().tail()


# In[26]:


trip_delta[trip_delta > '1 days 00:00:00']


# In[27]:


df.iloc[355003]


# In[28]:


df.iloc[trip_delta[trip_delta > '1 days 00:00:00'].index.values]


# In[29]:


delta_seconds = df['trip_duration_delta'].dt.total_seconds()


# In[30]:


delta_seconds.head()


# In[31]:


df['trip_duration'].head()


# In[32]:


df[delta_seconds != df['trip_duration']]


# In[33]:


delta_seconds = df['trip_duration_delta'].dt.total_seconds().apply(int) # Round the floats to ints


# In[34]:


df[delta_seconds != df['trip_duration']]


# In[35]:


df.loc[0,['pickup_latitude','pickup_longitude']].values # First pickup for testing


# In[36]:


g = geocoder.google([40.767936706542969, -73.982154846191392],method='reverse')
g.address


# In[37]:


pulat,pulng,dolat,dolng = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]


# In[38]:


print(pulat,pulng,dolat,dolng)


# In[39]:


google = "http://maps.googleapis.com/maps/api/distancematrix/json?"
geo = "origins={a},{b}&destinations={c},{d}"
reply_type = "&mode=driving&language=en-EN&sensor=false"


# In[40]:


# The geo piece requires the coordinates for pickup and destination

q = google+geo.format(a=pulat,b=pulng,c=dolat,d=dolng)+reply_type


# In[41]:


result= json.loads(requests.get(q).text)
result


# In[42]:


result['rows'][0]['elements'][0]['duration']['value']


# In[43]:


df.loc[0,'trip_duration']


# In[44]:


def get_google_estimate_now(pulat,pulng,dolat,dolng):
    google = "http://maps.googleapis.com/maps/api/distancematrix/json?"
    geo = "origins={a},{b}&destinations={c},{d}"
    reply_type = "&mode=driving&language=en-EN&sensor=false"
    q = google+geo.format(a=pulat,b=pulng,c=dolat,d=dolng)+reply_type
    result= json.loads(requests.get(q).text)
    return result 


# In[45]:


x = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]


# In[46]:


get_google_estimate_now(*x)


# In[47]:


small_df = df.loc[0:3,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]


# In[48]:


for vals in small_df.values:
    print(get_google_estimate_now(*vals))


# In[49]:


df.loc[0:3,['trip_duration']]


# In[50]:


google_estimate = []
df_coordinates = df.loc[:1000,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]

for vals in df_coordinates.values:
    google_estimate.append(get_google_estimate_now(*vals))
    print(google_estimate[-1:])


# In[51]:


len(google_estimate)


# In[52]:


google_estimate[1]


# In[53]:


estimates = [est['rows'][0]['elements'][0]['duration']['value'] for est in google_estimate]


# In[54]:


estimates[:10]


# In[55]:


df['trip_duration'][:10]


# In[56]:


first_thousand = pd.DataFrame({'trip_time':df['trip_duration'][:len(estimates)],'est_time':pd.Series(estimates)})


# In[57]:


first_thousand.head()


# In[58]:


first_thousand.plot(x = 'trip_time',
                    y= 'est_time',
                    kind='scatter',
                    title='correlation between trip time and google estimate')


# In[59]:


df['trip_duration'][:len(estimates)][df['trip_duration'][:len(estimates)]> 80000]


# In[60]:


first_thousand.iloc[531]


# In[61]:


first_thousand.drop(531).plot(x = 'est_time',
                              y='trip_time',
                              kind='scatter',
                              title = 'correlation between trip time and google estimate')


# In[62]:


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


# In[63]:


test_dt = df.loc[0,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'pickup_datetime']]


# In[64]:


get_google_estimate_future(*test_dt, api_key=environ["GOOGLE_API_KEY"])


# In[65]:


pd.DatetimeIndex


# In[66]:


df['day_of_week'] = df['pickup_datetime'].dt.dayofweek


# In[67]:


pd.DataFrame(list(Counter(df['day_of_week']).values()),
             index=['Mon','Tue','Wed','Thur','Fri','Sat','Sun']).plot(title='Number of trips by weekday',
                                                                      figsize=(10,8))


# In[68]:


ax = df['trip_duration'].groupby(df['day_of_week']).median().plot(title='Median length of trip by weekday',
                                                                  figsize=(10,8))
ax.set_xticklabels(['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])

