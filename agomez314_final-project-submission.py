#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import folium
from folium.plugins import HeatMap

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Setting up BigQuery library
PROJECT_ID = 'villanova-project'
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()
from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# load biquery commands
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[2]:


# Read data
df_train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
df_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')


# In[3]:


df_train.info()


# In[4]:


df_test.info()


# In[5]:


df_train.isnull().sum()


# In[6]:


obj_df = df_train.select_dtypes(include=['object'])
obj_df[obj_df.isnull().any(axis=1)].count()


# In[7]:


# sns.pairplot(df_train)


# In[8]:


# Checking for distribution of ALL DATA for each city
train_plot = sns.countplot(x="City", data=df_train)
train_plot


# In[9]:


# Checking for distribution of data BY UNIQUE INTERSECTION ID
fig = df_train.groupby(['City'])['IntersectionId'].nunique().sort_index().plot.bar()
fig.set_title('# of Intersections per city in train Set', fontsize=15)
fig.set_ylabel('# of Intersections', fontsize=15);
fig.set_xlabel('City', fontsize=17);


# In[10]:


# let's see the distribution of traffic by month and date
plt.figure(figsize=(15,12))

plt.subplot(211)
g = sns.countplot(x="Hour", data=df_train, hue='City', dodge=True)
g.set_title("Distribution by hour and city", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Hours of Day", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)

plt.subplot(212)
g1 = sns.countplot(x="Month", data=df_train, hue='City', dodge=True)
g1.set_title("Hour Count Distribution by Month and City", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("Months", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)

g1.set_ylim(0, max(sizes) * 1.15)

plt.subplots_adjust(hspace = 0.3)

plt.show()


# In[11]:




fig = df_train.groupby(['City']).TotalTimeStopped_p80.median().sort_index().plot(kind='barh')

fig.set_title('Average Stopping Time', fontsize=15)
fig.set_ylabel('City', fontsize=10);
fig.set_xlabel('Minutes stopped at intersection', fontsize=10);


# In[12]:


Atlanda=df_train[df_train['City']=='Atlanta'].copy()
Boston=df_train[df_train['City']=='Boston'].copy()
Chicago=df_train[df_train['City']=='Chicago'].copy()
Philadelphia=df_train[df_train['City']=='Philadelphia'].copy()


# In[13]:


Atlanda['TotalTimeWaited']=Atlanda['TotalTimeStopped_p20']+Atlanda['TotalTimeStopped_p40']+Atlanda['TotalTimeStopped_p50']+Atlanda['TotalTimeStopped_p60']+Atlanda['TotalTimeStopped_p80']
Boston['TotalTimeWaited']=Boston['TotalTimeStopped_p20']+Boston['TotalTimeStopped_p40']+Boston['TotalTimeStopped_p50']+Boston['TotalTimeStopped_p60']+Boston['TotalTimeStopped_p80']
Chicago['TotalTimeWaited']=Chicago['TotalTimeStopped_p20']+Chicago['TotalTimeStopped_p40']+Chicago['TotalTimeStopped_p50']+Chicago['TotalTimeStopped_p60']+Chicago['TotalTimeStopped_p80']
Philadelphia['TotalTimeWaited']=Philadelphia['TotalTimeStopped_p20']+Philadelphia['TotalTimeStopped_p40']+Philadelphia['TotalTimeStopped_p50']+Philadelphia['TotalTimeStopped_p60']+Philadelphia['TotalTimeStopped_p80']


# In[14]:


Atlanda['TotalTimeWaited'].hist(bins=100)


# In[15]:


temp_1=Atlanda.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_1.plot(kind='barh',title='Highest traffic startng street in Atlanta')


# In[16]:


temp_2=Boston.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_2.plot(kind='barh',title='Highest traffic startng street in Boston')


# In[17]:


temp_3=Chicago.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_3.plot(kind='barh',title='Highest traffic startng street in Chicago')


# In[18]:


temp_4=Philadelphia.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_4.plot(kind='barh',title='Highest traffic startng street in Philadelphia')


# In[19]:


# use plotly to plot where intersections are for all cities. Provide observational data. 
# then do a heatmap which groups intersectionId with TotalStoppingTime across space to see where the heaviest traffic is. 
# Investigate what's around here and provide observations.
traffic_df=Atlanda.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[33.7638493,-84.3801108], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[20]:


traffic_df=Boston.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[42.3158246,-71.0787574], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[21]:


traffic_df=Chicago.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[41.8420892,-87.7237629], zoom_start=11)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[22]:


traffic_df=Philadelphia.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[39.9484792,-75.1774329], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[23]:


t_stopped = ['TotalTimeStopped_p20',
             'TotalTimeStopped_p50', 
             'TotalTimeStopped_p80']
t_first_stopped = ['TimeFromFirstStop_p20',
                   'TimeFromFirstStop_p50',
                   'TimeFromFirstStop_p80']
d_first_stopped = ['DistanceToFirstStop_p20',
                   'DistanceToFirstStop_p50',
                   'DistanceToFirstStop_p80']


# In[24]:


plt.figure(figsize=(15,12))
plt.title('Correlation of Time and Distance Stopped', fontsize=17)
sns.heatmap(df_train[t_stopped + t_first_stopped + d_first_stopped].astype(float).corr(), vmax=1.0,  annot=True)
plt.show()


# In[25]:


f,ax=plt.subplots(1,2,figsize=(12,5))

ax[0].set_title('Total time stopped vs Time from first stop', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='TimeFromFirstStop_p80', data=df_train, ax=ax[0])

ax[1].set_title('Total time stopped vs Distance to first stop', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='DistanceToFirstStop_p80', data=df_train, ax=ax[1])

plt.show()


# In[26]:


f,ax=plt.subplots(1,2,figsize=(12,5))

ax[0].set_title('Total time stopped vs Distance to first stop', fontsize=15)
sns.scatterplot(x="TimeFromFirstStop_p80", y='DistanceToFirstStop_p80', data=df_train, ax=ax[0])

ax[1].set_title('Total time stopped per IntersectionId', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='IntersectionId', data=df_train, ax=ax[1])

plt.show()


# In[27]:


df_train_cleaned = df_train.copy().dropna()


# In[28]:


df_train_cleaned['TotalTimeStopped_p20'].hist()


# In[29]:


# totalTimeStopped['TotalTimeStopped_p20'] = np.log(totalTimeStopped['TotalTimeStopped_p20'])
# totalTimeStopped.hist('TotalTimeStopped_p20',figsize=(8,5))


# In[30]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_dataset.model_TotalTimeStopped_p20`)\nORDER BY iteration ')


# In[31]:


# get table prediction
# create a table. one column has predicted values, the other actual.
# sort the table by highest difference
# analze the top datapoints for any common patterns

