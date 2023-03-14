#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas
import numpy as np
import scipy
from sklearn.decomposition import PCA


# In[2]:


building_metadata = "../input/ashrae-energy-prediction/building_metadata.csv"
sample_submission = "../input/ashrae-energy-prediction/sample_submission.csv"
test = "../input/ashrae-energy-prediction/test.csv"
train = "../input/ashrae-energy-prediction/train.csv"
weather_test = "../input/ashrae-energy-prediction/weather_test.csv"
weather_train = "../input/ashrae-energy-prediction/weather_train.csv"


# In[3]:


def get_data(data_file, weather_data_file, building_data = 'building_metadata.csv', path = './'):
    building_data = pandas.read_csv(path+building_metadata, dtype={'building_id':np.uint16, 'site_id':np.uint8})
    data = pandas.read_csv(path+data_file, dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])
    weather_data = pandas.read_csv(path+weather_data_file, parse_dates=['timestamp'],
                                                               dtype={'site_id':np.uint8, 'air_temperature':np.float16,
                                                                      'cloud_coverage':np.float16, 'dew_temperature':np.float16,
                                                                      'precip_depth_1_hr':np.float16})
    data = data.merge(building_data, on='building_id', how='left')
    data = data.merge(weather_data, on=['site_id', 'timestamp'], how='left')
    
    return data


# In[4]:


train_data = get_data(train, weather_train)


# In[5]:


for col in train_data.columns:
    if np.nan in train_data[col]:
        print train_data[col]


# In[6]:


train_data = train_data.dropna()
train_data2 = train_data[train_data.meter_reading != 0]
train_data2 = train_data2.drop(columns=['building_id', 'site_id', 'primary_use', 'timestamp'], errors='ignore')
target = train_data2.meter_reading
train_data2 = train_data2.drop(columns=['meter_reading'], errors='ignore')


# In[7]:


import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

def getLinearRegressorPandasDF(train_dataX, train_dataY):
#     train_dataY = train_data['meter_reading']
#     train_dataX = train_dataX.drop(['timestamp'],axis=1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(train_dataX,train_dataY)
    Y_pred = linear_regressor.predict(train_dataX)
    print(Y_pred)
    return linear_regressor
    
lr = getLinearRegressorPandasDF(train_dataX=train_data2, train_dataY=target)


# In[8]:


test_data2 = test_data.drop(columns=['building_id', 'site_id', 'primary_use', 'timestamp', 'meter_reading'], errors='ignore')


# In[9]:


test_pred = lr.predict(test_data2)


# In[ ]:





# In[10]:


for i in tqdm(range(max(train_data.building_id))):
    train_data[train_data.building_id == i] = train_data[train_data.building_id == i].sort_values('timestamp').interpolate(method='linear', limit_direction='forward', axis=0)

