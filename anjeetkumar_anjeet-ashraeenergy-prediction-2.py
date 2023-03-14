#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import datetime
import gc
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.model_selection import train_test_split,KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objects as go
init_notebook_mode(connected=True)
import plotly.figure_factory as ff




#metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}




df_data_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv',parse_dates=['timestamp'])
df_meta_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
df_weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv',parse_dates=['timestamp'])
df_weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',parse_dates=['timestamp'])
df_data_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',parse_dates=['timestamp'])




## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




df_data_train = reduce_mem_usage(df_data_train)
df_weather_train = reduce_mem_usage(df_weather_train)
df_meta_train = reduce_mem_usage(df_meta_train)
df_data_test = reduce_mem_usage(df_data_test)
df_weather_test = reduce_mem_usage(df_weather_test)




def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
    return weather_df




df_weather_train = fill_weather_dataset(df_weather_train)
df_weather_test = fill_weather_dataset(df_weather_test)









get_ipython().run_cell_magic('time', '', 'df_data_train = pd.merge(df_data_train,df_meta_train,on=\'building_id\',how=\'left\')\ndf_data_test  = pd.merge(df_data_test,df_meta_train,on=\'building_id\',how=\'left\')\nprint ("Training Data Shape {}".format(df_data_train.shape))\nprint ("Testing Data Shape {}".format(df_data_test.shape))\ngc.collect()')




df_weather_train.info()




df_data_train.info()




get_ipython().run_cell_magic('time', '', 'df_data_train = df_data_train.merge(df_weather_train,on=[\'site_id\',\'timestamp\'], how=\'left\')\ndf_data_test  = df_data_test.merge(df_weather_test,on=[\'site_id\',\'timestamp\'], how=\'left\')\nprint ("Training Data Shape {}".format(df_data_train.shape))\nprint ("Testing Data Shape {}".format(df_data_test.shape))\ngc.collect()')




for df in [df_data_train,df_data_test]:
    df['square_feet'] = df['square_feet'].astype('float16')
    df['Age'] = df['timestamp'].dt.year - df['year_built']
    df['Age_isNa'] = df['year_built_isNa']




df_data_train.info()




df_meta_train.info()




df_merge_train_meta = df_data_train.merge(df_meta_train, how='left', on='building_id')




df_merge_test_meta = df_data_test.merge(df_meta_train, how='left', on='building_id')




df_merge_train_meta.loc[(df_merge_train_meta['site_id'] == 0) & (df_merge_train_meta['meter'] == 0),'meter_reading'] =       df_merge_train_meta[(df_merge_train_meta['site_id'] == 0) & (df_merge_train_meta['meter'] == 0)]       ['meter_reading'] * 0.293




df_merge_train_all = df_merge_train_meta.merge(df_weather_train,on=['site_id','timestamp'], how='left')




df_merge_test_all = df_merge_test_meta.merge(df_weather_test,on=['site_id','timestamp'], how='left')




#df_merge_test_all.isna()
#test_miss=pd.DataFrame({c:[sum(df_merge_test_all[c].isna()),(sum(df_merge_test_all[c].isna())/len(df_merge_test_all[c]))*100] \
 #                           for c in df_merge_test_all.columns} ,index=['Total','%'])




del test_miss




#train_miss=pd.DataFrame({c:[sum(df_merge_train_all[c].isna()),(sum(df_merge_train_all[c].isna())/len(df_merge_train_all[c]))*100] \
#                            for c in df_merge_train_all.columns} ,index=['Total','%'])




del df_data_train
del df_weather_train
del df_data_test
del df_weather_test




df_merge_train_all.dropna(axis=1,inplace=True)




df_merge_test_all.dropna(axis=1,inplace=True)




#df_merge_train_all.head(100)
df_merge_train_all = df_merge_train_all[(df_merge_train_all.meter_reading>0)]




df_merge_test_all.info()




df_merge_train_all.info()




df_merge_train_all.drop(['timestamp'],axis=1,inplace=True)




df_merge_test_all.drop(['timestamp'],axis=1,inplace=True)




df_merge_train_all.head()




df_meterType_0_train = df_merge_train_all[(df_merge_train_all.meter == 0) & (df_merge_train_all.meter_reading>0)]
df_meterType_1_train = df_merge_train_all[(df_merge_train_all.meter == 1) & (df_merge_train_all.meter_reading>0)]
df_meterType_2_train = df_merge_train_all[(df_merge_train_all.meter == 2) & (df_merge_train_all.meter_reading>0)]
df_meterType_3_train = df_merge_train_all[(df_merge_train_all.meter == 3) & (df_merge_train_all.meter_reading>0)]




df_meterType_0_test = df_merge_test_all[df_merge_test_all.meter == 0]
df_meterType_1_test = df_merge_test_all[df_merge_test_all.meter == 1]
df_meterType_2_test = df_merge_test_all[df_merge_test_all.meter == 2]
df_meterType_3_test = df_merge_test_all[df_merge_test_all.meter == 3]




print(np.unique(df_merge_test_all.primary_use)




columns_list  = list(df_meterType_0_train.columns)
features = list(set(columns_list)-set(['meter_reading','primary_use']))




print(features)




y = df_meterType_0_train.meter_reading.values




x = df_meterType_0_train[features].values




from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)




from sklearn.linear_model import LinearRegression




lm = LinearRegression()




lm.fit(X_train,y_train)




print(lm.intercept_)




predictions = lm.predict(X_test)




plt.scatter(y_test,predictions)




from sklearn import metrics




print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

