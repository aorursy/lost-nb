#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))


# In[2]:


df = pd.read_csv('../input/train.csv',index_col='id')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
#df['trip_duration'] = df['trip_duration'] / 3600


# In[3]:


df.describe()


# In[4]:


df_datetime = df['pickup_datetime']
df['year_pickup'] = df_datetime.dt.year
df['month_pickup'] = df_datetime.dt.month
df['day_pickup'] = df_datetime.dt.day
df['hour_pickup'] = df_datetime.dt.hour
df['weekday_pickup'] = df_datetime.dt.weekday
df['minute_pickup'] = df_datetime.dt.minute
df['second_pickup'] = df_datetime.dt.second


# In[5]:


selected_columns = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',
                    'dropoff_longitude','dropoff_latitude','year_pickup','month_pickup',
                    'day_pickup','hour_pickup','weekday_pickup','minute_pickup','second_pickup']


filter_passenger_duration =  (df['passenger_count'] > 0) & (df['trip_duration'] > 120) & (df['trip_duration'] < 10800 )
filter_df = df.loc[filter_passenger_duration]
X = filter_df[selected_columns]
y = filter_df['trip_duration']
X.shape, y.shape


# In[6]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[7]:


from sklearn.model_selection import ShuffleSplit
shuffle = ShuffleSplit(n_splits=5, test_size=0.99, random_state=42)


# In[8]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100,200,500,700],
    'max_features': ['auto'],
    "max_depth": [1,3,5,10]
}
gs = GridSearchCV(estimator=rf,cv=shuffle,param_grid=param_grid,scoring="neg_mean_squared_log_error")
#gs.fit(X, y)
#print(gs.best_params_)
#y_predict = gs.predict(X_test)
#print(y_predict.mean())


# In[9]:


from sklearn.model_selection import cross_val_score
import sklearn
losses = cross_val_score(rf, X, y, scoring="neg_mean_squared_log_error", cv=shuffle)
np.sqrt(- losses.mean())


# In[10]:


df_test = pd.read_csv('../input/test.csv',index_col="id")
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_test.head()


# In[11]:


df_test_datetime = df_test['pickup_datetime']
df_test['year_pickup'] = df_test_datetime.dt.year
df_test['month_pickup'] = df_test_datetime.dt.month
df_test['day_pickup'] = df_test_datetime.dt.day
df_test['hour_pickup'] = df_test_datetime.dt.hour
df_test['weekday_pickup'] = df_test_datetime.dt.weekday + 1
df_test['minute_pickup'] = df_test_datetime.dt.minute
df_test['second_pickup'] = df_test_datetime.dt.second
df_test.head()


# In[12]:


rf.fit(X, y)


# In[13]:


X_test = df_test[selected_columns]
X_test.shape
y_pred = rf.predict(X_test)
y_pred.mean(), len(y_pred)


# In[14]:


submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[15]:


submission['trip_duration'] = y_pred
submission.to_csv('submission.csv', index=False)
submission.head()

