#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for plot visualization

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import skew

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[ ]:


plt.figure(figsize=(8, 5), dpi=80)
sns.set_style("darkgrid")


# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')
train_dataset = pd.read_csv('../input/train.csv', nrows=2_000_000)


# In[ ]:


train_dataset.head(5)


# In[ ]:


train_dataset.tail(5)


# In[ ]:


train_dataset.dtypes


# In[ ]:


# lets check current memory usage status
train_dataset.info(memory_usage='deep')


# In[ ]:


for dtype in ['float','int','object']:
    selected_dtype = train_dataset.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))


# In[ ]:


train_dataset.drop(labels='key', axis=1, inplace=True)
test_dataset.drop(labels='key', axis=1, inplace=True)


# In[ ]:


# Let's again check the memory usage.
train_dataset.info(memory_usage='deep')


# In[ ]:


train_dataset.passenger_count = train_dataset.passenger_count.astype(dtype = 'uint8')


# In[ ]:


train_dataset.pickup_longitude = train_dataset.pickup_longitude.astype(dtype = 'float32')
train_dataset.pickup_latitude = train_dataset.pickup_latitude.astype(dtype = 'float32')
train_dataset.dropoff_longitude = train_dataset.dropoff_longitude.astype(dtype = 'float32')
train_dataset.dropoff_latitude = train_dataset.dropoff_latitude.astype(dtype = 'float32')
train_dataset.fare_amount = train_dataset.fare_amount.astype(dtype = 'float32')


# In[ ]:


# let's again check the memory_usage report
train_dataset.info(memory_usage='deep')


# In[ ]:


train_dataset.isnull().sum()


# In[ ]:


print(f'Row count before drop-null operation - {train_dataset.shape[0]}')
train_dataset.dropna(inplace = True)
print(f'Row count after drop-null operation - {train_dataset.shape[0]}')


# In[ ]:


train_dataset['pickup_datetime'] = pd.to_datetime(arg=train_dataset['pickup_datetime'], infer_datetime_format=True)
test_dataset['pickup_datetime'] = pd.to_datetime(arg=test_dataset['pickup_datetime'], infer_datetime_format=True)


# In[ ]:


train_dataset.dtypes


# In[ ]:


def add_new_date_time_features(dataset):
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['year'] = dataset.pickup_datetime.dt.year
    dataset['day_of_week'] = dataset.pickup_datetime.dt.dayofweek
    
    return dataset

train_dataset = add_new_date_time_features(train_dataset)
test_dataset = add_new_date_time_features(test_dataset)


# In[ ]:


train_dataset.describe()


# In[ ]:


print(f'Rows before removing coordinate outliers - {train_dataset.shape[0]}')

train_dataset = train_dataset[train_dataset.pickup_longitude.between(test_dataset.pickup_longitude.min(), test_dataset.pickup_longitude.max())]
train_dataset = train_dataset[train_dataset.pickup_latitude.between(test_dataset.pickup_latitude.min(), test_dataset.pickup_latitude.max())]
train_dataset = train_dataset[train_dataset.dropoff_longitude.between(test_dataset.dropoff_longitude.min(), test_dataset.dropoff_longitude.max())]
train_dataset = train_dataset[train_dataset.dropoff_latitude.between(test_dataset.dropoff_latitude.min(), test_dataset.dropoff_latitude.max())]

print(f'Rows after removing coordinate outliers - {train_dataset.shape[0]}')


# In[ ]:


train_dataset.describe()


# In[ ]:


train_dataset.fare_amount[(train_dataset.fare_amount <= 0) | (train_dataset.fare_amount >= 350)].count()


# In[ ]:


# Let's eliminate these rows
print(f'Row count before elimination - {train_dataset.shape[0]}')
train_dataset = train_dataset[train_dataset.fare_amount.between(0, 350, inclusive=False)]
print(f'Row count after elimination - {train_dataset.shape[0]}')


# In[ ]:


train_dataset.passenger_count[(train_dataset.passenger_count < 1) | (train_dataset.passenger_count > 8)].count()


# In[ ]:


# Let's eliminate these rows
print(f'Row count before elimination - {train_dataset.shape[0]}')
train_dataset = train_dataset[train_dataset.passenger_count.between(0, 8, inclusive=False)]
print(f'Row count after elimination - {train_dataset.shape[0]}')


# In[ ]:


def degree_to_radion(degree):
    return degree*(np.pi/180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = np.sin(lat_diff / 2)**2 + np.cos(degree_to_radion(from_lat)) * np.cos(degree_to_radion(to_lat)) * np.sin(long_diff / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return radius * c


# In[ ]:


train_dataset['distance'] = calculate_distance(train_dataset.pickup_latitude, train_dataset.pickup_longitude, train_dataset.dropoff_latitude, train_dataset.dropoff_longitude)
test_dataset['distance'] = calculate_distance(test_dataset.pickup_latitude, test_dataset.pickup_longitude, test_dataset.dropoff_latitude, test_dataset.dropoff_longitude)


# In[ ]:


train_dataset.sort_values(by='distance')


# In[ ]:


train_dataset.distance[(train_dataset.distance == 0)].count()


# In[ ]:


train_dataset[(train_dataset.pickup_latitude != train_dataset.dropoff_latitude) &
              (train_dataset.pickup_longitude != train_dataset.dropoff_latitude) &
              (train_dataset.distance == 0)].count()


# In[ ]:


def add_distances_from_airport(dataset):
    #coordinates of all these airports
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset['pickup_jfk_distance'] = calculate_distance(jfk_coords[0], jfk_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_jfk_distance'] = calculate_distance(jfk_coords[0], jfk_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    dataset['pickup_ewr_distance'] = calculate_distance(ewr_coords[0], ewr_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_ewr_distance'] = calculate_distance(ewr_coords[0], ewr_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    dataset['pickup_lga_distance'] = calculate_distance(lga_coords[0], lga_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_lga_distance'] = calculate_distance(lga_coords[0], lga_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    return dataset


train_dataset = add_distances_from_airport(train_dataset)
test_dataset = add_distances_from_airport(test_dataset)


# In[ ]:


sns.distplot(a=train_dataset.fare_amount)


# In[ ]:


sns.jointplot(x='distance', y='fare_amount', data=train_dataset)


# In[ ]:


g = sns.FacetGrid(train_dataset, col="year", hue="passenger_count")
g.map(plt.scatter, "distance", "fare_amount")
g.add_legend()


# In[ ]:


train_dataset[(train_dataset.distance>90) & (train_dataset.fare_amount<70)]


# In[ ]:


sns.countplot(x='day_of_week', data=train_dataset)


# In[ ]:


tc = train_dataset.pivot_table(index='day_of_week', columns='month', values='fare_amount')
sns.heatmap(data = tc)


# In[ ]:


train_dataset['fare_amount'].skew()


# In[ ]:


train_dataset['fare_amount'] = np.log1p(train_dataset['fare_amount'])
sns.distplot(train_dataset['fare_amount'], color='blue')


# In[ ]:


selected_predictors = [
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude',
    'pickup_jfk_distance',
    'dropof_jfk_distance',
    'pickup_ewr_distance',
    'dropof_ewr_distance',
    'pickup_lga_distance',
    'dropof_lga_distance',
    'hour',
    'month',
    'year',
    'distance'
]

X = train_dataset.loc[:, selected_predictors].values
y = train_dataset.iloc[:, 0].values
X_test_dataset = test_dataset.loc[:, selected_predictors].values

# Since test_dataset is too large, So we are going to keep only 5% of the dataset in test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/20)


# In[ ]:


rand_forest_regressor = RandomForestRegressor()
rand_forest_regressor.fit(X_train, y_train)

y_rand_forest_predict = rand_forest_regressor.predict(X_test)
random_forest_model_error = sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_rand_forest_predict)))
print(f' Random Forest Mean Squared Error - {random_forest_model_error}')


# In[ ]:


# parameters = {
#                 'learning_rate': [0.07, 0.1, 0.3],
#                 'max_depth': [3, 5, 7],
#                 'n_estimators': [200, 400, 500]
#             }

# XGB_hyper_params = GridSearchCV(estimator=XGB_regressor, param_grid=parameters, n_jobs=-1, cv=5)


# In[ ]:


# XGB_hyper_params.fit(X_train[:50_000], y_train[:50_000])
# # find out the best hyper parameters
# XGB_hyper_params.best_params_


# In[ ]:


XGB_model = XGBRegressor(learning_rate=0.3, max_depth=6, n_estimators=500)
XGB_model.fit(X_train, y_train)
y_XGB_predict = XGB_model.predict(X_test)

XGB_model_error = sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_XGB_predict)))

print(f'XGBoost Mean Squared Error - {XGB_model_error}')


# In[ ]:


# let's plot feature_importance again and check if there is any difference or not.
sns.barplot(y=list(train_dataset.loc[:, selected_predictors].columns), x=list(XGB_model.feature_importances_))


# In[ ]:


lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=35, n_estimators=300)

lgb_model.fit(X_train, y_train)
y_LGB_predict = lgb_model.predict(X_test)

LGB_model_error = sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_LGB_predict)))

print(f'LGBM Mean Squared Error - {LGB_model_error}')


# In[ ]:


# ensembled prediction over splitted test data
ensembled_prediction = (0.5*np.expm1(y_XGB_predict))+(0.5*np.expm1(y_LGB_predict))
ensembled_prediction_error = sqrt(mean_squared_error(np.expm1(y_test), ensembled_prediction))

print(f'Ensembled Mean Squared Error - {ensembled_prediction_error}')


# In[ ]:


# making prediction using test_dataset predictors
y_XGB_predict = np.expm1(XGB_model.predict(X_test_dataset))

# submitting our predictions
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = y_XGB_predict
submission.to_csv('xgb_submission.csv', index=False)
submission.head(10)


# In[ ]:


# making prediction using test_dataset predictors
y_LGB_predict = np.expm1(lgb_model.predict(X_test_dataset))

# submitting our predictions
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = y_LGB_predict
submission.to_csv('lgbm_submission.csv', index=False)
submission.head(10)


# In[ ]:


# making prediction using test_dataset predictors
# y_rand_forest_predict = np.expm1(rand_forest_regressor.predict(X_test_dataset))

# # submitting our predictions
# submission = pd.read_csv('../input/sample_submission.csv')
# submission['fare_amount'] = y_rand_forest_predict
# submission.to_csv('random_forest_submission.csv', index=False)
# submission.head(10)


# In[ ]:


# submitting our predictions
ensembled_prediction = (0.5*y_XGB_predict)+(0.5*y_LGB_predict)
submission.to_csv('ensembled_submission.csv', index=False)
submission.head(10)

