#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,accuracy_score
from tqdm import tqdm
import xgboost as xgb

import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train=pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test=pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')




submission=pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')









census=pd.read_csv('../input/new-york-city-census-data/census_block_loc.csv')
census.head()




census.County.value_counts()




train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]




train.trip_duration.describe()




len(train.trip_duration.value_counts()) #[:100000].hist()




train['log_trip_duration']=np.log(train['trip_duration'].values)
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()




train.head()




train[["id","trip_duration"]].head()




test["id"].head()




train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].head()




test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].head()




from sklearn.metrics.pairwise import manhattan_distances




manhattan_distances([[-73.988129,40.732029]],[[-73.990173,40.756680]])




def manhattan(row):
    return manhattan_distances([[row['pickup_longitude'],row['pickup_latitude']]],[[row['dropoff_longitude'],row['dropoff_latitude']]])[0][0]




tqdm.pandas()




get_ipython().run_line_magic('timeit', '')
train['manhattan_distance']=train.progress_apply(manhattan,axis=1)




get_ipython().run_line_magic('timeit', '')
test['manhattan_distance']=test.progress_apply(manhattan,axis=1)




set(train.columns)-set(test.columns)




train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])




train['pickup_hour']=train['pickup_datetime'].dt.hour
train['pickup_weekday']=train['pickup_datetime'].dt.weekday
train['pickup_week']=train['pickup_datetime'].dt.week
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_weekday']=test['pickup_datetime'].dt.weekday
test['pickup_week']=test['pickup_datetime'].dt.week




kmeans_pickup = MiniBatchKMeans(n_clusters=100,random_state=0,batch_size=5000)
kmeans_dropoff=MiniBatchKMeans(n_clusters=100,random_state=0,batch_size=5000)




kmeans_pickup.fit(train[['pickup_longitude','pickup_latitude']])
kmeans_dropoff.fit(train[['dropoff_longitude','dropoff_latitude']])




train['pickup_cluster']=kmeans_pickup.predict(train[['pickup_longitude','pickup_latitude']])
train['dropoff_cluster']=kmeans_pickup.predict(train[['dropoff_longitude','dropoff_latitude']])




test['pickup_cluster']=kmeans_pickup.predict(test[['pickup_longitude','pickup_latitude']])
test['dropoff_cluster']=kmeans_pickup.predict(test[['dropoff_longitude','dropoff_latitude']])




train.head()




train.log_trip_duration.hist()




train[['vendor_id','passenger_count','store_and_fwd_flag','log_trip_duration','manhattan_distance','pickup_hour','pickup_weekday'
       ,'pickup_week','pickup_cluster','dropoff_cluster']].head()




train_vendor=pd.get_dummies(train['vendor_id'],prefix='vendor_id')
test_vendor=pd.get_dummies(test['vendor_id'],prefix='vendor_id')




train_passenger_count=pd.get_dummies(train['passenger_count'],prefix='passengers')
test_passenger_count =pd.get_dummies(test['passenger_count'],prefix='passengers')




train_flag=pd.get_dummies(train['store_and_fwd_flag'],prefix='flag')
test_flag =pd.get_dummies(test['store_and_fwd_flag'],prefix='flag')




day_map={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
train['pickup_weekday']=train['pickup_weekday'].map(day_map)
test['pickup_weekday']=test['pickup_weekday'].map(day_map)




train_day=pd.get_dummies(train['pickup_weekday'])
test_day =pd.get_dummies(test['pickup_weekday'])




train_hours=pd.get_dummies(train['pickup_hour'],prefix='hour')
test_hours =pd.get_dummies(test['pickup_hour'],prefix='hour')




train_pickup=pd.get_dummies(train['pickup_cluster'],prefix='pickup_cluster')
test_pickup =pd.get_dummies(test['pickup_cluster'],prefix='pickup_cluster')




train_dropoff=pd.get_dummies(train['dropoff_cluster'],prefix='dropoff_cluster')
test_dropoff =pd.get_dummies(test['dropoff_cluster'],prefix='dropoff_cluster')




frames_train=[train_vendor,train_passenger_count,train_flag,train_hours,train_day,train_pickup,train_dropoff]
frames_test=[test_vendor,test_passenger_count,test_flag,test_hours,test_day,test_pickup,test_dropoff]




len(frames_train)




result_train = pd.concat(frames_train,axis=1)
result_test = pd.concat(frames_test,axis=1)




result_train.tail()




result_test.tail()




train['manhattan_distance'].head()




test['manhattan_distance'].head()




result_train=result_train.join(train['manhattan_distance'])
result_test=result_test.join(test['manhattan_distance'])




result_train.tail()




result_test.tail()




list(set(result_test.columns)-set(result_train.columns))[0]




result_test['passengers_9'].value_counts()




result_test.drop('passengers_9',inplace=True,axis=1)




y=train['log_trip_duration']




X_train,X_test,y_train,y_test=train_test_split(result_train,y)




dtrain=xgb.DMatrix(X_train,label=y_train)
dvalid=xgb.DMatrix(X_test,label=y_test)
dtest=xgb.DMatrix(result_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]




xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 9,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}




model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)




xgb.plot_importance(model, max_num_features=28, height=0.7)




xgb.to_graphviz(model, num_trees=5)




ypred=model.predict(dtest,ntree_limit=model.best_ntree_limit)




predict=pd.DataFrame(columns=submission.columns)




predict.id=test.id




predict.trip_duration=np.exp(ypred)




predict.head()




submission=pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission/sample_submission.csv')
submission.head()




submission.to_csv("submission.csv", index=False)






