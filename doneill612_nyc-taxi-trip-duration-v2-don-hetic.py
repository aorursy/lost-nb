#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import xgboost as xgb

from catboost import CatBoostRegressor, Pool
from catboost import cv as catboost_cv
from IPython.display import display

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error




get_ipython().run_line_magic('matplotlib', 'inline')




DO_XGBOOST = True
DO_CATBOOST = not DO_XGBOOST




train_raw = pd.read_csv('../input/train/train.csv', index_col='id', 
                        parse_dates=['pickup_datetime', 'dropoff_datetime'])
test_raw = pd.read_csv('../input/test/test.csv', index_col='id', 
                       parse_dates=['pickup_datetime'])




print('Train set:\n')
train_raw.info()




print('Test set:\n')
test_raw.info()




print('NaN or empty data (train): {}'.format(train_raw.isnull().values.any()))
print('NaN or empty data (test): {}'.format(test_raw.isnull().values.any()))




def check_hidden(*dfs):
    for df in dfs:
        display(df.describe())

check_hidden(train_raw, test_raw)




max_dur_hrs = np.max(train_raw["trip_duration"].values) / 3600
print(f'Max trip duration (hours): {max_dur_hrs:.3f}')

_, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), sharex=False)

ax0.set_title('Trip duration distribution')
ax0.set_xlabel('Trip duration (s)')

ax1.set_title('LogP1-Trip duration distribution')
ax1.set_xlabel('log(Trip Duration + 1) ')

ax0.hist(train_raw['trip_duration'], bins=100);
ax1.hist(np.log1p(train_raw['trip_duration']), bins=100);




train_set = train_raw.copy(deep=True).drop('dropoff_datetime', axis=1)
test_set = test_raw.copy(deep=True)




checkpoints = {
    'pre_engineering': {
        'train': train_set.copy(deep=True),
        'test': test_set.copy(deep=True)
    }
}




def add_checkpoint(key):
    d = { 'train': train_set.copy(deep=True), 'test': test_set.copy(deep=True) }
    checkpoints.update({
        key: d
    })




def restore_checkpoint(key):
    return checkpoints[key]['train'].copy(deep=True), checkpoints[key]['test'].copy(deep=True)




def cat_encode(*dfs):
    for df in dfs:
        for col in df.select_dtypes('object').columns:
            df[col] = df[col].astype('category').cat.codes




cat_encode(train_set, test_set)




def extract_date_columns(*dfs):
    for df in dfs:
        df['pickup_datetime_month'] = df.pickup_datetime.dt.month
        df['pickup_datetime_hour'] = df.pickup_datetime.dt.hour
        df['pickup_datetime_minute'] = df.pickup_datetime.dt.minute
        df['pickup_datetime_dow'] = df.pickup_datetime.dt.dayofweek




extract_date_columns(train_set, test_set)




def _haversine_distance(lat_a, long_a, lat_b, long_b):
    '''Calculates the haversine distance between two geographic coordinates in meters.
    
    The haversine distance is defined as "the distance a crow flies" between two points.
    
    Parameters
    ----------
        lat_a  : latitude of point A (array-like)
        long_a : longitude of point A (array-like)
        lat_b  : latitude of point B  (array-like)
        long_b : longitude of point B (array-like)
    
    Returns
    -------
        the haversine distance between the two points
    '''
    del_lat_rad = lat_b - lat_a
    del_long_rad = long_b - long_a
    
    lat_a, long_a, lat_b, long_b, del_lat_rad, del_long_rad =         map(np.radians, [lat_a, long_a, lat_b, long_b, del_lat_rad, del_long_rad])
    
    def _a(del_lat, del_long, lat_a, lat_b):
        return (
            (np.sin(del_lat / 2) ** 2) +
            (np.cos(lat_a) * np.cos(lat_b) *
             np.sin(del_long / 2) ** 2)
        )
    
    
    def _c(a):
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a));
    
    # radius of the earth in meters
    R = 6371e3
    
    a = _a(del_lat_rad, del_long_rad, lat_a, lat_b)
    c = _c(a)
    
    return R * c

def add_manhattan_haversine_distance(*dfs):
    '''Calculates the "true" Manhattan distance between two geographic coordinates.
    
    This is probably a bad estimation of a city-block distance, but is better than nothing.
    
    Parameters
    ----------
        dfs : dataframes in which to add the manhattan-haversine distance column
    '''
    for df in dfs:
        inbounds = df.loc[(df['pickup_longitude'] <= -73.94) & 
                           df['dropoff_longitude'] <= -73.94]

        inbounds_idx = inbounds.index

        oob = df.loc[~df.index.isin(inbounds_idx), :]
        oob_idx = oob.index

        ib_lat_a = inbounds['pickup_latitude']
        ib_lat_b = inbounds['dropoff_latitude']
        ib_long_a = inbounds['pickup_longitude']
        ib_long_b = inbounds['dropoff_longitude']

        oob_lat_a = oob['pickup_latitude']
        oob_lat_b = oob['dropoff_latitude']
        oob_long_a = oob['pickup_longitude']
        oob_long_b = oob['dropoff_longitude']

        df['city_distance'] = pd.Series(np.zeros(shape=(len(df.index))), index=df.index)
        df.loc[inbounds_idx, 'city_distance'] = _haversine_distance(ib_lat_a, ib_long_a, 
                                                                    ib_lat_a, ib_long_b) + \
                                                _haversine_distance(ib_lat_a, ib_long_a, 
                                                                    ib_lat_b, ib_long_a)
        df.loc[oob_idx, 'city_distance'] = _haversine_distance(oob_lat_a, oob_long_a, 
                                                               oob_lat_b, oob_long_b)




add_manhattan_haversine_distance(train_set, test_set)




add_checkpoint('with_manhattan')




_, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

ax0.set_title('Trip distances')

ax1.set_title('log(Trip distances + 1)')

ax0.hist(train_set['city_distance'], color='blue', alpha=0.2, bins=100, label='train')
ax0.hist(test_set['city_distance'], color='red', bins=100, label='test')
ax1.hist(np.log1p(train_set['city_distance']), color='blue', 
         alpha=0.2, bins=100, label='train')
ax1.hist(np.log1p(test_set['city_distance']), 
         color='red',  bins=100, label='test')

ax0.legend()
ax1.legend();




corr = train_set.corr()




_, ax = plt.subplots(1, figsize=(12, 12))

ax.xaxis.set_ticklabels([c for c in corr.columns.values], rotation=45)
ax.yaxis.set_ticklabels([c for c in corr.columns.values], rotation=0)

sns.heatmap(corr, annot=True, ax=ax, fmt='.3f');




_, ax = plt.subplots(1, figsize=(8, 8))
ax.set_title('Trip duration vs Trip Distance')
ax.set_xlabel('City distance (m)')
ax.set_ylabel('Trip duration (s)')
ax.scatter(train_set['city_distance'][:500], train_set['trip_duration'][:500]);




h = train_set.groupby('pickup_datetime_hour').mean()['trip_duration']
dow = train_set.groupby('pickup_datetime_dow').mean()['trip_duration']
_, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
ax0.plot(h.index.values, h.values)
ax0.set_title('Mean trip duration vs Hour')
ax0.set_ylabel('Mean trip duration (s)');
ax0.set_xlabel('Hour of day')

ax1.plot(dow.index.values, dow.values)
ax1.set_title('Mean trip duration vs DOW')
ax1.set_xlabel('Day of week (Sunday-Saturday)');




island_long_border = (-74.03, -73.75)
island_lat_border = (40.63, 40.85)

pickup_lat_train = train_set.pickup_latitude[:100000]
pickup_long_train = train_set.pickup_longitude[:100000]

dropoff_lat_train = train_set.dropoff_latitude[:100000]
dropoff_long_train = train_set.dropoff_longitude[:100000]

pickup_lat_test = test_set.pickup_latitude[:100000]
pickup_long_test = test_set.pickup_longitude[:100000]

dropoff_lat_test = test_set.dropoff_latitude[:100000]
dropoff_long_test = test_set.dropoff_longitude[:100000]

_fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
ax0.set_xlim(island_long_border)
ax0.set_ylim(island_lat_border);
ax0.set_title('Pickup locations (train)')
ax0.set_ylabel('Latitude')
ax0.set_xlabel('Longitude')
ax0.scatter(pickup_long_train, pickup_lat_train, c='blue', s=0.5)
ax1.scatter(dropoff_long_train, dropoff_lat_train, c='red', s=0.5)
ax1.set_title('Dropoff locations (train)')

_fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
ax0.set_xlim(island_long_border)
ax0.set_ylim(island_lat_border);
ax0.set_title('Pickup locations (test)')
ax0.set_ylabel('Latitude')
ax0.set_xlabel('Longitude')
ax0.scatter(pickup_long_test, pickup_lat_test, c='green', s=0.5)
ax1.scatter(dropoff_long_test, dropoff_lat_test, c='purple', s=0.5);
ax1.set_title('Dropoff locations (test)');




get_ipython().run_cell_magic('time', '', "cluster_set = train_set.loc[:, ['dropoff_latitude', 'dropoff_longitude']]\n\nk = 50\nmbkmeans = MiniBatchKMeans(n_clusters=k, batch_size=32)\nmbkmeans.fit(cluster_set)\ncluster_set['cluster'] = pd.Series(mbkmeans.labels_, index=cluster_set.index)\n\n_, ax = plt.subplots(1, figsize=(12, 12))\n\nisland_long_border = (-74.03, -73.75)\nisland_lat_border = (40.63, 40.85)\n\nlat = cluster_set.iloc[:100000, :]['dropoff_latitude']\nlon = cluster_set.iloc[:100000, :]['dropoff_longitude']\nclusters = cluster_set.iloc[:100000, :]['cluster']\n\nax.scatter(lon.values, lat.values, c=clusters, cmap='tab20c', alpha=0.5, s=1)\nax.set_xlim(island_long_border)\nax.set_ylim(island_lat_border);\n\ntrain_set['dropoff_cluster'] = cluster_set['cluster']\ntest_set['dropoff_cluster'] = \\\n    pd.Series(mbkmeans.predict(test_set.loc[:, ['dropoff_latitude', 'dropoff_longitude']].values),\n              index=test_set.index)\n\ntrain_set['pickup_cluster'] = \\\n    pd.Series(mbkmeans.predict(train_set.loc[:, ['pickup_latitude', 'pickup_longitude']].values),\n              index=train_set.index)\ntest_set['pickup_cluster'] = \\\n    pd.Series(mbkmeans.predict(test_set.loc[:, ['pickup_latitude', 'pickup_longitude']].values),\n              index=test_set.index);")




add_checkpoint('post_clustering')




train_set.drop('pickup_datetime', axis=1, inplace=True)
test_set.drop('pickup_datetime', axis=1, inplace=True)
add_checkpoint('pre_modeling')




X = train_set.drop('trip_duration', axis=1)
y = np.log1p(train_set['trip_duration'])




X_train, X_validate, y_train, y_validate = train_test_split(X, y)




xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_validate = xgb.DMatrix(X_validate, label=y_validate)

evallist = [(xgb_train, 'training-set'), (xgb_validate, 'validation-set')]

xgb_params = {
    'n_estimators': 250,
    'eta': 0.2,
    'subsample': 0.6,
    'max_depth': 10,
    'colsample_bytree': 0.75,
    'min_child_weight': 25,
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'nthread': -1,
    'silent': 1,
    'eval_metric': 'rmse'
}




if DO_XGBOOST:
    xgb_regressor = xgb.train(xgb_params, xgb_train, 100, evallist,
                              early_stopping_rounds=100, maximize=False,
                              verbose_eval=10)




cat_features = [0, 6, 12, 13]
train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
validation_pool = Pool(X_validate, label=y_validate, cat_features=cat_features)
cat_regressor = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='RMSE',
    iterations=250,
    learning_rate=0.2,
    depth=10
)




if DO_CATBOOST:
    cat_regressor.fit(train_pool, eval_set=validation_pool)




def create_xgb_submission():
    preds = xgb_regressor.predict(xgb.DMatrix(test_set))
    preds_df = pd.DataFrame(np.expm1(preds), index=test_set.index)
    preds_df.columns = ['trip_duration']
    
    preds_df.to_csv('submission.csv')
    display(pd.read_csv('submission.csv').head())
    
def create_catboost_submission():
    pool = Pool(test_set, cat_features=cat_features)
    preds = cat_regressor.predict(pool)
    preds_df = pd.DataFrame(np.expm1(preds), index=test_set.index)
    preds_df.columns = ['trip_duration']
    
    preds_df.to_csv('submission.csv')
    display(pd.read_csv('submission.csv').head())

if DO_XGBOOST:
    create_xgb_submission()
if DO_CATBOOST:
    create_catboost_submission()

