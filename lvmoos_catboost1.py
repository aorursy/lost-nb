#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time
import gc

import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor

get_ipython().system('pip install catboost')
get_ipython().system('pip install ipywidgets')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')
# !pip install -U xgboost

# import xgboost as xgb

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# AUTOTUNE = tf.data.experimental.AUTOTUNE

from catboost import CatBoostRegressor
#from catboost import CatBoostClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)





 # %% Read and set up data
#X_train = pd.read_pickle('/content/drive/My Drive/Colab Notebooks/A1/df_data_features.pkl')
X_train = pd.read_pickle('../input/featureengineering/df_data_features.pkl')
day_ini = X_train['d'].min()
day_train_last = 1913

print('start day',day_ini)

day_train_first = day_train_last - 1*365


#X_train = X_train.loc[(X_train['d'] >= day_train_first) | ((X_train['month'].isin([4,5])) & ~(X_train['year'].isin([2016, 2015, 2014, 2013]))) , :]
X_train = X_train.loc[(X_train['d'] >= day_train_first), :]
X_train = X_train.loc[((X_train['Lag28_rmean28_id'].notna()) & (X_train['type'] =='train')) | (X_train['type'] =='val'), :]

# X_train = data.loc[mask, :]
print('First training day: ', day_train_first)
print('Last training day: ', day_train_last)
print(X_train.groupby(['year', 'month'])['d'].count())



X_train.columns = [c.replace('[','') for c in X_train.columns]
X_train.columns = [c.replace(']','') for c in X_train.columns]
X_train.columns = [c.replace(' ','_') for c in X_train.columns]
X_train.columns = [c.replace(',','') for c in X_train.columns]
X_train.columns = [c.replace("'",'') for c in X_train.columns]
print(X_train.info())
X_train.head()




def repeat_data(df):
    rep_month = 0
    rep_event = 0
    rep_event_test = 0
    
    months = [5]
    events_test = ['Pesach End', 'OrthodoxEaster', 'Cinco De Mayo', "Mother's day"]
    
    # make new row with event2 instead of event1
    events2 = df[(df['event_name_2'].notna())]
    ts = events2.loc[:,'event_name_2']
    events2.loc[:,'event_name_1'] = ts
    ts = events2.loc[:,'event_type_1']
    events2.loc[:,'event_type_2'] = ts
    df = df.append(events2, ignore_index=True)
    df = df.drop(['event_name_2', 'event_type_2'], axis='columns')
    
    
    df_month = df.loc[df['month'].isin(months), :]
    df_events_test = df.loc[df['event_name_1'].isin(events_test), :]
    df_events = df.loc[df['event_name_1'].notna(), :]
    
    for i in range(rep_month):
          df = df.append(df_month,ignore_index=True)
    

    for i in range(rep_event):
          df = df.append(df_events,ignore_index=True)
    

    for i in range(rep_event_test):
          df = df.append(df_events_test,ignore_index=True)
    
    df[['event_name_1', 'event_type_1']] = df[['event_name_1', 'event_type_1']].astype('category')
    return df

#print(X_train.shape)
#X_train = repeat_data(X_train)
#print(X_train.shape)




# Additional features
def addTrendFeatures(df):

  df.loc[df['Lag1_rmean7_id'] > 0, 'Lag1_rmean1_id_div_Lag1_rmean7_id'] = df['Lag1_rmean1_id']/df['Lag1_rmean7_id']
  df.loc[(df['Lag1_rmean7_id'] == 0) & (df['Lag1_rmean1_id'] > 0), 'Lag1_rmean1_id_div_Lag1_rmean7_id'] = df['Lag1_rmean1_id']
  df.loc[(df['Lag1_rmean7_id'] == 0) & (df['Lag1_rmean1_id'] == 0), 'Lag1_rmean1_id_div_Lag1_rmean7_id'] = -1

  df.loc[df['Lag1_rmean28_id'] > 0, 'Lag1_rmean7_id_div_Lag1_rmean28_id'] = df['Lag1_rmean7_id']/df['Lag1_rmean28_id']
  df.loc[(df['Lag1_rmean28_id'] == 0) & (df['Lag1_rmean7_id'] > 0), 'Lag1_rmean7_id_div_Lag1_rmean28_id'] = df['Lag1_rmean7_id']
  df.loc[(df['Lag1_rmean28_id'] == 0) & (df['Lag1_rmean7_id'] == 0), 'Lag1_rmean7_id_div_Lag1_rmean28_id'] = -1
  df['Lag1_rmean1_id_div_Lag1_rmean7_id'] = df['Lag1_rmean1_id_div_Lag1_rmean7_id'].astype('float32')
  df['Lag1_rmean7_id_div_Lag1_rmean28_id'] = df['Lag1_rmean7_id_div_Lag1_rmean28_id'].astype('float32')

  print(df.head())
  return df

def addDateTypeAvg(df, cols):
    # The total dataset average over the grouping given columns in cols
    print(f'Calculating item average over {cols} grouping')
        
    df[f'{cols}_Avg'] = df.groupby(cols)['saleCount'].transform(lambda x : np.round(x.mean(),2))

    return df
#print(X_train.info())

#X_train = addDateTypeAvg(X_train, ['id','event_name_1'])
#X_train = addDateTypeAvg(X_train, ['item_id','event_name_1'])
#X_train["['id', 'event_name_1']_Avg"] = X_train["['id', 'event_name_1']_Avg"].astype('float32')
#X_train["['item_id', 'event_name_1']_Avg"] = X_train["['item_id', 'event_name_1']_Avg"].astype('float32')
#X_train = addTrendFeatures(X_train)
#print(X_train.info())

#X_train.head()





best_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
             'wday', 'month', 'year', 'event_name_1', 'event_type_1',
             'snap_CA', 'snap_TX', 'snap_WI', 'sell_price',
             'available',
             'Lag1_rmean1_id', 'Lag7_rmean1_id', 'Lag28_rmean1_id',
             'Lag1_rmean7_id', 'Lag7_rmean7_id', 'Lag28_rmean7_id',
             'Lag1_rmean28_id', 'Lag28_rmean28_id',
             "id_wday_Avg", "id_month_Avg",
             "item_id_wday_Avg", "item_id_month_Avg"]

cols_drop = ['event_name_2', 'event_type_2','available', 'cat_id','state_id', 'dept_id', 'year']

X_train = X_train.drop(cols_drop, axis='columns')
X_train.info()




cols_ignore = ['id', 'd', 'saleCount', 'type',
              'event_name_2', 'event_type_2', 'available']

col_feature = [i 
               for i in X_train.columns
               if i not in cols_ignore]

# VAlidation data mask
mask_val = (X_train['d'].between(1913 - 365, 1913-365 +28)) | (X_train['d'].between(1913-28, 1913))
mask_train = X_train['type']=='train'

print('Feature columns: \n',col_feature)

# print(X_train[col_feature].info())
cat_cols = X_train[col_feature].select_dtypes(include=['category']).columns
for c in X_train.select_dtypes(include=['category']).columns:
    if c not in ['id','type']:
        X_train.loc[:,c] = X_train.loc[:,c].cat.codes

X_train.info()




X_train[(X_train['id'] == 'FOODS_3_318_CA_1_validation') &  mask_train]




# dropping NAN values on price
def repNAN(df, cols, val):
    print('Setting NAN values to {}'.format(val))
    for c in cols:
        print(f'Col. {c} contains {df[c].isna().sum()} NAN')
        df.loc[df[c].isna(), c] = val
        
    return df

cols_rep = ['Lag1_rmean1_id', 'Lag7_rmean1_id', 'Lag28_rmean1_id',
            'Lag1_rmean7_id', 'Lag7_rmean7_id', 'Lag28_rmean7_id',
            'Lag1_rmean28_id', 'Lag28_rmean28_id']

#X_train = repNAN(X_train, cols_rep, -1)
X_train = repNAN(X_train, ['sell_price'], -1)
X_train = repNAN(X_train, ['saleCount'], 0)




# Basic neural network
def run_nn():
    n_feat = X_train[col_feature].shape[1]
    n_items = X_train['id'].nunique()
    print(n_feat, n_items)
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_feat,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_feat*3, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(n_feat*2, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(rate=0.8),
        tf.keras.layers.Dense(n_feat*1, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation=None)])

    model.summary()


    model.compile(optimizer="Adam", loss="mse")
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='mse', mode='auto',min_delta=0.01, patience=10)
    
    t0 = time.time()
#     history = model.fit(X_train.loc[~mask_val & mask_train, col_feature],
#                         X_train.loc[~mask_val & mask_train, 'saleCount'],
#                         batch_size=n_items,
#                         epochs=30,
#                         callbacks=[callback],
#                         validation_data=(X_train.loc[mask_val & mask_train, col_feature],
#                                          X_train.loc[mask_val & mask_train, 'saleCount']),
#                         validation_freq=1
#                        )
    history = model.fit(X_train.loc[mask_train, col_feature],
                        X_train.loc[mask_train, 'saleCount'],
                        batch_size=n_items,
                        epochs=30)
    plt.figure()
    plt.plot(history.history['loss'],label='Training')
    #plt.plot(history.history['val_loss'],label='Validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    return model

#model = run_nn()

# predictions = model.predict(X_train.loc[(X_train['id'] == 'FOODS_3_318_CA_1_validation') &  mask_train, col_feature][-50:])

# for i, j in zip(predictions, X_train.loc[(X_train['id'] == 'FOODS_3_318_CA_1_validation') &  mask_train, 'saleCount'][-50:]):
#     print(i,j)




# XGBoost model
def run_xgboost(max_depth=6,n_estimators=1000,min_child_weight=1, learning_rate=0.01, gamma=0, subsample=1,colsample_bytree = 1):
    
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma =  gamma,
        subsample=subsample,
        colsample_bytree = colsample_bytree,
        random_state =1,
        verbosity=2,
     #   tree_method='gpu_hist'
    ) 
    
    model.fit(X_train.loc[mask_train, col_feature],
              X_train.loc[mask_train, 'saleCount'],
#               eval_set=[(X_train.loc[mask_val & mask_train, col_feature], X_train.loc[mask_val & mask_train, 'saleCount'])],
#               eval_metric='rmse',
#               early_stopping_rounds=10,
              verbose=True)
        
#     model.fit(X_train.loc[~mask_val & mask_train, col_feature],
#               X_train.loc[~mask_val & mask_train, 'saleCount'],
# #               eval_set=[(X_train.loc[mask_val & mask_train, col_feature], X_train.loc[mask_val & mask_train, 'saleCount'])],
# #               eval_metric='rmse',
# #               early_stopping_rounds=10,
#               verbose=True)
    
    return model




# Sklearn randomforest
def run_rf(n_range=1):
  score_train = []
  score_val = []
  train_time = []

# USe the categorical codes
  for c in X_train.select_dtypes(include=['category']).columns:
      if c not in ['id','type']:
          X_train.loc[:,c] = X_train.loc[:,c].cat.codes

  print(f'training with trees {n_range}')
  for n in n_range:
    model = RandomForestRegressor(n_estimators = n,
                              max_depth = None,
                              random_state=1,
                              n_jobs = -1,# Use all available cores
                              bootstrap = True,
                              max_features='sqrt',
                              min_samples_leaf = 10,
                              verbose=10)
    
    t0 = time.time()
    print('\n',f'Training RF with n={n} initiated at {time.ctime(t0)}')
    model.fit(X_train.loc[~mask_val & mask_train, col_feature], X_train.loc[~mask_val & mask_train, 'saleCount'])
    dt = (time.time() - t0)/60
    train_time.append(dt)
    print('Tree depths: ', [estimator.get_depth() for estimator in model.estimators_])
    score_train.append(model.score(X_train.loc[~mask_val & mask_train, col_feature], X_train.loc[~mask_val & mask_train, 'saleCount']))
    score_val.append(model.score(X_train.loc[mask_val & mask_train, col_feature], X_train.loc[mask_val & mask_train, 'saleCount']))
    
    #rmse(model)
    for i, (t, v) in enumerate(zip(score_train, score_val)):
      print(f'{n_range[i]} trees (training, validation): ({np.round(t,2)}, {np.round(v,2)}), {np.round(train_time[i],3)}')

    if n != n_range[-1]:
      del model
      gc.collect()
    
    return model




def run_cb_val():
  from catboost import Pool
  model = CatBoostRegressor(iterations=2000,
                            learning_rate=0.05,
                            depth=16,
                            random_strength = 0,
                            l2_leaf_reg = 3,
                            min_data_in_leaf = 1,
                            boosting_type='Plain',
                            border_count=256,
                            task_type='GPU',
                          #logging_level='Verbose',
                            random_state=1,
                            one_hot_max_size = 1)

# USe the categorical codes
  cat_cols = X_train[col_feature].select_dtypes(include=['category']).columns
  print('Converting categorical datatype to object')
  print(X_train[col_feature].columns)
  # USe the categorical codes
  # cat_idx = []
  # for c in cat_cols:
  #   print(c)
  #   X_train.loc[:,c] = X_train.loc[:,c].astype('str')
  #   cat_idx = np.append(cat_idx, X_train.columns.get_loc(c))
  # cat_idx = [0,1,2,3,4,5,6]
  cat_idx = []

  # for c in X_train.select_dtypes(include=['category']).columns:
  #   if c not in ['id','type']:
  #     X_train.loc[:,c] = X_train.loc[:,c].cat.codes
  # print('Data pool')
  # train_data = Pool(
  #     data=X_train.loc[~mask_val & mask_train, col_feature],
  #     label=X_train.loc[~mask_val & mask_train, 'saleCount'],
  #     cat_features = cat_idx)
  
  # val_data = Pool(
  #     data=X_train.loc[mask_val & mask_train, col_feature],
  #     label=X_train.loc[mask_val & mask_train, 'saleCount'],
  #     cat_features = cat_idx)
  # print('Model fit')
  # model.fit(train_data,
  #           eval_set = val_data,
  #           early_stopping_rounds = 10)

  model.fit(X_train.loc[~mask_val & mask_train, col_feature],
            X_train.loc[~mask_val & mask_train, 'saleCount'],
            eval_set = (X_train.loc[mask_val & mask_train, col_feature], 
                        X_train.loc[mask_val & mask_train, 'saleCount']),
            early_stopping_rounds = 10,
            cat_features = cat_idx, plot=True)

  # rmse(model)
  # feat_imp(model)

  return model

def run_cb_train():
    model = CatBoostRegressor(iterations=1000,
                          learning_rate=0.05,
                          depth=16,
                          random_strength = 0,
                          l2_leaf_reg = 5,
                          min_data_in_leaf = 1,
                          boosting_type='Plain',
                          border_count=512,
                          task_type='GPU',
                          logging_level='Verbose',
                          random_state=1,
                          one_hot_max_size = 1
                              )

# USe the categorical codes
#     print('Converting categorical datatype to object')
#     #XT = X_train.loc[mask_train, col_feature]
#     #yt = X_train.loc[mask_train, 'saleCount']
#     cat_cols = X_train.loc[mask_train, col_feature].select_dtypes(include=['category']).columns
#     cat_idx = []
    
#     for c in cat_cols:
#         print(f'Converting column {c}')
#         X_train[c] = X_train[c].cat.codes
#         if (X_train[c] == -1).sum() > 0:
#             next_cat = max(X_train[c].unique())
#             print(f'Setting cat=-1 to {next_cat}')
#             X_train.loc[X_train[c]==-1, c] = next_cat
            
#         cat_idx.append(X_train.loc[mask_train, col_feature].columns.get_loc(c))
        
#     print(cat_idx)
    
#     print(X_train.loc[mask_train, col_feature].isna().sum())
    
#     dat = Pool(
#         data=X_train.loc[mask_train, col_feature],
#         label=X_train.loc[mask_train, 'saleCount'],
#         cat_features = cat_idx)
  
    print('Fitting model')
    model.fit(X_train.loc[mask_train, col_feature],
              X_train.loc[mask_train, 'saleCount'])
  
    #rmse(model)

    return model




def feat_imp(model):
  feature_importance = model.get_feature_importance(prettified=True)
  fig = plt.figure()
  plt.bar(feature_importance.iloc[:,0], feature_importance.iloc[:,1])
  plt.ylabel('Feature importance')
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('feature.pdf')
  plt.show()


def rmse(model):
    pred_train = model.predict(X_train.loc[~mask_val & mask_train, col_feature])
    pred_val = model.predict(X_train.loc[mask_val & mask_train, col_feature])

    diff_train = X_train.loc[~mask_val & mask_train, 'saleCount'] - pred_train
    diff_val = X_train.loc[mask_val & mask_train, 'saleCount'] - pred_val

    rmse_train = (1/(len(pred_train)) * np.matmul(diff_train,diff_train))**(1/2)
    rmse_val = (1/(len(pred_val))*np.matmul(diff_val, diff_val))**(1/2)
    print('RMSE (train, val): ', rmse_train, rmse_val)




# RMSE (train, val):  0.0004321664752965253 0.0014114955018167845
# 1055:	learn: 1.6697889	test: 1.8760822	best: 1.8760080 (1030)	total: 7m 46s	remaining: 6m 56s
# With ID
#757:	learn: 1.8419379	test: 1.8783848	best: 1.8782903 (747)	total: 10m 38s	remaining: 17m 26
# RMSE (train, val):  0.0003278893567667356 0.001412440601992821

#X_train.loc[mask_train, col_feature].info()

#model = run_cb_val() 

model = run_cb_train()

# print('Training model)')
# model = run_xgboost(max_depth=10,n_estimators=150,min_child_weight=50, learning_rate=0.1, gamma=0, subsample=1,colsample_bytree = 0.4)




feat_imp(model)




# Generate test set features
def addLaggedRollingAvg(df, grp_cols, val_col, windows, lag):
    # The index is shifted by one, in order to use the previous win days
    # and not the actual day where the feature is assigned
    # It just requires 1 value to make a rolling avg, venthough the windows is perhaps 30
    for w in windows:
      for l in lag:
        #print(f'Calculating item rolling average over {w} days at lag {l}')
        col_name = f'Lag{l}_rmean{w}_{grp_cols}'
        df[col_name] = df.groupby(grp_cols)[val_col]                         .transform(lambda x : np.round(x.shift(l).rolling(w, min_periods=1).mean(),2))

    return df

X_test = X_train.loc[X_train['d'] >= 1913-(28+1), :]

d_min_val = X_train.loc[X_train['type'] == 'val', 'd'].min()
d_max_val = X_train.loc[X_train['type'] == 'val', 'd'].max()

X_test.loc[X_test['id'] == 'FOODS_3_318_CA_3_validation', :].head(100)




for d in range(d_min_val,d_max_val+1):
    print(f'Day {d}')
    X_test = addLaggedRollingAvg(X_test, grp_cols='id', val_col = 'saleCount', windows=[1],  lag = [1,7])
    X_test = addLaggedRollingAvg(X_test, grp_cols='id', val_col = 'saleCount', windows=[7],  lag = [1, 7])
    X_test = addLaggedRollingAvg(X_test, grp_cols='id', val_col = 'saleCount', windows=[28], lag = [1])
    #X_test = addTrendFeatures(X_test)
    y_pred = model.predict(X_test.loc[X_test['d'] == d, col_feature])
    
    print('Negative sale predictions: ', sum(y_pred < 0))
    y_pred[y_pred < 0] = 0
    
    X_test.loc[X_test['d'] == d, 'saleCount'] = y_pred
        
X_test.loc[(X_test['id'] == 'FOODS_3_318_CA_3_validation') & (X_test['type'] == 'val'), :].head(30)




X_test[(X_test['d'] == d) & (X_test['saleCount'] < 0)]['d'].count()




# negative salecount predictions
neg_sales = X_test.loc[(X_test['saleCount'] <0) & (X_test['type'] == 'val'), :]
print('Negative sale predictions: ', len(neg_sales))
neg_sales.head()




print('Generating forecast')
# print(X_test.loc[(X_test['item_id'] == 1437 )& (X_test['store_id'] == 0), col_feature])
# y_pred = model.predict(X_test[col_feature])

X_forecast = X_test.loc[X_test['type']=='val',['id', 'd', 'saleCount']].copy()

#set negative sales to 0
X_forecast.loc[(X_forecast['saleCount'] < 0), 'saleCount'] = 0

# Round forecast to 3 decimals
X_forecast.loc[:,'saleCount'] = np.round(X_forecast.loc[:,'saleCount'],3)

days = X_forecast['d'].unique()
n_days = len(days)
day_min = min(days)

print(days)
assert n_days == 28, 'Wrong number of forecastet days'

day_1 = days.min()

X_forecast['d'] = 'F'+ X_forecast['d'].add(- day_min + 1).astype('str')
print('Done')





df_sub = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
#df_sub = pd.read_csv('/content/drive/My Drive/Colab Notebooks/A1/sample_submission.csv')

fcols = [f for f in df_sub.columns if 'F' in f]
    
for f in fcols:
    X_forecast_f = X_forecast[X_forecast['d'] == f]
    df_sub[f] = df_sub.merge(X_forecast_f,
              how = 'right',
              left_on = 'id',
              right_on = 'id')['saleCount']
        
df_sub.fillna(0)
#df_sub.to_csv('/content/drive/My Drive/Colab Notebooks/A1/submission_cb_3y_events.csv', index=False, na_rep=0)
df_sub.to_csv('submission.csv', index=False, na_rep=0)
print('Done')

