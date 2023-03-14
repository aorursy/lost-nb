#!/usr/bin/env python
# coding: utf-8



# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
#plotting directly without requering the plot()

import warnings
warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization

pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed
pd.set_option('display.max_rows', 500)

print(os.listdir("../input")) #showing all the files in the ../input directory

# Any results you write to the current directory are saved as output. Kaggle message :D




train = pd.read_csv('../input/train.csv', parse_dates=True, infer_datetime_format=True, dayfirst=True)
test = pd.read_csv('../input/test.csv', parse_dates=True, infer_datetime_format=True, dayfirst=True)




print('----- DETAILS of our DATASET -----')
print('Training shape: %s' %str(train.shape))
print('Testing shape: %s' %str(test.shape))
print('First date entry for the training dataset: %s' %train.index.min())
print('Last date entry for the training dataset: %s' %train.index.max())
print('First date entry for the testing dataset: %s' %test['date'].min())
print('Last date entry for the testing dataset: %s' %test['date'].max())
print('Number of unique items, training set: %s' %train['item'].nunique())
print('Number of unique items, testing set: %s' %test['item'].nunique())
train.head()




# Concatenating train & test
train['split'] = 'train'
test['split'] = 'test'
df_total = pd.concat([train,test], sort=False)
print('Total shape: {}'.format(df_total.shape))
del train, test
gc.collect()
df_total.head()




df_total['date'] = pd.to_datetime(df_total['date'], dayfirst=True,infer_datetime_format=True)
df_total['month'] = df_total['date'].dt.month
df_total['year'] = df_total['date'].dt.year
df_total['weekday'] = df_total['date'].dt.weekday
df_total['day_of_month'] = df_total['date'].dt.day
df_total['day_of_year'] = df_total['date'].dt.dayofyear
df_total.tail()




#sns.set_style('ticks') #setting the style for our plots
plt.style.use('dark_background') #another style,this one is from the matplotlib

fig = plt.figure(figsize=(16,14))

plt.subplot(5,1,1)
sns.lineplot(y='sales', x='year', data=df_total, label='Sales');plt.title('Sales per year')
plt.subplot(5,1,2)
sns.lineplot(y='sales', x='month', data=df_total, label='Sales');plt.title('Sales per month')
plt.subplot(5,1,3)
sns.lineplot(y='sales', x='weekday', data=df_total, label='Sales');plt.title('Sales per weekday')
plt.subplot(5,1,4)
sns.barplot(x=df_total['store'], y=df_total['sales'], errwidth=0,palette="PuBuGn_d");plt.title('Sales distribution across stores')
plt.subplot(5,1,5)
sns.barplot(x=df_total['item'], y=df_total['sales'], errwidth=0,palette="PuBuGn_d");plt.title('Sales distribution across items')
plt.tight_layout(h_pad=2.5)




# taking the log1+ of our target(sales)
df_total['sales'] = np.log1p(df_total.sales.values)




fig = plt.figure(figsize=(16,5))
df_total.set_index('date')['sales'].plot(); plt.title('Distribution of sales log1p')




val_months = (df_total.year==2017) & (df_total.month.isin([1,2,3]))
skip = (df_total.year==2017) & (~val_months)
df_total.loc[(val_months), 'split'] = 'val'
df_total.loc[(skip), 'split'] = 'skip'
train = df_total.loc[df_total.split.isin(['train','val','test']), :]
train_labels = train.loc[train['split'] == 'train', 'sales'].values.reshape((-1))
train_labels_val = train.loc[train['split'] == 'val', 'sales'].values.reshape((-1))
print('Shape Training set: %s' %str(train[train['split']== 'train'].shape))
print('Shape Validation set: %s' %str(train[train['split']== 'val'].shape))
print('Shape Testing set: %s' %str(train[train['split']== 'test'].shape))
print('train labels(Y): %s' %str(train_labels.shape))
print('train labels for the validation set(Y_val): %s' %str(train_labels_val.shape))




fig = plt.figure(figsize=(16,5))
df_total.set_index('date', inplace=True)
df_total.loc[df_total['split'] == 'train', 'sales'].plot(c='r', label='train')
df_total.loc[df_total['split'] == 'val', 'sales'].plot(c='b', label='validation')
df_total.loc[df_total['split'] == 'skip', 'sales'].plot(c='g', label='skip')
plt.title('Log 1+ Sales, training, skipped part and validation set.')
plt.legend()




lags=[91,98,105,112,119,126,182,364,546,728] #lag for previous months, 91 days until 728 days
alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5] #It's gonna be used for the exponential weighted moving average

train.loc[train['split'] == 'val', 'sales'] = np.nan 
#setting all the sales validation which is gonna be used for our transformation below, 
##this is gonna help our model generalize better
####we have saved our validation labels before, so it's ok

temp_groupby = train.groupby(['store','item'])

### creating lag features for the number of days in lag ###
for i in lags:
    train['_'.join(['sales', 'lag', str(i)])] =                 temp_groupby['sales'].shift(i).values + np.random.normal(scale=1.6, size=(len(train),)) #normalizing term

### creating the exponential weighted average, using the same days as in lag ###
for a in alpha:
    for i in lags:
        train['_'.join(['sales', 'lag', str(i), 'ewm', str(a)])] =             temp_groupby['sales'].shift(i).ewm(alpha=a).mean().values

### creating the rolling mean for 1 year and 1 year and half ###
for w in [364,546]:
    train['_'.join(['sales', 'rmean', str(w)])] =             temp_groupby['sales'].shift(1).rolling(window=w, 
                                                  min_periods=10,
                                                  win_type='triang').mean().values +\
            np.random.normal(scale=1.6, size=(len(train),)) #normalizing term

del temp_groupby #--cleaning the memory up--
gc.collect()




train.head()




fig = plt.figure(figsize=(16,5))
plt.subplot(2,1,1)
sns.lineplot(y='sales_lag_364_ewm_0.95', x='month',  data=train, label="12M EWM");
sns.lineplot(y='sales', x='month',  data=train, label="Normal Sales");
sns.lineplot(y='sales_lag_364', x='month',  data=train, label="sales 12M lag");
plt.legend()
plt.subplot(2,1,2)
sns.lineplot(y='sales_rmean_364', x='month',  data=train, label="12M mean");
sns.lineplot(y='sales', x='month',  data=train, label="Normal Sales");
sns.lineplot(y='sales_rmean_546', x='month',  data=train, label="sales 20M mean");
plt.tight_layout(h_pad=1.5)




train = pd.get_dummies(train, columns=['store','item','day_of_month','weekday','month'])
print('Shape after creating the dummies: {}'.format(train.shape))




# Final train, validation and testing datasets
train_val = train.loc[train.split=='val', :]
train_X = train.loc[train.split=='train', :]
test_sub = train.loc[train.split == 'test',:]
print('Training shape:{}, Validation shape:{}, Labels X: {}, Labels Validation: {}, Testing shape: {}'
      .format(train_X.shape, train_val.shape,train_labels.shape,train_labels_val.shape, test_sub.shape))




import time #implementing in this function the time spent on training the model
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
import gc

####### FUNCTIONS FOR CALCULATING THE SMAPE LOSS ##########
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds==0)&(target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def train_model(X, X_val, y, y_val, params=None, model_type='lgb', plot_feature_importance=False, model=None):

    mask = ['date', 'sales', 'split', 'id', 'year']
    cols = [col for col in train.columns if col not in mask]
    evals_result={}
    if model_type == 'lgb':
        start = time.time()
        X_train = lgb.Dataset(data=X.loc[:,cols].values, label=y, 
                       feature_name=cols)
        
        X_valid = lgb.Dataset(data=X_val.loc[:,cols].values, label=y_val, 
                     reference=X_train, feature_name=cols)
        
        model = lgb.train(params, X_train, num_boost_round=params['num_boost_round'], 
                      valid_sets=[X_train, X_valid], feval=lgbm_smape, 
                      early_stopping_rounds=params['early_stopping_rounds'], 
                      evals_result=evals_result, verbose_eval=500)
            
        y_pred_valid = model.predict(X_val.loc[:, cols].values, num_iteration=model.best_iteration)
        y_pred_valid = np.expm1(y_pred_valid)
        y_val = np.expm1(y_val)
        
        end = time.time()
        
        #y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        print('SMAPE validation data: {}'.format(smape(y_pred_valid, y_val)))
        
        if plot_feature_importance:
            # feature importance
            fig, ax = plt.subplots(figsize=(12,10))
            lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
            ax.grid(False)
            plt.title("LightGBM - Feature Importance", fontsize=15)
            
        print('Total time spent: {}'.format(end-start))
        return model
            
    if model_type == 'xgb':
        start = time.time()
        train_data = xgb.DMatrix(data=X.loc[:,cols].values, label=y)
        valid_data = xgb.DMatrix(data=X_val.loc[:,cols].values, label=y_val)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, 
                          early_stopping_rounds=200, verbose_eval=500, params=params)
        
        y_pred_valid = model.predict(xgb.DMatrix(X_val.loc[:,cols].values), ntree_limit=model.best_ntree_limit)
        
        end = time.time()

        print('SMAPE validation data: {}'.format(smape(y_pred_valid, y_val)))
        
        print('Total time spent: {}'.format(end-start))
        return model
            
    if model_type == 'cat':
        start = time.time()
        model = CatBoostRegressor(eval_metric='MAE', **params)
        model.fit(X.loc[:,cols].values, y, eval_set=(X_val.loc[:,cols].values, y_val), 
                  cat_features=[], use_best_model=True)

        y_pred_valid = model.predict(X_val.loc[:,cols].values)
        
        print('SMAPE validation data: {}'.format(smape(y_pred_valid, y_val)))
        
        if plot_feature_importance:
            feature_score = pd.DataFrame(list(zip(X.loc[:,cols].dtypes.index, 
                                                  model.get_feature_importance(Pool(X.loc[:,cols], label=y, cat_features=[])))), columns=['Feature','Score'])
            feature_score = feature_score.sort_values(by='Score', kind='quicksort', na_position='last')
            feature_score[:50].plot('Feature', 'Score', kind='barh', color='c', figsize=(12,10))
            plt.title("Catboost Feature Importance plot", fontsize = 14)
            plt.xlabel('')
        
        end = time.time()

        print('Total time spent: {}'.format(end-start))
        return model
        
    # Clean up memory
    gc.enable()
    del model, y_pred_valid, X_test,X_train,X_valid, y_pred, y_train, start, end,evals_result
    gc.collect()




params = {
          'num_leaves': 10,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 6,
         'learning_rate': 0.02,
        'num_boost_round': 25000, 
        'early_stopping_rounds':200,
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         'metric': 'mae',
         "lambda_l1": 0.2,
}
lgb_model = train_model(train_X,train_val,train_labels,train_labels_val,params, plot_feature_importance=True)




params_cat = {
    'iterations': 2000,
    'max_ctr_complexity': 6,
    'random_seed': 42,
    'od_type': 'Iter',
    'od_wait': 50,
    'verbose': 50,
    'depth': 4
}

cat_model = train_model(train_X,train_val,train_labels,train_labels_val,params_cat,model_type='cat', plot_feature_importance=True)




mask = ['date', 'sales', 'split', 'id', 'year']
cols = [col for col in train.columns if col not in mask]
predict_lgb = lgb_model.predict(test_sub.loc[:,cols].values, num_iteration=lgb_model.best_iteration)
predict_cat = cat_model.predict(test_sub.loc[:,cols].values)
predict_avg = (predict_lgb+predict_cat)/2
sub = pd.read_csv('../input/sample_submission.csv')
sub['sales'] = np.expm1(predict_avg)
sub.to_csv('lgb_model.csv', index=False)
sub.head()

