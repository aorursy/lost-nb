#!/usr/bin/env python
# coding: utf-8



import numpy as np # mathematical library including linear algebra
import pandas as pd #data processing and CSV file input / output

import xgboost as xgb # this is the extreme gradient boosting library
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing 
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')




df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])





# Create a vector containing the id's for our predictions
id_test = df_test.id

#Create a vector of the target variables in the training set
# Transform target variable so that loss function is correct (ie we use RMSE on transormed to get RMLSE)
# ylog1p_train_whole will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
ylog1p_train_all = np.log1p(df_train['price_doc'].values)
df_train = df_train.drop(["price_doc"], axis=1)

# Create joint train and test set to make data wrangling quicker and consistent on train and test
df_train["trainOrTest"] = "train"
df_test["trainOrTest"] = "test"
num_train = len(df_train)

df_all = pd.concat([df_train, df_test])
del df_train
del df_test

# Removing the id (could it be a useful source of leakage?)
df_all = df_all.drop("id", axis=1)




# Convert the date into a number (of days since some point)
fromDate = min(df_all['timestamp'])
df_all['timedelta'] = (df_all['timestamp'] - fromDate).dt.days.astype(int)
print(df_all[['timestamp', 'timedelta']].head())




# Add month-year count - i.e. how many sales in the month 
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)




for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))




# Alternative using the rather nice .select_dtypes
# df_numeric = df_all.select_dtypes(exclude=['object'])
# df_obj = df_all.select_dtypes(include=['object']).copy()

# for c in df_obj:
#    df_obj[c] = pd.factorize(df_obj[c])[0]

# df_all = pd.concat([df_numeric, df_obj], axis=1)




# Fill missing values with  -1.  
df_all.fillna(-1, inplace = True)




# Convert to numpy values
X_all = df_all.values
print(X_all.shape)

# Create a validation set, with last 20% of data
num_val = int(num_train * 0.2)

#X_train_all  = X_all[:num_train]
X_train      = X_all[:num_train-num_val]
X_val        = X_all[num_train-num_val:num_train]
ylog1p_train = ylog1p_train_all[:-num_val]
ylog1p_val   = ylog1p_train_all[-num_val:]

X_test = X_all[num_train:]

df_columns = df_all.columns

del df_all

#print('X_train_all shape is', X_train_all.shape)
print('X_train shape is',     X_train.shape)
print('y_train shape is',     ylog1p_train.shape)
print('X_val shape is',       X_val.shape)
print('y_val shape is',       ylog1p_val.shape)
print('X_test shape is',      X_test.shape)




##dtrain_all = xgb.DMatrix(X_train_all, ylog1p_train_all, feature_names = df_columns)
#dtrain     = xgb.DMatrix(X_train,     ylog1p_train,     feature_names = df_columns)
#dval       = xgb.DMatrix(X_val,       ylog1p_val,       feature_names = df_columns)
dtest      = xgb.DMatrix(X_test,                        feature_names = df_columns)




# Choose values for the key parameters - keep the number of estimators low for now - not more than 200

model = xgb.XGBRegressor(    objective = 'reg:linear'
                           , n_estimators =  
                           , max_depth = 5
                           # , min_child_weight = min_child_weight
                           , subsample = 1.0
                           , colsample_bytree = 
                           , learning_rate = 
                           , silent = 1)




eval_set  = [( X_train, ylog1p_train), ( X_val, ylog1p_val)]

model.fit(X = X_train, 
          y = ylog1p_train,
          eval_set = eval_set, 
          eval_metric = "rmse", 
          early_stopping_rounds = 30,
          verbose = True)




num_boost_round = model.best_iteration 
num_boost_round

# Is num_boost_rounds one less than the n_estimators you chose above?  If it is 
# what does this tell you?  What should you do about it?




# Fill eta below with whatever you used for learning_rate above.
# Likewise for colsample_bytree

# Different syntax used here than above, due to issues with xgboost package (we can't get 
# variable importance the other way)

xgb_params = {
    'eta': ,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': ,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}

dtrain_all = xgb.DMatrix(np.vstack((X_train, X_val)), ylog1p_train_all, feature_names = df_columns)
model = xgb.train(xgb_params, dtrain_all, num_boost_round = num_boost_round)




# Create a dataframe of the variable importances
dict_varImp = model.get_score(importance_type = 'weight')
df_ = pd.DataFrame(dict_varImp, index = ['varImp']).transpose().reset_index()
df_.columns = ['feature', 'fscore']




# Plot the relative importance of the top 10 features
df_['fscore'] = df_['fscore'] / df_['fscore'].max()
df_.sort_values('fscore', ascending = False, inplace = True)
df_ = df_[0:10]
df_.sort_values('fscore', ascending = True, inplace = True)
df_.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('xgboost feature importance', fontsize = 24)
plt.xlabel('')
plt.ylabel('')
plt.xticks([], [])
plt.yticks(fontsize=20)
plt.show()
#plt.gcf().savefig('feature_importance_xgb.png')




### 4b. Partial dependence plots




# from https://xiaoxiaowang87.github.io/monotonicity_constraint/
def partial_dependency(bst, X, y, feature_ids = [], f_id = -1):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """

    X_temp = X.copy()

    grid = np.linspace(np.percentile(X_temp[:, f_id], 0.1),
                       np.percentile(X_temp[:, f_id], 99.5),
                       50)
    y_pred = np.zeros(len(grid))

    if len(feature_ids) == 0 or f_id == -1:
        print ('Input error!')
        return
    else:
        for i, val in enumerate(grid):

            X_temp[:, f_id] = val
            data = xgb.DMatrix(X_temp, feature_names = df_columns)

            y_pred[i] = np.average(bst.predict(data))

    return grid, y_pred




lst_f = ['full_sq', 'timedelta', 'floor']
for f in lst_f:
    f_id = df_columns.tolist().index(f)


    feature_ids = range(X_train.shape[1])

    grid, y_pred = partial_dependency(model,
                                      X_train,
                                      ylog1p_train,
                                      feature_ids = feature_ids,
                                      f_id = f_id
                                      )

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.subplots_adjust(left = 0.17, right = 0.94, bottom = 0.15, top = 0.9)

    ax.plot(grid, y_pred, '-', color = 'red', linewidth = 2.5, label='fit')
    ax.plot(X_train[:, f_id], ylog1p_train, 'o', color = 'grey', alpha = 0.01)

    ax.set_xlim(min(grid), max(grid))
    ax.set_ylim(0.95 * min(y_pred), 1.05 * max(y_pred))

    ax.set_xlabel(f, fontsize = 24)
    ax.set_ylabel('Partial Dependence', fontsize = 24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()




## 5. Create the predictions




# Create the predictions
ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1




output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
output.to_csv('xgb_1.csv', index=False)

