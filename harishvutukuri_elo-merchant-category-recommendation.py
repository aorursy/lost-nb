#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (8,6)
plt.style.use('fivethirtyeight')




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




train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])




train.shape, test.shape




train.head()




target_col = 'target'

plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title("Loyalty score on target")
plt.show()




sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()




train[train['target']<-30]['target'].count()




cnt_srs = train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()


cnt_srs = test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()




# feature 1

sns.violinplot(x="feature_1", y=target_col, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()

# feature 2

sns.violinplot(x="feature_2", y=target_col, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

# feature 3

sns.violinplot(x="feature_3", y=target_col, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()




import datetime
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days




train['month'] = train.first_active_month.dt.month
train['year'] = train.first_active_month.dt.year
test['month'] = test.first_active_month.dt.month
test['year'] = test.first_active_month.dt.year




import gc
gc.collect()




def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df




holidays = [
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
    ]

def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday) - df[date_ref]).dt.days, period), 0)




historical = pd.read_csv("../input/historical_transactions.csv", parse_dates=['purchase_date'])
historical = binarize(historical)
historical = pd.get_dummies(historical, columns=['category_2', 'category_3'])
historical = reduce_mem_usage(historical)




gdf = historical.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "historical_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")




cnt_srs = train.groupby("historical_transactions")['target'].mean()
cnt_srs = cnt_srs.sort_index()
cnt_srs = cnt_srs[:-50]

sns.scatterplot(data=cnt_srs)
plt.title('Loyalty score by Number of historical transactions')
plt.show()




for d_name, d_day in holidays:
    dist_holiday(historical, d_name, d_day, 'purchase_date')




agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['sum', 'mean'],
        'category_2_2.0': ['sum', 'mean'],
        'category_2_3.0': ['sum', 'mean'],
        'category_2_4.0': ['sum', 'mean'],
        'category_2_5.0': ['sum', 'mean'],
        'category_3_A': ['sum', 'mean'],
        'category_3_B': ['sum', 'mean'],
        'category_3_C': ['sum', 'mean'],
        'authorized_flag': ['nunique', 'mean', 'sum'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std','skew'],
        'installments': ['sum', 'mean', 'max', 'min', 'std', 'skew'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Mothers_Day_2017': ['mean', 'sum'],
        'fathers_day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Valentine_Day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum']
        }




# Adding more features from historical transactions

historical.loc[:, 'purchase_date'] = pd.DatetimeIndex(historical['purchase_date']).                                      astype(np.int64) * 1e-9
gdf = historical.groupby("card_id").agg(agg_func)
gdf.columns = ['_historical_'.join(col).strip() for col in gdf.columns.values]
gdf.reset_index(inplace=True)

train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")




new= pd.read_csv("../input/new_merchant_transactions.csv", parse_dates=['purchase_date'])
new = binarize(new)
new = pd.get_dummies(new, columns=['category_2', 'category_3'])
new = reduce_mem_usage(new)




gdf = new.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "new_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")




cnt_srs = train.groupby("new_transactions")[target_col].mean()
cnt_srs = cnt_srs.sort_index()

sns.scatterplot(data=cnt_srs, size=(20,15))
plt.title('Loyalty score by Number of new merchant transactions')
plt.show()




for d_name, d_day in holidays:
    dist_holiday(new, d_name, d_day, 'purchase_date')




# Adding more features from new transactions
new.loc[:, 'purchase_date'] = pd.DatetimeIndex(new['purchase_date']).                                      astype(np.int64) * 1e-9
gdf = new.groupby("card_id").agg(agg_func)
gdf.columns = ['_new_'.join(col).strip() for col in gdf.columns.values]
gdf.reset_index(inplace=True)

train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")




del new, historical




import gc
gc.collect()




target = train['target']
del train['target']

features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]




xgb_params = {
            'gpu_id': 0,  
            'objective': 'reg:linear', 
            'eval_metric': 'rmse', 
            'silent': True, 
            'booster': 'gbtree', 
            'n_jobs': 4, 
            'tree_method': 'gpu_hist', 
            'grow_policy': 'lossguide', 
            'max_depth': 12, 
            'seed': 538, 
            'colsample_bylevel': 0.9, 
            'colsample_bytree': 0.8, 
            'gamma': 0.0001, 
            'learning_rate': 0.006150886706231842, 
            'max_bin': 128, 
            'max_leaves': 47, 
            'min_child_weight': 40, 
            'reg_alpha': 10.0, 
            'reg_lambda': 10.0, 
            'subsample': 0.9,
            'n_estimators': 20000
}




import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

folds = KFold(n_splits=10, shuffle=True, random_state=15)
oof = np.zeros(len(train))
xgb_predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    X_train, y_train = (train.iloc[trn_idx][features], target.iloc[trn_idx])
    X_valid, y_valid = (train.iloc[val_idx][features], target.iloc[val_idx])
    
    clf = xgb.XGBRegressor(**xgb_params)
    clf.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose=1000, early_stopping_rounds = 1000)
    oof[val_idx] = clf.predict(X_valid, ntree_limit=clf.best_ntree_limit)
    
    xgb_predictions += clf.predict(test[features], ntree_limit=clf.best_ntree_limit) / folds.n_splits




print("CV score with XGB: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))




xgb.plot_importance(clf, height=0.8, grid=False, title='XGBoost - Feature Importance', max_num_features=20)
plt.figure(figsize=(20,18))
plt.show()




sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = xgb_predictions
sub_df.to_csv("xgb_preds_updated.csv", index=False)




plt.figure(figsize=(20,18))
xgb.plot_tree(clf, num_trees=3)
plt.show()

