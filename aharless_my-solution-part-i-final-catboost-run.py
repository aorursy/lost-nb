#!/usr/bin/env python
# coding: utf-8



AS_DEMO = True

if AS_DEMO:
    NROUNDS = 100
else:
    NROUNDS = 6500  # Warning: needs to run overnight on multiple cores




import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm




# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df




properties2016 = pd.read_csv('../input/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)




train2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2016 = add_date_features(train2016)
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')

train2017 = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = add_date_features(train2017)
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

train_df = pd.concat([train2016, train2017], axis = 0)

print("Train: ", train_df.shape)




sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False).rename(columns = {'ParcelId': 'parcelid'})
test_df = pd.merge(sample_submission[['parcelid']], properties2017, how = 'left', on = 'parcelid')

print("Test: ", test_df.shape)




missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))




# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))




exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing        and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))




cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh        and not 'sqft' in c        and not 'cnt' in c        and not 'nbr' in c        and not 'number' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])




# some out of range int is a good choice
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)




X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)




test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)




num_ensembles = 10
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    # TODO(you): Use CV, tune hyperparameters
    model = CatBoostRegressor(
        iterations=NROUNDS, learning_rate=0.007,
        depth=6, l2_leaf_reg=14,
        bagging_temperature=8,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles




submission = pd.DataFrame({
    'ParcelId': test_df['parcelid'],
})
# https://www.kaggle.com/c/zillow-prize-1/discussion/33899, Oct,Nov,Dec
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    # TODO(you): predict for every `test_date`
    submission[label] = y_pred

submission.to_csv(
    'catboost_opt4.csv',
    float_format='%.6f',
    index=False)






