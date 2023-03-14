#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


import os
print(os.listdir("../input"))

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Santander dataset
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission_data = pd.read_csv('../input/sample_submission.csv')




#Size of training data
train_data.shape




train_data['target'].head(5)




train_data.describe()




X_train, X_val, y_train, y_val = train_test_split(feature_train_data, train_data['target'], test_size = 0.20, random_state = 25)




from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb




df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')




random_state = 42

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}




df_train.head()




skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=random_state)
skf.get_n_splits(X_train, y_train)




features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values
feature_importance_df = pd.DataFrame()
predictions = df_test[['ID_code']]

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    print("FOLD: ", fold, "TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 5
    p_valid = 0
    yp = 0
    
    for i in range(N):
        
        trn_data = lgb.Dataset(X_train, label = y_train)
        val_data = lgb.Dataset(X_valid, label = y_valid)
        
        lgb_clf = lgb.train(lgb_params,
                   trn_data,
                   100000,
                   valid_sets = [trn_data, val_data],
                    verbose_eval = 5000,
                    early_stopping_rounds = 3000)
        
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    
    
    #Get importance of the fold when predicting test set
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions['fold{}'.format(fold+1)] = yp/N




X_train.head()




predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)




dfa = pd.DataFrame(np.random.randn(5, 4),
                    columns=list('ABCD'),
                   index=pd.date_range('20130101', periods=5))




dfa




dfa.columns





























































































