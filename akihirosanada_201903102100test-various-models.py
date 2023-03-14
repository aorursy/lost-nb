#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv", low_memory=False)\ntest = pd.read_csv("../input/test.csv", low_memory=False)')




# basic models
# from https://www.kaggle.com/ankitdhall97/basic-models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb




train.shape




train.head()




from sklearn.model_selection import *




train_sub = train.sample(frac=0.2)
X = train_sub.drop(["ID_code", "target"],axis=1)
y = train_sub["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)




[print(x.shape) for x in [X_train, X_test, y_train, y_test]]




from sklearn.metrics import *




def model_score(model):
    return {"train":roc_auc_score(y_train, model.predict(X_train)),
            "test":roc_auc_score(y_test, model.predict(X_test))}




lsvc = LinearSVC(verbose=True)
get_ipython().run_line_magic('time', 'lsvc.fit(X_train, y_train)')
model_score(lsvc)




xgb = XGBClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'xgb.fit(X_train, y_train)')
model_score(xgb)




lr = LogisticRegression(n_jobs=-1)
get_ipython().run_line_magic('time', 'lr.fit(X_train, y_train)')
model_score(lr)




rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=10)
get_ipython().run_line_magic('time', 'rf.fit(X_train, y_train)')
model_score(rf)




gbc = GradientBoostingClassifier(verbose=1)
get_ipython().run_line_magic('time', 'gbc.fit(X_train, y_train)')
model_score(gbc)




gnb = GaussianNB()
get_ipython().run_line_magic('time', 'gnb.fit(X_train, y_train)')
get_ipython().run_line_magic('time', 'model_score(gnb)')




train_sub_true = train.loc[train.target == 1,:].reset_index(drop=True)
train_sub_true.shape




train_sub_false = train.loc[train.target != 1,:]                    .sample(n=train_sub_true.shape[0]).reset_index(drop=True)
train_sub_2 = pd.concat([train_sub_true, train_sub_false],axis=0)                            .reset_index(drop=True)




train_sub_2.shape




X_train_2, X_test_2, y_train_2, y_test_2 =     train_test_split(train_sub_2.drop(["ID_code", "target"],axis=1),
                     train_sub_2["target"])




def model_score_2(model):
    return {"train":roc_auc_score(y_train_2, model.predict(X_train_2)),
            "test":roc_auc_score(y_test_2, model.predict(X_test_2))}




gnb2 = GaussianNB()
get_ipython().run_line_magic('time', 'gnb2.fit(X_train_2, y_train_2)')
get_ipython().run_line_magic('time', 'model_score_2(gnb2)')




lsvc2 = LinearSVC(verbose=True)
get_ipython().run_line_magic('time', 'lsvc2.fit(X_train_2, y_train_2)')
model_score_2(lsvc2)




xgb2 = XGBClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'xgb2.fit(X_train_2, y_train_2)')
model_score_2(xgb2)




lr2 = LogisticRegression(n_jobs=-1)
get_ipython().run_line_magic('time', 'lr2.fit(X_train_2, y_train_2)')
model_score_2(lr2)




rf2 = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=10)
get_ipython().run_line_magic('time', 'rf2.fit(X_train_2, y_train_2)')
model_score_2(rf2)




gbc2 = GradientBoostingClassifier(verbose=1)
get_ipython().run_line_magic('time', 'gbc2.fit(X_train_2, y_train_2)')
model_score_2(gbc2)




from IPython.display import display
def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)




display_all(train.describe().T)




def make_features(df):
    df = df.copy()
    df["mean"] = df.mean(axis=1)
    df["skew"] = df.skew(axis=1)
    df["std"] = df.std(axis=1)
    df["kurt"] = df.kurt(axis=1)
    df["max"] = df.max(axis=1)
    df["min"] = df.min(axis=1)
    df["max_min"] = df["max"] - df["min"]
    df["mean_std"] = df["mean"] - df["std"]
    return df




train_sub_3 = make_features(train_sub_2.drop(["ID_code", "target"], axis=1))




X_train_3, X_test_3, y_train_3, y_test_3 =     train_test_split(train_sub_3,
                     train_sub_2["target"])




def model_score_3(model):
    return {"train":roc_auc_score(y_train_3, model.predict(X_train_3)),
            "test":roc_auc_score(y_test_3, model.predict(X_test_3))}




gnb3 = GaussianNB()
get_ipython().run_line_magic('time', 'gnb3.fit(X_train_3, y_train_3)')
get_ipython().run_line_magic('time', 'model_score_3(gnb3)')




X_for_predict = test.drop(["ID_code"], axis=1)




submission = pd.DataFrame({"ID_code":test.ID_code, 
                           "target":gnb2.predict_proba(X_for_predict)[:,1]})




submission.head()




submission.to_csv("submission.csv", index=False)






