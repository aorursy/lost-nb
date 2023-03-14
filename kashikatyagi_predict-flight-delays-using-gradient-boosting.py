#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score




train = pd.read_csv("/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv")
train.head()




test = pd.read_csv("/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv")
test.head()




X_train, y_train = train[['Distance', 'DepTime']].values, train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_test = test[['Distance', 'DepTime']].values
X_train




y_train




X_test




X_train_part, X_valid, y_train_part, y_valid =     train_test_split(X_train, y_train, 
                     test_size=0.3, random_state=123)

scaler = StandardScaler()  #to standardise features 
X_train_part = scaler.fit_transform(X_train_part)
X_valid = scaler.transform(X_valid)
X_train_part




X_valid




logit = LogisticRegression()

logit.fit(X_train_part, y_train_part)
logit_valid_pred = logit.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, logit_valid_pred)




logit_valid_pred




X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit.fit(X_train_scaled, y_train)
logit_test_pred = logit.predict_proba(X_test_scaled)[:, 1]

pd.Series(logit_test_pred, 
          name='dep_delayed_15min').to_csv('logit_Results.csv', 
                                           index_label='id', header=True)




logit_test_pred.shape




logit_test_pred[logit_test_pred > 0.3] = 1
logit_test_pred[logit_test_pred <= 0.3] = 0




pd.Series(logit_test_pred, 
          name='dep_delayed_15min').to_csv('logit_results_thresh.csv', 
                                           index_label='id', header=True)

