#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[ ]:


# hold-out
from sklearn.model_selection import train_test_split

# K折交叉验证
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

# K折分布保持交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

# 时间序列划分方法
from sklearn.model_selection import TimeSeriesSplit

# booststrap 采样
from sklearn.utils import resample


# In[ ]:


# X = np.zeros((20, 5))
# Y = np.array([1, 2, 3, 4] * 5)
# print(X, Y)

X = np.zeros((20, 5))
Y = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5)
print(X, Y)


# In[ ]:


# 直接按照比例拆分
train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size = 0.2)
print(train_y, val_y)

# 按照比例 & 标签分布划分
train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size = 0.2, stratify=Y)
print(train_y, val_y)


# In[ ]:


kf = KFold(n_splits=5)
for train_idx, test_idx, in kf.split(X, Y):
    print(train_idx, test_idx)
    print('Label', Y[test_idx])
    print('')


# In[ ]:


kf = RepeatedKFold(n_splits=5, n_repeats=2)
for train_idx, test_idx, in kf.split(X, Y):
    print(train_idx, test_idx)
    print('Label', Y[test_idx])
    print('')


# In[ ]:


kf = StratifiedKFold(n_splits=5)
for train_idx, test_idx, in kf.split(X, Y):
    print(train_idx, test_idx)
    print('Label', Y[test_idx])
    print('')


# In[ ]:


kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
for train_idx, test_idx, in kf.split(X, Y):
    print(train_idx, test_idx)
    print('Label', Y[test_idx])
    print('')


# In[ ]:


kf = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx, in kf.split(X, Y):
    print(train_idx, test_idx)
    print('Label', Y[test_idx])
    print('')


# In[ ]:


train_X, train_Y = resample(X, Y, n_samples=16)
val_X, val_Y = resample(X, Y, n_samples=4)
print(train_Y, val_Y)


# In[ ]:


get_ipython().system(' unzip ../input/two-sigma-connect-rental-listing-inquiries/train.json.zip')
get_ipython().system(' unzip ../input/two-sigma-connect-rental-listing-inquiries/test.json.zip')


# In[ ]:


get_ipython().system('ls ./')


# In[ ]:


train_df = pd.read_json('./train.json')
test_df = pd.read_json('./test.json')
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price", 
                    "num_photos", "num_features", "num_description_words","created_year", 
                    "created_month", "created_day", "listing_id", "created_hour"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour


# In[ ]:


categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)


# In[ ]:


train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler


# In[ ]:


clf = LogisticRegression()
clf = RandomForestClassifier()
clf = LGBMClassifier()
# clf = XGBClassifier()


# In[ ]:


# 这写了一个bug，你能改好吗？
train_X = StandardScaler().fit_transform(train_df[features_to_use])
test_X = StandardScaler().fit_transform(test_df[features_to_use])

kf = StratifiedKFold(n_splits=5)
test_pred = None
for train_idx, test_idx, in kf.split(train_X, train_df['interest_level']):
    
    print(train_idx, test_idx)
    clf.fit(train_X[train_idx], train_y[train_idx])
    print('Val loss', log_loss(train_y[test_idx], 
                   clf.predict_proba(train_X[test_idx])))
    
    if test_pred is None:
        test_pred = clf.predict_proba(test_X)
    else:
        test_pred += clf.predict_proba(test_X)

test_pred /= 5


# In[ ]:


train_X = train_df[features_to_use].values
test_X = test_df[features_to_use].values

kf = StratifiedKFold(n_splits=5)
test_pred = None
for train_idx, test_idx, in kf.split(train_X, train_df['interest_level']):
    
    print(train_idx, test_idx)
    clf.fit(train_X[train_idx], train_y[train_idx])
    print('Val loss', log_loss(train_y[test_idx], 
                   clf.predict_proba(train_X[test_idx])))
    
    if test_pred is None:
        test_pred = clf.predict_proba(test_X)
    else:
        test_pred += clf.predict_proba(test_X)

test_pred /= 5


# In[ ]:


# lightGBM
clf = LGBMClassifier(learning_rate=0.05, n_estimators=2000, n_jobs=2)

train_X = train_df[features_to_use].values
test_X = test_df[features_to_use].values

kf = StratifiedKFold(n_splits=5)
test_pred = None
for train_idx, test_idx, in kf.split(train_X, train_df['interest_level']):
    
    print(train_idx, test_idx)
    clf.fit(train_X[train_idx], train_y[train_idx], 
            eval_set=[(train_X[test_idx], train_y[test_idx]), (train_X[test_idx], train_y[test_idx])],
           verbose=50, early_stopping_rounds=50)
    print('Val loss', log_loss(train_y[test_idx], 
                   clf.predict_proba(train_X[test_idx])))
    
    if test_pred is None:
        test_pred = clf.predict_proba(test_X)
    else:
        test_pred += clf.predict_proba(test_X)

test_pred /= 5


# In[ ]:


out_df = pd.DataFrame(test_pred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_starter2.csv", index=False)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'LGBMClassifier')


# In[ ]:


from sklearn.metrics import make_scorer
def my_scorer(clf, X, y_true):
    class_labels = clf.classes_
    y_pred_proba = clf.predict_proba(X)
    return log_loss(y_true, y_pred_proba)

from sklearn.model_selection import GridSearchCV
parameters = {
    'num_leaves':( 4, 8, 16, 32), 
    'subsample':(0.75, 0.85, 0.95),
    'min_child_samples': (5, 10, 15)
}

clf = GridSearchCV(LGBMClassifier(), param_grid=parameters, n_jobs=6, scoring=my_scorer, cv=5)
clf.fit(train_X, train_y)


# In[ ]:


from sklearn.metrics import make_scorer
def my_scorer(clf, X, y_true):
    class_labels = clf.classes_
    y_pred_proba = clf.predict_proba(X)
    return log_loss(y_true, y_pred_proba)

from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'num_leaves':( 4, 8, 16, 32), 
    'subsample':[0.75, 1],
    'min_child_samples': (5, 10, 15)
}

clf = GridSearchCV(LGBMClassifier(), param_grid=parameters, 
                   n_jobs=6, scoring=my_scorer, cv=StratifiedKFold(n_splits=5))
clf.fit(train_X, train_y)

