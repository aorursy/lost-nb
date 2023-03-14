#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[3]:


def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    classes = list(le.classes_)
    test_ids = test.id
    
    train_features = train.drop(['species', 'id'], axis = 1)
    train_target = train.species
    test_features = test.drop(['id'], axis = 1)
    
    return train_features, train_target, labels, test_features,               test_ids, classes
    
train_features, train_target, labels, test_features,         test_ids, classes = encode(df_train, df_test)


# In[4]:


sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, cross_index in sss:
    train_training_data, train_cross_data = train_features.values[train_index], train_features.values[cross_index]
    train_training_target, train_cross_target = labels[train_index], labels[cross_index]


# In[5]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="rbf", probability = True),
    SVC(kernel="linear", probability = True),
    NuSVC(probability = True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

Analysis_cols = ["Classifier", "Accuracy", "Log loss"]
analysis = pd.DataFrame(columns = Analysis_cols)

for clf in classifiers:
    clf.fit(train_training_data, train_training_target)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('***Results***')
    train_prediction = clf.predict(train_cross_data)
    acc = accuracy_score(train_cross_target, train_prediction)
    print("Accuracy: {:.4%}".format(acc))
    
    train_prediction = clf.predict_proba(train_cross_data)
    ll = log_loss(train_cross_target, train_prediction)
    print("Log Loss: {}".format(ll))
    
    analysis = pd.DataFrame([[name, acc*100, ll]], columns = Analysis_cols)
    
print('='*30)

