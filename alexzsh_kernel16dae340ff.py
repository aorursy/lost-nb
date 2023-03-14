#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


ls


# In[ ]:


cd ../input/daguan-cups-dataset/new_data/


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


cd new_data/


# In[ ]:


ls


# In[ ]:


import os

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer


# to splite dataset into train and test data.
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# importing Machine learning Algorithm from sklearn
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.neighbors.nearest_centroid import NearestCentroid # Centroid Based Classifier
from sklearn.neighbors import KNeighborsClassifier #KNN Classifier
from sklearn.svm import SVC # Support Vector Machine
from sklearn.naive_bayes import BernoulliNB # Naive Bayes Classifier
from sklearn.tree import DecisionTreeClassifier # Decission Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:



import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import pickle
column = "word_seg"
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
feature=vec.fit_transform(train['word_seg']+train['word_seg'])
test_feature=vec.transform(test['word_seg']+train['word_seg'])
with open("vec.pic","w") as fw:
    vec_pic=pickle.dumps(vec)
    f.write(vec_pic)
    del vec_pic
with open("fea.pic","w") as fw:
    fea_pic=pickle.dumps(feature)
    f.write(fea_pic)
    del fea_pic
with open("test_fea.pic","w") as fw:
    test_fea_pic=pickle.dumps(test_feature)
    f.write(test_fea_pic)
    del test_fea_pic
    
label=(train['class']-1).astype(int)
label, feature = shuffle(label, feature, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.20, random_state=0)
print("data over")


# In[ ]:





# In[ ]:


linearRegr = LinearRegression().fit(x_train, y_train)
# logisticRegr = LogisticRegression().fit(x_train, y_train)
# model_centroid = NearestCentroid().fit(x_train, y_train)
# model_knn = KNeighborsClassifier(19).fit(x_train, y_train)
# model_svm = SVC().fit(x_train, y_train)
# model_nb = BernoulliNB().fit(x_train, y_train)
# model_dtree = DecisionTreeClassifier(criterion = "entropy",random_state = 100, max_depth=3, min_samples_leaf=5).fit(x_train, y_train)
# model_rfc = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1).fit(x_train, y_train)


# In[ ]:


# Find the accuracy of each model
phrase = "The accuracy of %s is %0.2f"
# accu_lir = linearRegr.score(x_test, y_test)
# print(phrase % ("Linear Regression", 100*accu_lir))
accu_lr = logisticRegr.score(x_test, y_test)
print(phrase % ("Logistic Regression", 100*accu_lr))
# accu_centroid = model_centroid.score(x_test, y_test)
# print(phrase % ("Centroid Based Classifier", 100*accu_centroid))
# accu_knn = model_knn.score(x_test, y_test)
# print(phrase % ("KNN", 100*accu_knn))
# accu_svm = model_svm.score(x_test, y_test)
# print(phrase % ("SVM", 100*accu_svm))
# accu_nb = model_nb.score(x_test, y_test)
# print(phrase % ("Naive Bayes", 100*accu_nb))
# accu_dtree = model_dtree.score(x_test, y_test)
# print(phrase % ("Decission Tree", 100*accu_dtree))
# accu_rfc = model_rfc.score(x_test, y_test)
# print(phrase % ("RandomForest Classifier", 100*accu_rfc))
# y_pred = model_centroid.predict(x_test)


# In[ ]:




