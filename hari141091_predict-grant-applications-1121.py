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




import numpy as np 
import pandas as pd 




new = pd.read_csv('../input/unimelb_training.csv')
new.head()




new.describe() 




new.columns




new.info()




import matplotlib.pyplot as plt




new.isnull().sum().any




new.columns




new.isnull().values.any()




## from sklearn.model_selection import train_test_split
## from sklearn.linear_model import LogisticRegression
## model = LogisticRegression()
## model.fit(X_train, y_train)




new_binary = new.filter(['Grant.Status','RFCD.Percentage.5','SEO.Percentage.5'], axis=1)
new_binary.head()




new_binary.fillna(0, inplace=True)
new_binary.head()




x = new_binary.iloc[:,1:]
x.head()




y = new_binary.iloc[:,0]
y.head()




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)




from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)




X_train.shape




y_train.shape




X_test.shape




y_test.shape




dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)




# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))






