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
import seaborn as sns
import sklearn.tree as tree
import sklearn.ensemble as ensem
from sklearn.model_selection import train_test_split




identity = pd.read_csv("../input/train_identity.csv",header=0)
transaction = pd.read_csv("../input/train_transaction.csv",header=0)




tempData = transaction[["TransactionAmt","ProductCD","card4","isFraud"]]




tempData.head()




#how many frauddata and  non fraud records 
tempData.isFraud.value_counts()




### Description of isFraud column 
tempData.groupby("isFraud").describe()




tempData.card4.isna().any()




tempData.card4.isna().sum()




from IPython.display import display, Markdown





sns.distplot(tempData.TransactionAmt)




sns.jointplot(x="isFraud", y="TransactionAmt", data=tempData);




data = tempData.groupby('isFraud').apply(lambda x: x.sample(n=20000))
data.reset_index(drop=True, inplace=True)




data.head()




data.ProductCD.value_counts()




data.card4.value_counts()




data.card4.value_counts()




data.replace({"ProductCD":{'C':0,'H':1,'R':2,'S':3,'W':4},
            "card4":{"american express":0,"discover":1,"mastercard":2,"visa":3}
           }, inplace=True)




data.card4.value_counts()




data.isna().any()




data.ProductCD.value_counts()




data.dropna(axis=0,inplace=True)




data.head()




indData = data.loc[:,"TransactionAmt":"card4"]
depdData = data.loc[:,'isFraud']
indTrain, indTest, depTrain, depTest = train_test_split(indData, depdData, test_size=0.2, random_state=0)




mytree =  tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)




import sklearn.metrics as metric




mytree.fit(indTrain,depTrain)
predVal =  mytree.predict(indTest)
actVal = depTest.values
metric.confusion_matrix(actVal, predVal)




metric.accuracy_score(actVal, predVal)




rft =  ensem.RandomForestClassifier(criterion='entropy',max_depth=30,
                                   n_estimators=500,verbose=0)
rft.fit(indTrain,depTrain)
predVal =  rft.predict(indTest)
actVal = depTest.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))






