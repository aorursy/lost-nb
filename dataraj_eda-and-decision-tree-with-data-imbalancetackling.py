#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.tree as tree
import sklearn.ensemble as ensem
from sklearn.model_selection import train_test_split


# In[3]:


identity = pd.read_csv("../input/train_identity.csv",header=0)
transaction = pd.read_csv("../input/train_transaction.csv",header=0)


# In[4]:


tempData = transaction[["TransactionAmt","ProductCD","card4","isFraud"]]


# In[5]:


tempData.head()


# In[6]:


#how many frauddata and  non fraud records 
tempData.isFraud.value_counts()


# In[7]:


### Description of isFraud column 
tempData.groupby("isFraud").describe()


# In[8]:


tempData.card4.isna().any()


# In[9]:


tempData.card4.isna().sum()


# In[10]:


from IPython.display import display, Markdown


# In[11]:



sns.distplot(tempData.TransactionAmt)


# In[12]:


sns.jointplot(x="isFraud", y="TransactionAmt", data=tempData);


# In[13]:


data = tempData.groupby('isFraud').apply(lambda x: x.sample(n=20000))
data.reset_index(drop=True, inplace=True)


# In[14]:


data.head()


# In[15]:


data.ProductCD.value_counts()


# In[16]:


data.card4.value_counts()


# In[17]:


data.card4.value_counts()


# In[18]:


data.replace({"ProductCD":{'C':0,'H':1,'R':2,'S':3,'W':4},
            "card4":{"american express":0,"discover":1,"mastercard":2,"visa":3}
           }, inplace=True)


# In[19]:


data.card4.value_counts()


# In[20]:


data.isna().any()


# In[21]:


data.ProductCD.value_counts()


# In[22]:


data.dropna(axis=0,inplace=True)


# In[23]:


data.head()


# In[24]:


indData = data.loc[:,"TransactionAmt":"card4"]
depdData = data.loc[:,'isFraud']
indTrain, indTest, depTrain, depTest = train_test_split(indData, depdData, test_size=0.2, random_state=0)


# In[25]:


mytree =  tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)


# In[26]:


import sklearn.metrics as metric


# In[27]:


mytree.fit(indTrain,depTrain)
predVal =  mytree.predict(indTest)
actVal = depTest.values
metric.confusion_matrix(actVal, predVal)


# In[28]:


metric.accuracy_score(actVal, predVal)


# In[29]:


rft =  ensem.RandomForestClassifier(criterion='entropy',max_depth=30,
                                   n_estimators=500,verbose=0)
rft.fit(indTrain,depTrain)
predVal =  rft.predict(indTest)
actVal = depTest.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))


# In[ ]:




