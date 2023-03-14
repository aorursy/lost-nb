#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


df=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
df.head(5)


# In[3]:


test = test_df.iloc[:, 1:].values


# In[4]:


df.info()


# In[5]:


X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
y


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.200, random_state = 1,stratify =y)


# In[8]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[10]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[11]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[12]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[14]:


y_pred = logreg.predict(test)


# In[15]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('logreg1.csv', index=False)


# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# In[17]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[18]:


y_pred = gnb.predict(X_test)


# In[19]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[21]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[22]:


y_pred = gnb.predict(test)


# In[23]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('gaussian.csv', index=False)


# In[24]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[25]:


bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, y_train)


# In[26]:


bnb.score(X_test, y_test)


# In[27]:


y_pred = bnb.predict(X_test)


# In[28]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[30]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[31]:


y_pred = bnb.predict(test)


# In[32]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('Bernollin.csv', index=False)


# In[33]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier


# In[34]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[35]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[36]:


y_pred = clf.predict(test)


# In[37]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('DTrees.csv', index=False)


# In[38]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
import pandas as pd


# In[39]:


xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[40]:


xg_cl.fit(X_train, y_train)


# In[41]:


y_pred = xg_cl.predict(X_test)


# In[42]:


accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[43]:


y_pred = xg_cl.predict(test)


# In[44]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('xgboost1.csv', index=False)

