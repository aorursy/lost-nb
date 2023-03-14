#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[3]:


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.info()


# In[6]:


dataset.isna().sum()


# In[7]:


dataset.nunique()


# In[8]:


dataset.dropna(inplace=True)


# In[9]:


X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, 1].values


# In[10]:


X[1]


# In[11]:


y[1]


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[15]:


logreg = LogisticRegression()


# In[16]:


logreg.fit(X_train, y_train)


# In[17]:


X_test_dataset =test.iloc[:,2:]


# In[18]:


y_pred = logreg.predict(X_test_dataset)


# In[19]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[20]:


y_pred


# In[21]:


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred
    })


# In[22]:


submission_rfc.to_csv('submission_rfc.csv', index=False)


# In[23]:


from sklearn.naive_bayes import GaussianNB


# In[24]:


gnb = GaussianNB()


# In[25]:


gnb.fit(X_train, y_train)


# In[26]:


y_pred = gnb.predict(X_test)


# In[27]:


from sklearn import metrics


# In[28]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix


# In[30]:


print(confusion_matrix(y_test, y_pred))


# In[31]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[32]:


from sklearn.naive_bayes import BernoulliNB


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


bnb = BernoulliNB(binarize=0.0)


# In[35]:


bnb.fit(X_train, y_train)


# In[36]:


bnb.score(X_test, y_test)


# In[37]:


y_pred = bnb.predict(X_test)


# In[38]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix


# In[40]:


print(confusion_matrix(y_test, y_pred))


# In[41]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[42]:


from sklearn.naive_bayes import MultinomialNB


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


mnb = MultinomialNB(alpha=0.01)


# In[45]:


from sklearn.preprocessing import Normalizer


# In[46]:


normalizer = Normalizer(norm='l2', copy=True)


# In[47]:


X_train = Normalizer(copy=False).fit_transform(X_train)


# In[48]:


X_train


# In[49]:


from sklearn.preprocessing import LabelEncoder


# In[50]:


from sklearn.preprocessing import OneHotEncoder


# In[51]:


yNB_pred = gnb.predict(X_test_dataset)


# In[52]:


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": yNB_pred
    })


# In[53]:


submission_rfc.to_csv('NB_NazarbekovaB.csv', index=False)


# In[54]:


import os


# In[55]:


os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[56]:


import xgboost as xgb
import pandas as pd


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)


# In[58]:


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


# In[59]:


xg_cl.fit(X_train, y_train)


# In[60]:


y_pred = xg_cl.predict(X_test_dataset)


# In[61]:


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred
    })


# In[62]:


submission_rfc.to_csv('XG_NazarbekovaB.csv', index=False)


# In[63]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz


# In[64]:


clf = DecisionTreeClassifier()


# In[65]:


clf = clf.fit(X_train,y_train)


# In[66]:


y_pred = clf.predict(X_test)


# In[67]:


y_pred

