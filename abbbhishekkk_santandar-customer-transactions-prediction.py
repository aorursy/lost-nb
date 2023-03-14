#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


data=pd.read_csv('../input/train.csv')


# In[3]:


(data['target'].value_counts()/data['target'].count()*100).plot(kind='hist',color='r')


# In[4]:


data['target'].value_counts()/data['target'].count()


# In[5]:


data.corr()


# In[6]:


data.describe()


# In[7]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[8]:


X = add_constant(data)
X.shape[1]


# In[9]:


for i in range(1,data.shape[1]):
    print(data.iloc[:,i].plot(kind='hist'))
    plt.xlabel(data.columns[i])
    plt.show()


# In[10]:


X=data.drop(columns=['target','ID_code'])


# In[11]:


X.head()


# In[12]:


Y=data.iloc[:,1]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[15]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[16]:


lr.fit(x_train,y_train)


# In[17]:


pred=lr.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[19]:


accuracy_score(y_test,pred)


# In[20]:


recall_score(y_test,pred)


# In[21]:


precision_score(y_test,pred)


# In[22]:


f1_score(y_test,pred)


# In[23]:


get_ipython().system('pip install imblearn')


# In[24]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()


# In[25]:


x_train1,ytrain1=sm.fit_sample(x_train,y_train)


# In[26]:


lr.fit(x_train1,ytrain1)


# In[27]:


pred1=lr.predict(x_test)


# In[28]:


accuracy_score(y_test,pred1),recall_score(y_test,pred1),precision_score(y_test,pred1),f1_score(y_test,pred1)


# In[29]:


from sklearn.metrics import precision_recall_curve,roc_auc_score


# In[30]:


roc_auc_score(y_test,pred1),roc_auc_score(y_test,pred)


# In[31]:





# In[31]:





# In[31]:





# In[31]:




