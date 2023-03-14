#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[2]:


#Getting the datasets
train=pd.read_csv('../input/train.csv',index_col=0)
test=pd.read_csv('../input/test.csv',index_col=0)


# In[3]:


train.head()
set(train['color'])


# In[4]:


def get_IV(dataset,col_name,class_var,nbins):
    binned=pd.crosstab(pd.cut(dataset[col_name],nbins),dataset[class_var])
    #binned=binned.apply(lambda x:x*100/np.sum(x))
    #WOE=binned.apply(lambda x:np.log(x/(np.sum(x)-x)),axis=1)
    #return IV.apply(sum)
    return binned


# In[5]:


(get_IV(train,'bone_length','type',5))


# In[6]:


help(train.icol)


# In[7]:


clf=GaussianNB()
clf.fit(train.drop(['type','color'],axis=1),train['type'])
pred=pd.DataFrame({'id':test.index.values,'type':clf.predict(test.drop(['color'],axis=1))})


# In[8]:


pred.to_csv('C:\Users\sample_submissions.csv',index=False)


# In[9]:




