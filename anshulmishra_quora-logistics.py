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
()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression


# In[3]:


df = pd.read_csv('../input/train.csv')
test=pd.read_csv("../input/test.csv")
id=df.qid
text=df.question_text
testxt=test.question_text
ans=df.target
id1=test.qid


# In[4]:


vect=TfidfVectorizer(stop_words='english')
X_train=vect.fit_transform(text)
X_test=vect.transform(testxt)


# In[5]:


clas=LogisticRegression()
clas.fit(X_train, ans)


# In[6]:


f=clas.predict(X_test)


# In[7]:


sub = pd.DataFrame()
sub['qid'] = id1
sub['prediction'] = f
sub.to_csv('submission.csv',index=False)


# In[8]:




