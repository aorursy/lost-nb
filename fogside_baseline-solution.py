#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


ls -lh ../input/


# In[3]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[4]:


from collections import Counter


# In[5]:


Counter(train_df.TARGET.values).most_common()


# In[6]:


train_df.info()


# In[7]:


feats_list = ['bathrooms', 'bedrooms', 'listing_id', 'price']


# In[8]:


train_df.isna().sum()


# In[9]:


print("before drop: ", len(train_df))
train_df.dropna(inplace=True, subset=feats_list)
print("after drop: ", len(train_df))


# In[10]:


X_train = train_df.loc[:, feats_list]
X_test = test_df.loc[:, feats_list]

y_train = train_df.loc[:, 'TARGET'].values


# In[11]:


X_test.shape


# In[12]:


X_test.isna().sum()


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


lg = LogisticRegression(multi_class='ovr', solver='lbfgs',
                        class_weight={'low':0.33, 'high':2.9, 'medium':1.})


# In[15]:


lg.fit(X_train, y_train)


# In[16]:


y_pred = lg.predict(X_test)


# In[17]:


from collections import Counter
Counter(y_pred).most_common()


# In[18]:


submit = pd.DataFrame.from_dict({'Id':test_df.Id.values, 'TARGET': y_pred})
submit.to_csv("sumbit.csv", index=False)


# In[19]:


submit.head()


# In[20]:





# In[20]:




