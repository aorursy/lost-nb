#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd ../input


# In[2]:


ls 


# In[3]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[5]:


print(train.shape)
train.head()


# In[6]:


print(test.shape)
test.head()


# In[7]:


len(np.unique(train['id']))


# In[8]:


len(np.unique(train['num_room']) )


# In[9]:


train = train.set_index('timestamp')
test = test.set_index("timestamp")
train.head()


# In[10]:


train.drop("id", axis=1, inplace=True)


# In[11]:


train.head()


# In[12]:


train.describe()


# In[13]:


train.shape[0]


# In[14]:


train['price_doc'].values


# In[15]:


ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.8)
plt.xticks(rotation=90)
plt.title("linear scale")
ax.legend()


# In[16]:


ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.3)
ax.set_yscale('log')
plt.xticks(rotation=90)
plt.title("logarithmic scale")
ax.legend()


# In[17]:


# no missing value for the target value
train[train['price_doc'].isnull()]


# In[18]:


train.columns[train.isnull().any()]


# In[19]:


train2 = train.fillna(train.median())


# In[20]:


train2.head()


# In[21]:


train2.columns[train2.isnull().any()]


# In[22]:


train2.corr()


# In[23]:


categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[24]:


train2.shape


# In[25]:


train2[categorical].head()


# In[26]:


np.unique(train2['product_type'])


# In[27]:


for cat in categorical:
    print(cat, ':', np.unique(train2[cat]))


# In[28]:


yes_no_mapping = {'no': 0, 'yes': 1}


# In[29]:


# ordinal features which could be rendered as 0 and 1,
# each corresponding to 'no' and 'yes'
categorical[2:-1]


# In[30]:


for i in categorical[2:-1]:
    train2[i] = train2[i].map(yes_no_mapping)


# In[31]:


categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[32]:


np.unique(train2['ecology'].values)


# In[33]:


rate_mapping = {'excellent': 3, 'good': 2, 'satisfactory': 2, 'poor': 1, 'no data': np.nan} 


# In[34]:


train2['ecology'] = train2['ecology'].map(rate_mapping)


# In[35]:


print(len(train2[train2['ecology'].isnull()]))


# In[36]:


print(len(train2[train2['ecology'].notnull()]))


# In[37]:


print(train2.shape[0])


# In[38]:


print(len(train2[train2['ecology'].isnull()]) + len(train2[train2['ecology'].notnull()]))


# In[39]:


train2 = train2.fillna(train2.median())


# In[40]:


print(len(train2[train2['ecology'].isnull()]))


# In[41]:


train2.corr()


# In[42]:


ls


# In[43]:


train2.head()


# In[44]:


ls ../


# In[45]:


train2.head() 


# In[46]:


ls ../


# In[47]:


test = pd.read_csv("test.csv")


# In[48]:


test.head()


# In[49]:


test = test.set_index('timestamp')
test.head()


# In[50]:


test.drop("id", axis=1, inplace=True)
print(test.shape)


# In[51]:


for i in test.columns:
    if i not in train.columns:
        print(i)


# In[52]:


categorical = []
for i in test.columns:
    if type(test[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[53]:


categorical[2:-1]


# In[54]:


for i in categorical[2:-1]:
    test[i] = test[i].map(yes_no_mapping)


# In[55]:


test['ecology'] = test['ecology'].map(rate_mapping)


# In[56]:


len(test[test['ecology'].isnull()])


# In[57]:


test = test.fillna(test.median())


# In[58]:


test.columns[test.isnull().any()]


# In[59]:


# there are 33 missing values in a column called 'producty_type'
len(test[test['product_type'].isnull()])


# In[60]:


ls


# In[61]:


train2.head()


# In[62]:




