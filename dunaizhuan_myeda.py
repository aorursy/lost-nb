#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
plt.style.use('ggplot')


# In[3]:


print(os.listdir('../input'))


# In[4]:


folder_path = '../input/ieee-fraud-detection/'
train_id = pd.read_csv(f'{folder_path}train_identity.csv')
train_tr = pd.read_csv(f'{folder_path}train_transaction.csv')
test_id = pd.read_csv(f'{folder_path}test_identity.csv')
test_tr = pd.read_csv(f'{folder_path}test_transaction.csv')
#sub = pd.read_csv(f'{folder_path}sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_tr, train_id,on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')


# In[5]:


train_id.columns


# In[6]:


train_tr.columns


# In[7]:


train_id.head()


# In[8]:


train_tr.head()


# In[9]:


train.head()


# In[10]:


train.shape[0]


# In[11]:


train['isFraud'].mean()


# In[12]:


train['DeviceType'].fillna('nan').value_counts()


# In[13]:


train['DeviceType'] = train['DeviceType'].fillna('nan')
train.groupby(['DeviceType'])['isFraud'].mean()


# In[14]:


(train[train['isFraud']==1]['DeviceType']=='mobile').astype(int).sum() ,(train[train['isFraud']==1]['DeviceType']=='desktop').astype(int).sum()


# In[15]:


train['isFraud'].mean()


# In[16]:


(train['isFraud']==1).astype(int).sum()-5657-5554


# In[17]:


train_mobile = train[train['DeviceType']=='mobile']
train_desktop = train[train['DeviceType']=='desktop']
train_nan = train[train['DeviceType']=='nan']


# In[18]:


train_mobile.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()


# In[19]:


train_desktop.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()


# In[20]:


train_nan.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()


# In[21]:


train['ProductCD'].unique()


# In[22]:


train['addr1'].unique()


# In[23]:


train['addr2'].unique()


# In[24]:


train.shape[0]


# In[25]:


train['M9'].unique()


# In[26]:


for i in range(12,39):
    print(train['id_'+str(i)].isnull().any())


# In[27]:


for i in range(12,39):
    print(train['id_'+str(i)].nunique())


# In[28]:


for col, values in train.iteritems():
    num_uniques = values.nunique()
    print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))
    print (values.unique())
    print ('\n')


# In[29]:


for i in range(1,10):
    print(train['M'+str(i)].unique())


# In[30]:


# M1-M9: match, such as names on card and address, etc.


# In[31]:


for i in range(1,16):
    print(train['D'+str(i)].nunique())


# In[32]:


train['D1'].head()


# In[33]:


train['D2'].fillna(-1.).value_counts()


# In[34]:


train['P_emaildomain'].unique()


# In[35]:


test['P_emaildomain'].fillna('0').value_counts()


# In[36]:


train['P_emaildomain'].fillna('0').value_counts()


# In[37]:


train['R_emaildomain'].unique()


# In[38]:


train['R_emaildomain'].fillna('0').value_counts()


# In[39]:


test['R_emaildomain'].fillna('0').value_counts()


# In[40]:


train['P_emaildomain'].isnull().any()


# In[41]:


train_id.columns


# In[42]:


train_id['id_03'].unique()


# In[43]:


train_id['id_04'].unique()


# In[44]:


train_id.head()


# In[45]:


train['ProductCD'].nunique()


# In[46]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum()[:20]


# In[47]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum()[20:40]


# In[48]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum().index


# In[49]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum().value_counts()


# In[50]:


train['V103'].hist(bins=100)


# In[51]:


(train['V2'].isnull()==train['V10'].isnull()).any()


# In[52]:


train.head()


# In[53]:


print(train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].head(10))


# In[54]:


V_variables = [i for i in list(train) if 'V' in i]
na_value = train[V_variables].isnull().sum()
na_list = na_value.unique()
na_value = na_value.to_dict()
cols_same_null = []
for i in range(len(na_list)):
    cols_same_null.append([k for k,v in na_value.items() if v == na_list[i]])
print(cols_same_null)


# In[55]:


train['D15'].unique()


# In[56]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0].head(10)


# In[57]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0].head(30)


# In[58]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1].head(30)


# In[59]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D1'].describe()


# In[60]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1]['D1'].describe()


# In[61]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1]['D2'].describe()


# In[62]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D2'].describe()


# In[63]:


train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D2'].value_counts()


# In[64]:


train_tr['card4'].isnull().sum()


# In[65]:


for col1 in train_tr.columns[train_tr.columns.str.startswith('D')]:
    for col2 in train_tr.columns[train_tr.columns.str.startswith('D')]:
        if col1==col2:
            continue
        else:
            tmp = train_tr[train_tr[col1].isnull()]
            k = tmp[col2].isnull().sum() - tmp.shape[0]
            if abs(k)<50 :
                print('%s -> %s %d'%(col1,col2,k))
            


# In[66]:


tmp = train_tr[train_tr[col1].isnull()]
k = tmp[col2].isnull().sum() - tmp.shape[0]


# In[67]:


for col1 in test_tr.columns[test_tr.columns.str.startswith('D')]:
    for col2 in test_tr.columns[test_tr.columns.str.startswith('D')]:
        if col1==col2:
            continue
        else:
            tmp = test_tr[test_tr[col1].isnull()]
            k = tmp[col2].isnull().sum() - tmp.shape[0]
            if abs(k)<50 :
                print('%s -> %s %d'%(col1,col2,k))
            


# In[68]:


k= train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]]


# In[69]:


k_F =  k[-train_tr['D8'].isnull()][train.isFraud==1]
k_NF =  k[-train_tr['D8'].isnull()][train.isFraud==0]


# In[70]:


k_NF['D9'].hist()


# In[71]:


k_F['D9'].hist()


# In[72]:


k_F['D8'].hist()


# In[73]:


k_NF['D8'].hist()


# In[74]:


plt.scatter(x=k_F['D1'],y=k_F['D2'])


# In[75]:


plt.scatter(x=k_NF['D1'],y=k_NF['D2'])


# In[76]:


sns.heatmap(k.corr())


# In[77]:


plt.scatter(x=k_NF['D2'],y=k_NF['D4'])


# In[78]:


plt.scatter(x=k_F['D2'],y=k_F['D4'])


# In[79]:


plt.scatter(x=k_F['D1'],y=k_F['D4'])


# In[80]:


plt.scatter(x=k_NF['D1'],y=k_NF['D4'])


# In[81]:


tmp = train.corr()


# In[82]:


sns.heatmap(train.corr())


# In[83]:


plt.figure(figsize=(100,100))
sns.heatmap(tmp)


# In[84]:


['V305','V107']


# In[85]:


corr_no_V = train.drop(train_tr.columns[train_tr.columns.str.startswith('V')],axis=1).corr()


# In[86]:


plt.figure(figsize=(32,32))
sns.heatmap(corr_no_V)


# In[ ]:




