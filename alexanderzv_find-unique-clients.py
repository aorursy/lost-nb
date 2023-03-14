#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import hashlib
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


pd.options.display.max_rows = 500
pd.options.display.max_columns = 100


# In[3]:


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_ind = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')


# In[4]:


train = train.merge(train_ind, how = 'left', on ='TransactionID' )
del train_ind


# In[5]:


train['card1'] = train['card1'].fillna(0)
train['card2'] = train['card2'].fillna(0)
train['card3'] = train['card3'].fillna(0)
train['card5'] = train['card5'].fillna(0)
train['card4'] = train['card4'].fillna('nan')
train['card6'] = train['card6'].fillna('nan')


# In[6]:


def card_info_hash(x):
    s = (str(int(x['card1']))+
         str(int(x['card2']))+
         str(int(x['card3']))+
         str(x['card4'])+
         str(int(x['card5']))+
         str(x['card6']))
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h


# In[7]:


def device_hash(x):
    s =  str(x['id_30'])+str(x['id_31'])+str(x['id_32'])+str(x['id_33'])+str( x['DeviceType'])+ str(x['DeviceInfo'])
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h


# In[8]:


train['card_hash'] = train.apply(lambda x: card_info_hash(x), axis=1   )
train['device_hash'] = train.apply(lambda x: device_hash(x), axis=1   )


# In[9]:


def get_data_by_card_hash( data, card_hash):
    mask = data['card_hash']==card_hash
    return data.loc[mask,:].copy()


def get_data_by_device_hash( data, device_hash):
    mask = data['device_hash']==device_hash
    return data.loc[mask,:].copy()


def get_data_by_card_and_device_hash( data, card_hash, device_hash):
    mask = (data['card_hash']==card_hash) &(data['device_hash']==device_hash)
    return data.loc[mask,:].copy()


# In[10]:


s = train.groupby(['card_hash' , 'device_hash'])['isFraud'].agg(['mean', 'count'])


# In[11]:


s[(s['mean']==1) & (s['count']>15) ].head(500)


# In[ ]:





# In[12]:


very_strange_thing = get_data_by_card_and_device_hash(train, '751777fefa9891d', '669f9256b04fb02')


# In[13]:


very_strange_thing[[ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307']]


# In[14]:


#magic
very_strange_thing['V307_diff'] = very_strange_thing['V307'].diff().shift(-1)


# In[15]:


very_strange_thing['difference'] = very_strange_thing['V307_diff'] - very_strange_thing['TransactionAmt']


# In[16]:


very_strange_thing[[ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference']]


# In[17]:


len(very_strange_thing)


# In[18]:


def dt_features(data):
    data = data.copy()
    
    start_dt = data['TransactionDT'].min()
    data['TransactionDT_norm']  = (data['TransactionDT'] - start_dt)/3600
    data['TransactionDT_norm_days'] = data['TransactionDT_norm']/24
    data['TransactionDT_diff'] = data['TransactionDT_norm'].diff().fillna(0)
    data['TransactionDT_diff_days'] = data['TransactionDT_diff']/24
    return data


# In[19]:


very_strange_thing = dt_features(very_strange_thing)


# In[20]:


list_of_interesting_features = [ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference', 
                                'TransactionDT_norm_days', 'TransactionDT_diff', 'TransactionDT_diff_days', 
                               
                               ] + ['D{}'.format(i) for i in range(1,16)]


# In[21]:


very_strange_thing[list_of_interesting_features]

# look at  TransactionDT_norm_days and  D8	D9
# D1 looks like  TransactionDT_norm_days rounded to int


# In[22]:


plt.figure(figsize=(14,6))
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V202'] , label = 'V202' )
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V203'] , label = 'V203')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V204'] , label = 'V204')

plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V306'] , label = 'V306')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V307'] , label = 'V307')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V308'] , label = 'V308')

plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V316'] , label = 'V316')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V317'] , label = 'V317')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V318'] , label = 'V318')

plt.xlabel('Datetime')

plt.legend()
plt.show()


# In[23]:


list_of_interesting_features = [ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference', 
                                'TransactionDT_norm_days', 'TransactionDT_diff', 'TransactionDT_diff_days', 
                               
                               ] + ['C{}'.format(i) for i in range(1,15)]


# In[24]:


very_strange_thing[list_of_interesting_features]
# c2


# In[ ]:





# In[25]:


s = get_data_by_card_and_device_hash(train, 'b4f15ed9e7c1e0a', 'b62aa9813bf1ea8')[[ 'TransactionID',
'isFraud',
'TransactionDT',
'TransactionAmt',
'ProductCD',
'device_hash','card_hash', 'V307']]


# In[26]:


#magic
s['V307_diff'] = s['V307'].diff().shift(-1)
s['difference'] = s['V307_diff'] - s['TransactionAmt']


# In[27]:


s

