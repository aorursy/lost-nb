#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x = 2
f'{x} is a number'


# In[ ]:


x = 2
'{} is a number'.format(x)


# In[ ]:





# In[ ]:





# In[ ]:


tmp = pd.read_csv('../input/train.csv', nrows=2)
print(tmp.shape)
tmp


# In[ ]:


tmp['device']


# In[ ]:


# json_normalize(tmp['device']) # 此时会报错，因为读入的方式不对


# In[ ]:


import json
from pandas.io.json import json_normalize

csv_path = '../input/train.csv'
JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
nrows = 2 

df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, 
                     nrows=nrows)


# In[ ]:


dc = json_normalize(df['device'])
dc.columns = [f"{column}.{subcolumn}" for subcolumn in dc.columns]
dc


# In[ ]:


import os
import json
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, 
                     nrows=nrows) # Important!!
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

print(os.listdir("../input"))


# In[ ]:


train = load_df(nrows=100000)
test =load_df('../input/test.csv', nrows=100000)
train.head()


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


np.log1p(0)


# In[ ]:


train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(6,4))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# In[ ]:


gdf.describe()


# In[ ]:


ids = train[["fullVisitorId", "sessionId", "visitId", "visitNumber"]]
ids.sort_values(by='sessionId')
ids.isnull().sum()


# In[ ]:


ids['sessionId'].unique().shape, ids.shape, ids['fullVisitorId'].unique().shape


# In[ ]:


gong = ids.groupby(by='sessionId')['sessionId'].count()
gong1 = gong.sort_values(ascending=False)[:7].index.values.tolist()
gong2 = ids.loc[[i in gong1 for i in ids.sessionId], :].index
gong3 = train.loc[gong2].sort_values(by='sessionId')
gong3


# In[ ]:


gong3.columns


# In[ ]:


gong3['date'].unique().shape[0] == 1


# In[ ]:


gong3[[c for c in gong3.columns if gong3[c].unique().shape[0] != 1]]


# In[ ]:


nzi = pd.notnull(train["totals.transactionRevenue"]).sum()
nzr = (gdf["totals.transactionRevenue"] > 0).sum()
print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train.shape[0])
print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])


# In[ ]:


print("Number of unique visitors in train set : ",train.fullVisitorId.nunique(), " out of rows : ",train.shape[0])
print("Number of unique visitors in test set : ",test.fullVisitorId.nunique(), " out of rows : ",test.shape[0])
print("Number of common visitors in train and test set : ",len(set(train.fullVisitorId.unique()).intersection(set(test.fullVisitorId.unique())) ))


# In[ ]:


[c for c in train.columns if train[c].nunique()==1]


# In[ ]:


train.shape


# In[ ]:




