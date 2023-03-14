#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.tail()


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.info()


# In[ ]:


#Check for missing data
train_data.isna().sum()
#train_data.isnull().sum().sum()


# In[ ]:


train_data['kills'].describe()


# In[ ]:


train_data[train_data['kills']==60]


# In[ ]:


train_data[train_data['matchId']==5518]


# In[ ]:


#sns.set()
sns.distplot(train_data['kills'],bins=40,kde=True)
plt.figure(figsize=(20,20))
plt.show()


# In[ ]:


[x for x in train_data['kills'].quantile() if train_data['kills'].quantile(x)


# In[ ]:


train_data['groupId'].nunique()


# In[ ]:


train_data[train_data['DBNOs']==0]


# In[ ]:


train_data[train_data['winPlacePerc']==1].groupby(train_data['groupId']).tail()


# In[ ]:


train_data[train_data['winPlacePerc']==1].groupby(train_data['Id']).tail()


# In[ ]:


train_data.groupby(train_data['groupId'])['winPlacePerc'].value_counts().max()


# In[ ]:


train_data['teamKills'].value_counts()


# In[ ]:


train_data['kills'].value_counts()


# In[ ]:


sns.jointplot(x='kills',y='winPlacePerc',data= train_data)


# In[ ]:


sns.pairplot(train_data,x_vars=['kills','teamKills'],y_vars=['winPlacePerc'])
plt.figure(figsize=(15,15))


# In[ ]:


sns.jointplot(x='winPlacePerc',y='boosts',data= train_data,height=10, ratio=3,color = 'orange')


# In[ ]:


sns.jointplot(x='winPlacePerc',y='heals',data= train_data,height=10, ratio=3, color = 'green')


# In[ ]:


sns.jointplot(x='winPlacePerc',y='headshotKills',data= train_data,height=10, ratio=3, color = 'red')


# In[ ]:


features = list(train_data.columns)
x = train_data.iloc[:,:25].values
y = train_data.loc[:,'winPlacePerc']
scaler = StandardScaler() 
x = scaler.fit_transform(x)


# In[ ]:


pca = PCA(n_components = 25)
principalComp = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComp)


# In[ ]:


print(principalDf)


# In[ ]:


pca= PCA(.95)


# In[ ]:


pca.fit(x)


# In[ ]:


pca.n_components_


# In[ ]:


pca.components_


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


plt.semilogy(pca.explained_variance_ratio_,'--o')


# In[ ]:


pd.DataFrame(pca.components_,columns =x.columns)


# In[ ]:


import numpy as np


# In[ ]:


sns.heatmap(np.log(pca.inverse_transform(np.eye(x.shape[1]))), cmap="hot", cbar=False)

