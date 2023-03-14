#!/usr/bin/env python
# coding: utf-8



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




train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')




train_data.head()




train_data.tail()




train_data.shape




test_data.shape




train_data.info()




#Check for missing data
train_data.isna().sum()
#train_data.isnull().sum().sum()




train_data['kills'].describe()




train_data[train_data['kills']==60]




train_data[train_data['matchId']==5518]




#sns.set()
sns.distplot(train_data['kills'],bins=40,kde=True)
plt.figure(figsize=(20,20))
plt.show()




[x for x in train_data['kills'].quantile() if train_data['kills'].quantile(x)




train_data['groupId'].nunique()




train_data[train_data['DBNOs']==0]




train_data[train_data['winPlacePerc']==1].groupby(train_data['groupId']).tail()




train_data[train_data['winPlacePerc']==1].groupby(train_data['Id']).tail()




train_data.groupby(train_data['groupId'])['winPlacePerc'].value_counts().max()




train_data['teamKills'].value_counts()




train_data['kills'].value_counts()




sns.jointplot(x='kills',y='winPlacePerc',data= train_data)




sns.pairplot(train_data,x_vars=['kills','teamKills'],y_vars=['winPlacePerc'])
plt.figure(figsize=(15,15))




sns.jointplot(x='winPlacePerc',y='boosts',data= train_data,height=10, ratio=3,color = 'orange')




sns.jointplot(x='winPlacePerc',y='heals',data= train_data,height=10, ratio=3, color = 'green')




sns.jointplot(x='winPlacePerc',y='headshotKills',data= train_data,height=10, ratio=3, color = 'red')




features = list(train_data.columns)
x = train_data.iloc[:,:25].values
y = train_data.loc[:,'winPlacePerc']
scaler = StandardScaler() 
x = scaler.fit_transform(x)




pca = PCA(n_components = 25)
principalComp = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComp)




print(principalDf)




pca= PCA(.95)




pca.fit(x)




pca.n_components_




pca.components_




pca.explained_variance_ratio_




plt.semilogy(pca.explained_variance_ratio_,'--o')




pd.DataFrame(pca.components_,columns =x.columns)




import numpy as np




sns.heatmap(np.log(pca.inverse_transform(np.eye(x.shape[1]))), cmap="hot", cbar=False)

