#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')




import os
print(os.listdir("../input"))




train_complete=pd.read_csv("../input/train_V2.csv")
train_complete.head()




train=train_complete.sample(100000,random_state =1)
train.head()




train=train.drop(['Id','groupId','matchId'],axis=1)
train.head()




train.info()




dftrain=train.copy()




corr=dftrain.corr()
corr





plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True)




plt.figure(figsize=(15,10))
corr1=corr.abs()>0.5
sns.heatmap(corr1,annot=True)




plt.title('Correlation B/w Winning % and other Independent Variable')
dftrain.corr()['winPlacePerc'].sort_values(ascending=False).plot(kind='bar',figsize=(10,8))




k = 10 #number of variables for heatmap
f,ax = plt.subplots(figsize=(11, 11))
cols = dftrain.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(dftrain[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()




sns.set()
cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']
sns.pairplot(dftrain[cols], size = 2.5)
plt.show()




train.info()




train_complete=pd.get_dummies(train)
train_complete.head()




from statsmodels.stats.outliers_influence import variance_inflation_factor
x_features=list(train_complete)
x_features.remove('winPlacePerc')
data_mat = train_complete[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]




x_features=list(train_complete)
x_features.remove('winPlacePerc')
x_features.remove('maxPlace')
x_features.remove('numGroups')
x_features.remove('winPoints')
x_features.remove('rankPoints')
x_features.remove('matchType_squad-fpp')
x_features.remove('matchDuration')
data_mat = train_complete[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]




x=train_complete[x_features]
y=train_complete[['winPlacePerc']]
# Train Test Split
validation_size = 0.30
seed = 1
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)




model_final=GradientBoostingRegressor(n_estimators=100,learning_rate=0.5)
model_final.fit(x_train,y_train)
print(model_final.score(x_train,y_train))
print(model_final.score(x_validation,y_validation))




test=pd.read_csv('../input/test_V2.csv')
dftest=test.drop(['Id','groupId','matchId'],axis=1)
dftest.head()




dftest.info()




test_complete=pd.get_dummies(dftest)
x_test=test_complete[x_features]




pred=model_final.predict(x_test)
pred[1:5]




pred_df=pd.DataFrame(pred,test['Id'],columns=['winPlacePerc'])
pred_df.head()




pred_df.to_csv('sample_submission.csv')

