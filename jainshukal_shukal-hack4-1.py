#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.





import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from matplotlib.artist import setp




df_train=pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')




df_test=pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')
df_test.columns




df_train.head()
df_train.columns




train=df_train.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)




test=df_test.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)




train.head()





train.shape




train=train.fillna(method='ffill')




test=test.fillna(method='ffill')




train.isnull().any()




plt.figure(figsize=(20,10))
sns.countplot(data=train,x='MSSubClass')




sns.barplot(train.OverallQual, train.SalePrice)




plt.figure(figsize=(20,10))
sns.countplot(data=train,x='YearBuilt')
plt.xticks(rotation=90)




fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,2),(0,0))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'], color=('yellowgreen'))
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(0,1))
plt.scatter(x=train['TotalBsmtSF'], y=train['SalePrice'], color=('red'))
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(1,0))
plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'], color=('deepskyblue'))
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(1,1))
plt.scatter(x=train['MasVnrArea'], y=train['SalePrice'], color=('gold'))
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )




plt.figure(figsize=(20,10))
sns.distplot(train['SalePrice'],color ='b')




train.head()




train['MSZoning'].unique()
MSZoning_map={'RL':0,'RM':1,'C(all)':2,'FV':3,'RH':4}
for data in train,test:
    data['MSZoning']=data['MSZoning'].map(MSZoning_map)
    data['MSZoning']=data['MSZoning'].fillna(0)




train['Street'].unique()
street_map={'Pave':0,'Grvl':1}
for data in train,test:
    data['Street']=data['Street'].map(street_map)




train['LotShape'].unique()
Lotshape_map={'Reg':0,'IR1':1,'IR2':2,'IR3':3}
for data in test,train:
    data['LotShape']= data['LotShape'].map(Lotshape_map)




train['LandContour'].unique()
landcontour_map={'Lvl':0,'Bnk':1,'Low':2,'HLS':3}
for data in train,test:
    data['LandContour']= data['LandContour'].map(landcontour_map)




attributes_train = ['SalePrice','Street','LotShape','LandContour', 'MSSubClass', 'MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']
attributes_test =  ['MSSubClass', 'Street','LotShape','LandContour','MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']
train = train[attributes_train]
test =test[attributes_test]




X=train.drop(['SalePrice'],axis=1)
y=train['SalePrice']




from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.429)




from sklearn.preprocessing import MinMaxScaler




from sklearn.ensemble import GradientBoostingRegressor




from sklearn.ensemble import RandomForestRegressor




grad_boast=GradientBoostingRegressor()




grad_boast.fit(X_train,y_train)




y_pred=grad_boast.predict(X_test)
y_pred




grad_boast.score(X_train,y_train)




rand=RandomForestRegressor()
rand.fit(X_train,y_train)
rand.score(X_train,y_train)




submission = pd.DataFrame({
        "Id":df_test['Id'],
        "SalePrice":y_pred
    })
submission.to_csv('shukal1submission.csv',index=False,header=1 )






