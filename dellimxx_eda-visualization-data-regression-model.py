#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime




train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sampleSubmission.csv')




train_csv.head(5)




# datetime 변수를 여러개로 나누어 보자.
train_csv['date']  = train_csv.datetime.apply(lambda x: x.split()[0])
train_csv['hour'] = train_csv.datetime.apply(lambda x: x.split()[1].split(':')[0])
train_csv['weekday'] = train_csv.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
train_csv['month'] = train_csv.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)




train_csv.describe()




get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(15,10),ncols=2,nrows= 1)
sns.distplot(train_csv['count'],ax=axes[0])
sns.distplot(train_csv['count'].apply(lambda x: np.log1p(x)),ax=axes[1])
plt.show()




sns.boxplot(data = train_csv,y='count',orient='v')
plt.show()




fig,axes = plt.subplots(ncols=2 ,nrows=2)
fig.set_size_inches(15,10)
sns.boxplot(data=train_csv,x='season',y='count',ax=axes[0][0])
sns.boxplot(data=train_csv,x='holiday',y='count',ax=axes[0][1])
sns.boxplot(data=train_csv,x='workingday',y='count',ax=axes[1][0])
sns.boxplot(data=train_csv,x='weather',y='count',ax=axes[1][1])

fig1,axes1 = plt.subplots()
fig1.set_size_inches(15,10)
sns.boxplot(data=train_csv,x='hour',y='count')




corrs = train_csv[['temp','atemp','windspeed','humidity','count']].corr()
sns.heatmap(corrs,annot=True,vmax=3)




fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(15,10)
sns.regplot(data=train_csv,x='temp',y='count',ax=axes[0][0])
sns.regplot(data=train_csv,x='atemp',y='count',ax=axes[0][1])
sns.regplot(data=train_csv,x='humidity',y='count',ax=axes[1][0])
sns.regplot(data=train_csv,x='windspeed',y='count',ax=axes[1][1])
plt.show()




#Remove Outlier
train_csv_Outliers = train_csv[np.abs(train_csv["count"]-train_csv["count"].mean())<=(2*train_csv["count"].std())]
print("Oulier포함:",train_csv.shape)
print("Outlier제거:",train_csv_Outliers.shape)




fig,axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(15,10)
sns.distplot(train_csv['count'],ax=axes[0][0])
sns.distplot(np.log1p(train_csv['count']),ax=axes[0][1])
sns.distplot(train_csv_Outliers['count'],ax=axes[1][0])
sns.distplot(np.log1p(train_csv['count']),ax=axes[1][1])
plt.show()




fig,axes = plt.subplots()
fig.set_size_inches(15,10)
corrs = train_csv[['season', 'holiday', 'workingday', 'weather','date',
       'hour', 'weekday', 'month','temp','atemp','windspeed','humidity','registered','casual','count']].corr()
sns.heatmap(corrs,annot=True)




corrs.iloc[:,-1]




Train = pd.read_csv('../input/train.csv')
Test = pd.read_csv('../input/test.csv')




# datetime 변수를 여러개로 나누어 보자.
Train['date']  = Train.datetime.apply(lambda x: x.split()[0])
Train['hour'] = Train.datetime.apply(lambda x: x.split()[1].split(':')[0])
Train['weekday'] = Train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
Train['month'] = Train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)




# datetime 변수를 여러개로 나누어 보자.
Test['date']  = Test.datetime.apply(lambda x: x.split()[0])
Test['hour'] = Test.datetime.apply(lambda x: x.split()[1].split(':')[0])
Test['weekday'] = Test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
Test['month'] = Test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)




season_dm = pd.get_dummies(Train['season'],prefix='season')
weather_dm = pd.get_dummies(Train['weather'],prefix='weather')
hour_dm = pd.get_dummies(Train['hour'],prefix='hour')
weekday_dm = pd.get_dummies(Train['wwekday'],prefix = 'weekday')
month_dm = pd.get_dummies(Train['month'],prefix='month')
season_dm1 = pd.get_dummies(Test['season'],prefix='season')
weather_dm1 = pd.get_dummies(Test['weather'],prefix='weather')
hour_dm1 = pd.get_dummies(Test['hour'],prefix='hour')
weekday_dm1 = pd.get_dummies(Test['wwekday'],prefix = 'weekday')
month_dm1 = pd.get_dummies(Test['month'],prefix='month')

InputX = InputX.join(season_dm)
InputX = InputX.join(weather_dm)
InputX = InputX.join(hour_dm)
InputX = InputX.join(weekday_dm)
InputX = InputX.join(month_dm)
TestX = TestX.join(season_dm1)
TestX = TestX.join(weather_dm1)
TestX = TestX.join(hour_dm1)
TestX = TestX.join(weekday_dm1)
TestX = TestX.join(month_dm1)




print(TestX.shape)
print(InputX.)





















