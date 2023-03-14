#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')




df_train.head()




df_train.info()




print("Number of Country_Region: ", df_train['Country_Region'].nunique())
print("Dates are ranging from day", min(df_train['Date']), "to day", max(df_train['Date']), ", a total of", df_train['Date'].nunique(), "days")
print("The countries that have Province/Region given are : ", df_train[df_train['Province_State'].isna()==False]['Country_Region'].unique())




df_train.columns




df_train['Province_State'].unique()




plt.figure(figsize=(40,40))
temp_df= df_train[df_train['ConfirmedCases']>5000]
sns.barplot(y = temp_df['Country_Region'] , x = temp_df['ConfirmedCases']>10000)
sns.set_context('paper')
plt.ylabel("Country_Region",fontsize=30)
plt.xlabel("Counts",fontsize=30)
plt.title("Counts of Countries affected by the pandemic that have confirmed cases > 5000",fontsize=30)
plt.xticks(rotation = 90)




confirmed_total_dates = df_train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_dates = df_train.groupby(['Date']).agg({'Fatalities':['sum']})
total_dates = confirmed_total_dates.join(fatalities_total_dates)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_dates.plot(ax=ax1)
ax1.set_title("Global confirmed cases", size=13)
ax1.set_ylabel("Total Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_dates.plot(ax=ax2, color='orange')
ax2.set_title("Global deceased cases", size=13)
ax2.set_ylabel("Total Number of cases", size=13)
ax2.set_xlabel("Date", size=13)




italy = df_train[df_train['Country_Region'] == 'Italy']
plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'ConfirmedCases' , data = italy)
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
plt.title('Confirmed Cases per date in Italy',size=20)
plt.show()




italy = df_train[df_train['Country_Region'] == 'Italy']
plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'Fatalities' , data = italy,color='orange')
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Fatalities',size=15)
plt.title('Fatalities in Italy per Date',size=20)
plt.show()




usa = df_train[df_train['Country_Region'] == 'US']
plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'ConfirmedCases' , data = usa,color='g')
plt.xticks(rotation = 90,size=13)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
plt.title('Confirmed Cases in US per Date',size=20)
plt.show()




plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'Fatalities' , data = usa,color='purple')
plt.title('Fatalities in US per Date',size=20)
plt.xticks(rotation = 90,size=13)
plt.xlabel('Date',size=15)
plt.ylabel('Fatalities',size=15)
plt.show()




plt.figure(figsize=(20,10))
sns.barplot(x='Province_State',y='ConfirmedCases',data=usa,ci=None)
plt.xticks(rotation = 90,size=13)
plt.xlabel('Province_State',size=15)
plt.ylabel('Confirmed Cases',size=15)
plt.title('Confirmed Cases in US Province_State ',size=20)
plt.show()




#we now do the analysis of NYC as per week.
import warnings
warnings.filterwarnings('ignore')
temp_df = usa[usa['Province_State'] == 'New York']
temp_df['Date'] = pd.to_datetime(temp_df['Date'])
temp_df.insert(6,'Week',temp_df['Date'].dt.week)
f,axes = plt.subplots(1,2,figsize=(12,5))
sns.lineplot(x = 'Week',y = 'ConfirmedCases',color='r',data=temp_df,ax = axes[0])
sns.lineplot(x = 'Week',y = 'Fatalities',color='b',data=temp_df,ax = axes[1])

axes[0].title.set_text('Confirmed Cases in NYC per week')
axes[1].title.set_text('Fatalities in NYC per week')




china  = df_train[df_train['Country_Region'] == 'China']

plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'ConfirmedCases' , data = china,color='aqua')
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
sns.set_context('paper')
plt.title('Confirmed Cases in China per Date',size=20)
plt.show()




china  = df_train[df_train['Country_Region'] == 'China']

plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'Fatalities' , data = china,color='grey')
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Fatalities',size=15)
sns.set_context('paper')
plt.title('Fatalities in China per Date',size=20)
plt.show()




plt.figure(figsize=(20,10))
sns.barplot(x='Province_State',y='ConfirmedCases',data=china)
plt.xticks(rotation = 90,size=13)
plt.title('Confirmed Cases in China Province_State',size=20)
plt.ylabel('Confirmed Cases',size=15)
plt.xlabel('Province_State',size=15)
plt.show()




#we now do the analysis of Hubei as per week.
import warnings
warnings.filterwarnings('ignore')
china_t = china[china['Province_State'] == 'Hubei']
china_t['Date'] = pd.to_datetime(china_t['Date'])
china_t.insert(6,'Week',china_t['Date'].dt.week)
f,axes = plt.subplots(1,2,figsize=(12,5))
sns.lineplot(x = 'Week',y = 'ConfirmedCases',color='r',data=china_t,ax = axes[0])
sns.lineplot(x = 'Week',y = 'Fatalities',color='b',data=china_t,ax = axes[1])

axes[0].title.set_text('Confirmed Cases in Hubei per week')

axes[1].title.set_text('Fatalities in Hubei per week')




df_train = df_train[['Date','Province_State','Country_Region','ConfirmedCases','Fatalities']]
df_train.head()




#Using pd.to_datetime for adding new features
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train.insert(1,'Week',df_train['Date'].dt.week)
df_train.insert(2,'Day',df_train['Date'].dt.day)
df_train.insert(3,'DayofWeek',df_train['Date'].dt.dayofweek)
df_train.insert(4,'DayofYear',df_train['Date'].dt.dayofyear)

df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test.insert(1,'Week',df_test['Date'].dt.week)
df_test.insert(2,'Day',df_test['Date'].dt.day)
df_test.insert(3,'DayofWeek',df_test['Date'].dt.dayofweek)
df_test.insert(4,'DayofYear',df_test['Date'].dt.dayofyear)




df_train.head()




# Replacing all the Province_State that are null by the Country_Region values
df_train.Province_State.fillna(df_train.Country_Region, inplace=True)
df_test.Province_State.fillna(df_test.Country_Region, inplace=True)




df_train.head()




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_train.Country_Region = le.fit_transform(df_train.Country_Region)
df_train['Province_State'] = le.fit_transform(df_train['Province_State'])

df_test.Country_Region = le.fit_transform(df_test.Country_Region)
df_test['Province_State'] = le.fit_transform(df_test['Province_State'])




#One Hot Encoding columns
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    i = 0
    for each in cols:
        #print (each)
        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)
        if i == 0: 
            print (dummies)
            i = i + 1
        df = pd.concat([df, dummies], axis=1)
    return df




#Handling categorical data

objList = df_train.select_dtypes(include = "object").columns
df_train = one_hot(df_train, objList) 
df_test = one_hot(df_test, objList) 

print (df_train.shape)




#Avoiding duplicated data.
df_train = df_train.loc[:,~df_train.columns.duplicated()]
df_test = df_test.loc[:,~df_test.columns.duplicated()]
print (df_test.shape)




# Dropping the object type columns
df_train.drop(objList, axis=1, inplace=True)
df_test.drop(objList, axis=1, inplace=True)
print (df_train.shape)




df_train.head()




#Selecting only the type Object Columns
df_train.select_dtypes(include = "object").columns




df_train




X = df_train.drop(['Date', 'ConfirmedCases', 'Fatalities'], axis=1)
y = df_train[['ConfirmedCases', 'Fatalities']]




from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error
from sklearn.ensemble import BaggingRegressor




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




y_train.head()




n_folds = 5
cv = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(X_train.values)




def predict_scores(reg_alg):
    r2 = make_scorer(r2_score)
    m = reg_alg()
    m.fit(X_train, y_train['ConfirmedCases'])
    y_pred = m.predict(X_test)
    m_r = cross_val_score(m, X_train, y_train['ConfirmedCases'], cv=cv, scoring = r2)
    sc_Cases.append(m_r)
    
    m.fit(X_train, y_train['Fatalities'])
    y_pred = m.predict(X_test)
    m_r2 = cross_val_score(m, X_train, y_train['Fatalities'], cv=cv, scoring = r2)
    sc_Fatalities.append(m_r2)


    
reg_models = [KNeighborsRegressor, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor,BayesianRidge,
              BaggingRegressor]

sc_Cases = []
sc_Fatalities = []

for x in reg_models:
    predict_scores(x)




sc_Cases




sc_Fatalities




from sklearn.ensemble import BaggingRegressor





#Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

param_grid = {
              'n_estimators':[10, 30, 50, 100,250,500,750,1000,1250,1500,1750],
              'max_samples':[2,4,6,8,10,20,40,60,100],
              "max_features": [0.5, 1.0],
              'n_jobs':[-2, -1, 1, 2, 3, 4, 5],
              "bootstrap_features": [True, False]
             }
'''param_grid = {"criterion": ["mae"],
              "min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              }'''

asdf = BaggingRegressor()


clf_CC = RandomizedSearchCV(asdf, param_grid )
clf_Fat = RandomizedSearchCV(asdf, param_grid )

clf_CC.fit(X_train, y_train['ConfirmedCases'])
clf_Fat.fit(X_train, y_train['Fatalities'])




model1 = clf_CC
model1.fit(X_train, y_train['ConfirmedCases'])

model2 = clf_Fat
model2.fit(X_train, y_train['Fatalities'])




df_test['ConfirmedCases'] = model1.predict(df_test.drop(['Date', 'ForecastId'], axis=1))
df_test['Fatalities'] = model2.predict(df_test.drop(['Date', 'ForecastId', 'ConfirmedCases'], axis=1))




import warnings
warnings.filterwarnings('ignore')
df_results = df_test[['ForecastId', 'ConfirmedCases', 'Fatalities']] 
df_results['ConfirmedCases'] = df_results['ConfirmedCases'].astype(int)
df_results['Fatalities'] = df_results['Fatalities'].astype(int)

df_results.head()




df_results.to_csv('submission.csv', index=False)

