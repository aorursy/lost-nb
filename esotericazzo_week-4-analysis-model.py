#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# In[3]:


df_train.describe()


# In[4]:


df_train.head()


# In[5]:


print("Number of Country_Region: ", df_train['Country_Region'].nunique())
print("Dates are ranging from day", min(df_train['Date']), "to day", max(df_train['Date']), ", a total of", df_train['Date'].nunique(), "days")
print("The countries that have Province/Region given are : ", df_train[df_train['Province_State'].isna()==False]['Country_Region'].unique())


# In[6]:


df = df_train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)
top10 = pd.DataFrame(df).head(10)
top10


# In[7]:


top10.columns


# In[8]:


plt.figure(figsize=(20,10))
sns.barplot(x = top10.index , y = top10['ConfirmedCases'])
sns.set_context('paper')
plt.xlabel("Country_Region",fontsize=30)
plt.ylabel("Counts",fontsize=30)
plt.title("Counts of Countries affected by the pandemic that have maximum cases",fontsize=30)
plt.xticks(rotation = 45,fontsize=12)


# In[9]:


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


# In[10]:


usa = df_train[df_train['Country_Region'] == 'US']


# In[11]:


usa.drop('Id',axis=1,inplace=True)


# In[12]:


usa.set_index('Date')


# In[13]:


usa_1 = pd.DataFrame(usa.groupby(['Province_State'])['Fatalities'].max().sort_values())
usa_1['ConfirmedCases'] = usa.groupby(['Province_State'])['ConfirmedCases'].max().sort_values()
usa_1.head(10)


# In[14]:


plt.figure(figsize=(20,10))
sns.barplot(x = usa_1.index,y=usa_1['ConfirmedCases'])
sns.set_context('paper')
plt.xticks(rotation=90)
plt.title('States affected by the pandemic',fontsize=15)


# In[15]:


#we now do the analysis of NYC as per week.
import warnings
warnings.filterwarnings('ignore')
temp_df = usa[usa['Province_State'] == 'New York']
temp_df['Date'] = pd.to_datetime(temp_df['Date'])
temp_df.insert(5,'Week',temp_df['Date'].dt.week)
f,axes = plt.subplots(1,2,figsize=(12,5))
sns.lineplot(x = 'Week',y = 'ConfirmedCases',color='r',data=temp_df,ax = axes[0])
sns.lineplot(x = 'Week',y = 'Fatalities',color='b',data=temp_df,ax = axes[1])

axes[0].title.set_text('Confirmed Cases in NYC per week')
axes[1].title.set_text('Fatalities in NYC per week')


# In[16]:


plt.figure(figsize=(20,10))
sns.lineplot(x = 'Date' , y = 'ConfirmedCases' , data = usa,color='g')
plt.xticks(rotation = 90,size=13)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
plt.title('Confirmed Cases in US per Date',size=20)
plt.show()


# In[17]:


plt.figure(figsize=(20,10))
sns.barplot(x = 'Date' , y = 'Fatalities' , data = usa,color='purple')
plt.title('Fatalities in US per Date',size=20)
plt.xticks(rotation = 90,size=13)
plt.xlabel('Date',size=15)
plt.ylabel('Fatalities',size=15)
plt.show()


# In[18]:


china = df_train[df_train['Country_Region'] == 'China']
df_china = pd.DataFrame(china.groupby(['Date','Country_Region'])['ConfirmedCases'].sum().reset_index())


# In[19]:


plt.figure(figsize=(20,5))
sns.barplot(x = df_china['Date'] ,y = df_china['ConfirmedCases'])
plt.title('Confirmed Cases in China per day',size=20)
plt.xticks(rotation=90)
plt.xlabel('Date',fontsize=15)
plt.ylabel('ConfirmedCases',fontsize=15)


# In[20]:


spain = pd.DataFrame(df_train[df_train['Country_Region'] == 'Spain'])
spain.drop('Id',axis=1,inplace=True)
spain.set_index('Date',inplace=True)


# In[21]:


plt.figure(figsize=(20,10))
sns.barplot(x = spain.index , y = 'ConfirmedCases' , data = spain,color='aqua')
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
sns.set_context('paper')
plt.title('Confirmed Cases in Spain per Date',size=20)
plt.show()


# In[22]:


plt.figure(figsize=(20,10))
sns.barplot(x = spain.index , y = 'Fatalities' , data = spain,color='red')
plt.xticks(rotation = 90,size=12)
plt.xlabel('Date',size=15)
plt.ylabel('Confirmed Cases',size=15)
sns.set_context('paper')
plt.title('Confirmed Cases in Spain per Date',size=20)
plt.show()


# In[23]:


italy = df_train[df_train['Country_Region'] == 'Italy']
df_italy = pd.DataFrame(italy.groupby(['Date','Country_Region'])['ConfirmedCases'].sum().reset_index())


# In[24]:


plt.figure(figsize=(20,5))
sns.barplot(x = df_italy['Date'] ,y = df_italy['ConfirmedCases'])
plt.title('Confirmed Cases in Italy per Date',size=20)
plt.xticks(rotation=90)
plt.xlabel('Date',fontsize=15)
plt.ylabel('ConfirmedCases',fontsize=15)


# In[25]:


# 1. Converting the object type column into datetime type
df_train['Date'] = df_train.Date.apply(pd.to_datetime)
df_test['Date'] = df_test.Date.apply(pd.to_datetime)

#Extracting Date and Month from the datetime and converting the feature as int
#df_train.Date = df_train.Date.dt.strftime("%m%d")
#df_test.Date = df_test.Date.dt.strftime("%m%d")


# In[26]:


df_train.insert(1,'Month',df_train['Date'].dt.month)

df_train.insert(2,'Day',df_train['Date'].dt.day)


# In[27]:


df_train.head()


# In[28]:


df_test.insert(1,'Month',df_test['Date'].dt.month)

df_test.insert(2,'Day',df_test['Date'].dt.day)


# In[29]:


df_test.head()


# In[30]:


df_train['Province_State'].fillna(df_train['Country_Region'],inplace=True)


# In[31]:


df_test['Province_State'].fillna(df_test['Country_Region'],inplace=True)


# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_train.Country_Region = le.fit_transform(df_train.Country_Region)
df_train['Province_State'] = le.fit_transform(df_train['Province_State'])

df_test.Country_Region = le.fit_transform(df_test.Country_Region)
df_test['Province_State'] = le.fit_transform(df_test['Province_State'])


# In[33]:


#Avoiding duplicated data.
df_train = df_train.loc[:,~df_train.columns.duplicated()]
df_test = df_test.loc[:,~df_test.columns.duplicated()]
print (df_test.shape)


# In[34]:


# Dropping the object type columns

objList = df_train.select_dtypes(include = "object").columns
df_train.drop(objList, axis=1, inplace=True)
df_test.drop(objList, axis=1, inplace=True)
print (df_train.shape)


# In[35]:


df_train.drop('Date',axis=1,inplace=True)


# In[36]:


df_test.drop('Date',axis=1,inplace=True)


# In[37]:


df_train.head()


# In[38]:


X = df_train.drop(['Id','ConfirmedCases', 'Fatalities'], axis=1)
y = df_train[['ConfirmedCases', 'Fatalities']]


# In[39]:


from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error
skfold = ShuffleSplit(random_state=7)


# In[40]:


from sklearn.tree import DecisionTreeRegressor


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[42]:


clf_CC = DecisionTreeRegressor()
clf_Fat = DecisionTreeRegressor()

dec_cc = cross_val_score(clf_CC, X_train, y_train['ConfirmedCases'], cv = skfold)
dec_fat = cross_val_score(clf_Fat, X_train, y_train['Fatalities'], cv = skfold)

print (dec_cc.mean(), dec_fat.mean())


# In[43]:


X_test_CC = df_test.drop(['ForecastId'],axis=1)
X_test_Fat = df_test.drop(['ForecastId'],axis=1)


# In[44]:


clf_CC.fit(X_train, y_train['ConfirmedCases'])
Y_pred_CC = clf_CC.predict(X_test_CC) 

clf_Fat.fit(X_train, y_train['Fatalities'])
Y_pred_Fat = clf_Fat.predict(X_test_Fat) 


# In[45]:


df_cc = pd.DataFrame(Y_pred_CC)


# In[46]:


df_fat = pd.DataFrame(Y_pred_Fat)


# In[47]:


import warnings
warnings.filterwarnings('ignore')

# Calling DataFrame constructor on list 
df_results = pd.DataFrame(columns=['ForecastId','ConfirmedCases','Fatalities']) 
df_results


# In[48]:


df_results['ForecastId'] = df_test['ForecastId']
df_results['ConfirmedCases'] = df_cc.astype(int)
df_results['Fatalities'] = df_fat.astype(int)

df_results.head()


# In[49]:


df_results.to_csv('submission.csv', index=False)


# In[ ]:




