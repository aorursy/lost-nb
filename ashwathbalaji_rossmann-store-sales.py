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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')


# In[3]:


print(train.shape)
print(test.shape)
print(store.shape)


# In[4]:


train.head()


# In[5]:


store.head()


# In[6]:


test.head()


# In[7]:


train.dtypes


# In[8]:


store.dtypes


# In[9]:


train.describe(include='object')


# In[10]:


train.describe()[['Sales','Customers']]


# In[11]:


train.Store.nunique()


# In[12]:


train.DayOfWeek.value_counts().sort_values()


# In[13]:


print(train.Open.value_counts() , '\n',train.Promo.value_counts())


# In[14]:


print(train.isna().sum())
print('-'*20)
print(store.isna().sum())
print('-'*20)
print(test.isna().sum())


# In[15]:


store1 = train[train['Store']==1]
store1.head()


# In[16]:


print(store1.shape)


# In[17]:


store1['Date'] = pd.to_datetime(store1['Date'])
print(min(store1['Date']))
print(max(store1['Date']))
store1['Year'] = store1['Date'].dt.year
store1['Month'] = store1['Date'].dt.month


# In[18]:


store1.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(14,4))
plt.show()


# In[19]:


import seaborn as sns
sns.distplot(store1.Sales , bins=10)
plt.show()


# In[20]:


sns.distplot(train.Sales)
plt.show()


# In[21]:


store.isna().sum()


# In[22]:


store[store['Store']==1].T


# In[23]:


store[~(store['Promo2']==0)].iloc[0]


# In[24]:


store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])


# In[25]:


store.isna().sum()


# In[26]:


df = train.merge(store , on='Store' , how='left')
print(train.shape)
print(store.shape)
print(df.shape)


# In[27]:


df.head(3)


# In[28]:


df.isna().sum()


# In[29]:


df['Date'] = pd.to_datetime(df['Date'])


# In[30]:


df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# df['DayOfWeek'] = df['Date'].dt.strftime(%a)


# In[31]:


df.dtypes


# In[32]:


cat_cols = df.select_dtypes(include=['object']).columns

for i in cat_cols:
    print(i)
    print(df[i].value_counts())
    print('-'*20)


# In[33]:


df['StateHoliday'] = df['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})
df['StateHoliday'] = df['StateHoliday'].astype(int)


# In[34]:


df['StoreType'] = df['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
df['StoreType'] = df['StoreType'].astype(int)


# In[35]:


df['Assortment'] = df['Assortment'].map({'a':1 , 'b':2 , 'c':3})
df['Assortment'] = df['Assortment'].astype(int)


# In[36]:


df['PromoInterval'] = df['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
df['PromoInterval'] = df['PromoInterval'].astype(int)


# In[37]:


df.dtypes


# In[38]:


X = df.drop(['Sales','Date','Customers'],1)
#Transform Target Variable
y = np.log(df['Sales']+1)

from sklearn.model_selection import train_test_split
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.30 , random_state = 1 )

X_train.shape , X_val.shape , y_train.shape , y_val.shape


# In[39]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=11)
dt.fit(X_train , y_train)
y_pred_dt = dt.predict(X_val)


# In[40]:


y_pred_dt = np.exp(y_pred_dt)-1
y_val = np.exp(y_val)-1


# In[41]:


from sklearn.metrics import r2_score , mean_squared_error

print(r2_score(y_val , y_pred_dt))
print(np.sqrt(mean_squared_error(y_val , y_pred_dt)))


# In[42]:


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


# In[43]:


rmspe(y_val,y_pred_dt)


# In[44]:


def get_rmspe_score(model, input_values, y_actual):
    y_predicted=model.predict(input_values)
    y_actual=np.exp(y_actual)-1
    y_predicted=np.exp(y_predicted)-1
    score=rmspe(y_actual, y_predicted)
    return score


# In[45]:


from sklearn.model_selection import RandomizedSearchCV

params = {
    'max_depth' : list(range(5,25))
}

base  = DecisionTreeRegressor()

model_tuned = RandomizedSearchCV(base , params , return_train_score=True).fit(X_train , y_train)


# In[46]:


model_cv_results = pd.DataFrame(model_tuned.cv_results_).sort_values(by='mean_test_score' , ascending=False)
model_cv_results


# In[47]:


model_cv_results.set_index('param_max_depth')['mean_test_score'].plot(color='g',legend=True)
model_cv_results.set_index('param_max_depth')['mean_train_score'].plot(color='r' , legend=True)
plt.grid(True)
plt.show()


# In[48]:


import xgboost as xgb


# In[49]:


dtrain = xgb.DMatrix(X_train,y_train)
dvalidate = xgb.DMatrix(X_val[X_train.columns],y_val)

params = {
    'eta' : 1,
    'max_depth' : 5,
    'objecive' : 'reg:linear'
}

model_xg = xgb.train(params, dtrain , 5)

y_pred_xg = model_xg.predict(dvalidate)

y_pred_xg = np.exp(y_pred_xg)-1


rmspe(y_val , y_pred_xg)

ROOT MEAN SQUARE PERCENTAGE ERROR
# In[50]:


plt.barh(X_train.columns , dt.feature_importances_)
plt.show()


# In[51]:


test.shape


# In[52]:


test.head()


# In[53]:


test_cust = train.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)


# In[54]:


test_1 = test.merge(test_cust , on='Store' , how='left')
test_1.head()


# In[55]:


test_m = test_1.merge(store , on='Store' , how='left')


# In[56]:


test_m.shape


# In[57]:


test_m['Open'].fillna(1,inplace=True)

test_m['Date'] = pd.to_datetime(test_m['Date'])

test_m['Day'] = test_m['Date'].dt.day
test_m['Month'] = test_m['Date'].dt.month
test_m['Year'] = test_m['Date'].dt.year

test_m.drop('Date',1,inplace=True)


# In[58]:


cat_cols = test_m.select_dtypes(include=['object']).columns

for i in cat_cols:
    print(i)
    print(test_m[i].value_counts())
    print('-'*20)


# In[59]:


test_m['StateHoliday'] = test_m['StateHoliday'].map({'0':0 , 'a':1})
test_m['StateHoliday'] = test_m['StateHoliday'].astype(int)

test_m['StoreType'] = test_m['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
test_m['StoreType'] = test_m['StoreType'].astype(int)

test_m['Assortment'] = test_m['Assortment'].map({'a':1 , 'b':2 , 'c':3})
test_m['Assortment'] = test_m['Assortment'].astype(int)

test_m['PromoInterval'] = test_m['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
test_m['PromoInterval'] = test_m['PromoInterval'].astype(int)


# In[60]:


test_m.dtypes


# In[61]:


X_train.dtypes


# In[62]:


test_m.isna().sum()


# In[63]:


test_pred = dt.predict(test_m[X_train.columns])
test_pred_inv = np.exp(test_pred)-1


# In[64]:


test_pred_inv


# In[65]:


submission = pd.DataFrame({'Id' : test_m['Id'] , 'Sales' : test_pred_inv})
submission['Sales'] = submission['Sales'].astype(int)
submission['Id']= submission.index
submission['Id'] = submission['Id']+1
submission.head()


# In[66]:


submission.shape


# In[67]:


submission


# In[68]:


submission.to_csv('sumbission.csv',index=False)

