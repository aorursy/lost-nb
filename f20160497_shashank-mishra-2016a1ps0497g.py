#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df=pd.read_csv("../input/train.csv")
#train_df=pd.get_dummies(train_df,columns=["year"])
train_df=train_df.drop(["Total Bags","type"],axis=1)
train_df.head()


# In[3]:


y=train_df["AveragePrice"]
y1=train_df[(train_df["id"]<=9125)]
y1=y1["AveragePrice"]
y1=y1.reset_index(drop=True)
y2=train_df[(train_df["id"]>9125)]
y2=y2["AveragePrice"]
y2=y2.reset_index(drop=True)
#y=np.multiply(y,100)
y2.head()


# In[4]:


x=train_df.drop(["AveragePrice"],axis=1)


# In[5]:


x1=x[(x['id']<=9125)]
x1=x1.reset_index(drop=True)
x1.head()


# In[6]:


x2=x[(x['id']>9125)]
x2=x2.reset_index(drop=True)
x2.head()


# In[7]:


test_df=pd.read_csv("../input/test.csv")
#test_df=pd.get_dummies(test_df,columns=["year"])
test_df.head()


# In[8]:


FOODID=pd.read_csv("../input/test.csv", usecols = ['id'])
FOODID.head()


# In[9]:


test_df=test_df.drop(["Total Bags","type"],axis=1)
test_df.head()


# In[10]:


from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5,shuffle=False,random_state=12)
kf.get_n_splits(x)
print(kf)


# In[11]:


predicted_y = []
expected_y = []
for train_index, test_index in kf.split(x):
    #print("TRAIN:", train_index, "\nTEST:", test_index)
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model =xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1,min_child_weight=0.25, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=10)
    model.fit(x_train,y_train)
    predicted_y.extend(model.predict(x_test))
    expected_y.extend(y_test)
mse =mean_squared_error(expected_y,predicted_y)
print(mse)


# In[12]:


kf = KFold(n_splits=5,shuffle=False,random_state=12)
kf.get_n_splits(x1)
print(kf)


# In[13]:


predicted_y = []
expected_y = []
for train_index, test_index in kf.split(x1):
    #print("TRAIN:", train_index, "\nTEST:", test_index)
    x_train, x_test = x1.loc[train_index], x1.loc[test_index]
    y_train, y_test = y1[train_index], y1[test_index]
    model1 =xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1,min_child_weight=0.25, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=10)
    model1.fit(x_train,y_train)
    predicted_y.extend(model1.predict(x_test))
    expected_y.extend(y_test)
mse =mean_squared_error(expected_y,predicted_y)
print(mse)    


# In[14]:


model1.get_booster().get_score(importance_type='weight')


# In[15]:


kf = KFold(n_splits=5,shuffle=False,random_state=12)
kf.get_n_splits(x2)
print(kf)


# In[16]:


predicted_y = []
expected_y = []
for train_index, test_index in kf.split(x2):
    #print("TRAIN:", train_index, "\nTEST:", test_index)
    x_train, x_test = x2.loc[train_index], x2.loc[test_index]
    y_train, y_test = y2[train_index], y2[test_index]
    model2 =xgboost.XGBRegressor(n_estimators=2000, learning_rate=0.1,min_child_weight=0.25, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=8)
    model2.fit(x_train,y_train)
    predicted_y.extend(model2.predict(x_test))
    expected_y.extend(y_test)
mse =mean_squared_error(expected_y,predicted_y)
print(mse)


# In[17]:


model2.get_booster().get_score(importance_type='weight')


# In[18]:


print(expected_y)
print(len(expected_y))


# In[19]:


print(predicted_y)
print(len(predicted_y))


# In[20]:


predicted_price=[]
for i in range(len(test_df)):
    xx=test_df[i:i+1]
    #xx=xx.reshape(1,15)
    print("{0} {1} {2}".format(model.predict(xx),model1.predict(xx),model2.predict(xx)))
    if test_df.iloc[i]['id']<=9125:
        #print(1111111)
        predicted_price.extend((model1.predict(xx)*0.75+model.predict(xx)*0.25))
    else :
        #print(2222222)
        predicted_price.extend((model2.predict(xx)+model.predict(xx))/2)
print(predicted_price)
print(len(predicted_price))


# In[21]:


FOODID.head()


# In[22]:


final  = pd.concat([FOODID, pd.DataFrame(predicted_price)], axis=1)
final.head(20)


# In[23]:


final.to_csv("result_last.csv", index=False)


# In[24]:




