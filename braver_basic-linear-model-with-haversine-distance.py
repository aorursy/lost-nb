#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import packages
import numpy as np
import pandas as pd
import datetime as dt
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from haversine import haversine
import seaborn as sns


# In[2]:


#Import files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


#let's take a brief look at the data files
train.head()


# In[4]:


test.head()


# In[5]:


#Apply for train set
print("start: ", dt.datetime.now())
train['distance'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),
                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
print("finish: ", dt.datetime.now())


# In[6]:


#Apply for test set
print("start: ", dt.datetime.now())
test['distance'] = test.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),
                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
print("finish: ", dt.datetime.now())


# In[7]:


#There are some outliers found 
#Show histogram to determine reasonable outlier level
train.loc[train['trip_duration']<10000,'trip_duration'].hist(bins=100)


# In[8]:


#For now I'm cutting outliers at two hours, 7200 seconds
max_duration = 60*60*2 #Two hours
train = train[train['trip_duration']<max_duration]


# In[9]:


#look at relation of distance vs. time
sns.set(style="whitegrid")
bins = (train['distance']/5).round(0)*5

ax = sns.lvplot(x=bins, y=train['trip_duration'], scale="linear")
ax.locator_params(axis='x', nbins=10)
plt.show()


# In[10]:


train['distance_sqrt'] = np.sqrt(train['distance'])
train['distance_2'] = train['distance']**2
test['distance_sqrt'] = np.sqrt(test['distance'])
test['distance_2'] = test['distance']**2
cols = ['distance','distance_sqrt','distance_2']
X_train = train[cols]
X_test = test[cols]
y_train = train['trip_duration']
y_test = train['trip_duration']
X_train_train, X_train_test, y_train_train, y_train_test =     model_selection.train_test_split(X_train, y_train, test_size=0.33, random_state = 42)


# In[11]:


lm = linear_model.LinearRegression()


# In[12]:


#Train the linear regressor using the trai|n_train set
y_train_train_log = np.log(y_train_train + 1)
lm.fit(X_train_train, y_train_train_log)
print('Inner train_train score:', lm.score(X_train_train,y_train_train_log))
print('Coefficients: \n', lm.coef_)


# In[13]:


#Score the train_test performance
pred = lm.predict(X_train_test)
pred = np.exp(pred) - 1
y_train_test_log = np.log(y_train_test + 1)
print('Score on train_test set:', lm.score(X_train_test, y_train_test_log))


# In[14]:


err = pd.DataFrame()
err['y'] = y_train_test
err['pred'] = pred
err['error'] = err['y'] - err['pred']
err = err[(err['error']>-10000)]
err['error'].hist(bins=100)


# In[15]:


#Analyse prediction error on trip_duration axis
plt.scatter(err['y'], err['error'])


# In[16]:


y_train_log = np.log(y_train + 1)
#Train the model using the full train set
lm.fit(X_train, y_train_log)
print('Inner train score:', lm.score(X_train,y_train))
print('Coefficients: \n', lm.coef_)


# In[17]:


#Make the prediction for the full test set
pred = lm.predict(X_test)
pred = np.exp(pred) - 1


# In[18]:


submission = pd.DataFrame()
submission['id'] = test['id']
submission['trip_duration'] = pred
submission.loc[submission['trip_duration']<0,'trip_duration'] = 0
submission.to_csv('submission.csv', index=False)


# In[19]:


submission.head()


# In[20]:




