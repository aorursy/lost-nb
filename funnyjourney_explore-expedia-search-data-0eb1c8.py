#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import ml_metrics as metrics


# In[2]:


dtype={'is_booking':bool,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'hotel_cluster' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}
# feature selection
# downsample the data: 60% of the 2014 booking data
# originally have 30million training data, 3million test data, but only ~20 features,
# so we can down sample the data

#Specifying dtypes helps reduce memory requirements for reading in csv file later.


# In[3]:


df0 = pd.read_csv('../input/train.csv',dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',',nrows=2000000)


# In[4]:


df0.head()


# In[5]:



# take data from 2014 as sampling 50%
df0['year']=df0['date_time'].dt.year
train = df0.query('is_booking==True & year==2014').sample(frac=0.6)
train.shape


# In[6]:



train.tail()


# In[7]:


train.isnull().sum(axis=0)


# In[8]:


#datetime features
train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')

train['month']= train['date_time'].dt.month
train['plan_time'] = ((train['srch_ci']-train['date_time'])/np.timedelta64(1,'D')).astype(float)
train['hotel_nights']=((train['srch_co']-train['srch_ci'])/np.timedelta64(1,'D')).astype(float)


# In[9]:


train.head()


# In[10]:


#fill Missing Values

#fill orig_destination_distance with mean of the whole or mean of the same orig_destination pair
m=train.orig_destination_distance.mean()
train['orig_destination_distance']=train.orig_destination_distance.fillna(m)

#fill missing dates with -1
train.fillna(-1,inplace=True)


# In[11]:


# Since we extract the plan_time from srch_ci and date_time, we drop date_time and srch_ci
# we extract how many nights of stay, so we drop srch_co
lst_drop=['date_time','srch_ci','srch_co']
train.drop(lst_drop,axis=1,inplace=True)


# In[12]:


train.head()


# In[13]:


y=train['hotel_cluster']
X=train.drop(['hotel_cluster','is_booking','year'],axis=1) # in training dataset, have clicking and booking event


# In[14]:


y.shape,X.shape


# In[15]:


y.nunique()


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
rf_tree = RandomForestClassifier(n_estimators=31,max_depth=10,random_state=123)
rf_tree.fit(X_train,y_train)


# In[17]:


importance = rf_tree.feature_importances_
indices=np.argsort(importance)[::-1][:10]
importance[indices]


# In[18]:


plt.barh(range(10), importance[indices],color='r')
plt.yticks(range(10),X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.show()


# In[19]:


rf_tree.classes_


# In[20]:


dict_cluster = {}
for (k,v) in enumerate(rf_tree.classes_):
    dict_cluster[k] = v


# In[21]:


y_pred=rf_tree.predict_proba(X_test)
#take largest 5 probablities' indexes
a=y_pred.argsort(axis=1)[:,-5:]


# In[22]:


y_pred


# In[23]:


a


# In[24]:



#take the corresonding cluster of the 5 top indices
b = []
for i in a.flatten():
    b.append(dict_cluster.get(i))
cluster_pred = np.array(b).reshape(a.shape)
cluster_pred


# In[25]:


print("score:",metrics.mapk(y_test,cluster_pred,k=5))


# In[26]:


get_ipython().run_line_magic('pinfo', 'metrics.mapk')


# In[27]:


y_test.head()


# In[28]:


#import and process test data
dtype1={'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}


# In[29]:


# feature engineering on test data
test = pd.read_csv('../input/test.csv',dtype=dtype1,usecols=dtype1,parse_dates=['date_time'] ,sep=',')
test['srch_ci']=pd.to_datetime(test['srch_ci'],infer_datetime_format = True,errors='coerce')
test['srch_co']=pd.to_datetime(test['srch_co'],infer_datetime_format = True,errors='coerce')

test['month']=test['date_time'].dt.month
test['plan_time'] = ((test['srch_ci']-test['date_time'])/np.timedelta64(1,'D')).astype(float)
test['hotel_nights']=((test['srch_co']-test['srch_ci'])/np.timedelta64(1,'D')).astype(float)

n=test.orig_destination_distance.mean()
test['orig_destination_distance']=test.orig_destination_distance.fillna(m)
test.fillna(-1,inplace=True)


# In[30]:


test1=test.sample(frac=0.1) # random sampled 5% of the test data


# In[31]:


test1.shape, train.shape


# In[32]:


lst_drop=['date_time','srch_ci','srch_co']
test1.drop(lst_drop,axis=1, inplace=True)
target=train['hotel_cluster']
train1=train.drop(['hotel_cluster','is_booking','year'],axis=1)
train1.shape, test1.shape


# In[33]:


#on All training sample
rf_all = RandomForestClassifier(n_estimators=31,max_depth=10,random_state=123)
rf_all.fit(train1,target)


# In[34]:


importance = rf_all.feature_importances_
indices=np.argsort(importance)[::-1][:10]
importance[indices]

plt.barh(range(10), importance[indices],color='r')
plt.yticks(range(10),train1.columns[indices])
plt.xlabel('Feature Importance')
plt.show()


# In[35]:


y_pred=rf_all.predict_proba(test1) # predict on test dataset
y_pred


# In[36]:


#take largest 5 probablities' indexes
a=y_pred.argsort(axis=1)[:,-5:]


# In[37]:


a


# In[38]:


dict_cluster = {}
for (k,v) in enumerate(rf_tree.classes_):
    dict_cluster[k] = v
b = []
for i in a.flatten():
    b.append(dict_cluster.get(i))
predict_class=np.array(b).reshape(a.shape)


# In[39]:


predict_class


# In[40]:


predict_class=map(lambda x: ' '.join(map(str,x)), predict_class)


# In[41]:


submission = pd.DataFrame()
submission['hotel_cluster'] = predict_class
submission.to_csv('rf01expedia.csv', index=False)


# In[42]:





# In[42]:





# In[42]:


# IMPORTANT! - Another Method for Hotel Cluster Prediction
# Expedia Hotel Cluster Predictions
# Link: https://www.kaggle.com/omarelgabry/expedia-hotel-recommendations/expedia-hotel-cluster-predictions

