#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[2]:


train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')


# In[3]:


train.head()


# In[4]:


test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')


# In[5]:


test.head()


# In[6]:


train.isnull().sum()


# In[7]:


test.isnull().sum()


# In[8]:


submission=pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')


# In[9]:


submission.head()


# In[10]:


train.shape+test.shape


# In[11]:


data = pd.concat([train, test])


# In[12]:


data.head()


# In[13]:


data.isnull().sum()


# In[14]:


data['Province_State']=data['Province_State'].fillna('PS',inplace=True)


# In[15]:


data.isnull().sum()


# In[16]:


train_len=len(train)


# In[17]:


data.describe()


# In[18]:


trg=['ConfirmedCases','Fatalities']
features = ["past__{}".format(col) for col in trg]
    


# In[ ]:





# In[19]:


for cols in data.columns:
    if (data[cols].dtype==np.number):
        continue
    data[cols]=LabelEncoder().fit_transform(data[cols])


# In[20]:


train=data[:train_len]


# In[21]:


train = train.drop('ForecastId',axis=1)


# In[22]:


test=data[train_len:]


# In[23]:


drop=['Id','ConfirmedCases','Fatalities']
test = test.drop(drop,axis=1)


# In[24]:


train.head()


# In[25]:


test.head()


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


model = LogisticRegression(random_state=71,n_jobs=-1,verbose=0)


# In[28]:


x_train=train.drop(labels=['Fatalities','ConfirmedCases','Id'],axis=1)
y_train1=train['ConfirmedCases']
y_train2=train['Fatalities']


# In[29]:


m1=model.fit(x_train,y_train1)


# In[30]:


m2=model.fit(x_train,y_train2)


# In[31]:


x_test=test.drop(labels=['ForecastId'],axis=1)


# In[32]:


pred1=m1.predict(x_test)


# In[33]:


pred2=m2.predict(x_test)


# In[34]:


data_to_submit = pd.DataFrame({
    'ForecastId':submission['ForecastId'],
    'ConfirmedCases':pred1,
    'Fatalities':pred2
})
data_to_submit.to_csv('submission.csv', index = False)


# In[ ]:




