#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('../input/train_v2.csv')
transactions = pd.read_csv('../input/transactions_v2.csv')
members = pd.read_csv('../input/members_v3.csv')


# In[3]:


#user_logs = pd.read_csv('../input/user_logs.csv')
transactions.head(1)


# In[4]:


print(transactions.msno.nunique())
print(train.msno.nunique())


# In[5]:


train_transactions = pd.merge(train, transactions, on='msno', how='left')
train_transactions.head(1)


# In[6]:


composite = pd.merge(train_transactions, members, on='msno', how='left')
composite.head(1)


# In[7]:


print(composite.msno.nunique())
composite.shape


# In[8]:


composite['membership_expire_date'] = pd.to_datetime(composite['membership_expire_date'], format='%Y%m%d')
composite['transaction_date'] = pd.to_datetime(composite['transaction_date'], format='%Y%m%d')


# In[9]:


mask = composite['registration_init_time'].notnull()
composite.loc[composite[mask].index, 'registration_init_time'
             ] = composite.loc[composite[mask].index, 'registration_init_time'].astype('int')


# In[10]:


composite.head(1)


# In[11]:


composite['registered_via'].value_counts()


# In[12]:


def churn_rate(df, col):
    col_rate = df.groupby(col)[['is_churn']].mean().reset_index().sort_values(
        'is_churn', ascending=False)
    sns.barplot(x=col, y='is_churn', data=col_rate)
    print(plt.show())
    return col_rate


# In[13]:


rv = churn_rate(composite,'registered_via')


# In[14]:


rv


# In[15]:


composite.is_churn.mean()


# In[16]:


cancel = churn_rate(composite,'is_cancel')


# In[17]:


cancel


# In[18]:


composite['registration_init_time'] = pd.to_datetime(
    composite['registration_init_time'], format='%Y%m%d')


# In[19]:


composite.head(1)


# In[20]:


most_recent = composite['registration_init_time'].max()


# In[21]:


composite['registration_init_weeks'] = (
    most_recent - composite.registration_init_time)/ np.timedelta64(1, 'W')


composite.head(1)


# In[22]:


composite['registration_init_weeks'] = composite['registration_init_weeks'].round(decimals=0)


# In[23]:


weeks = composite.groupby('registration_init_weeks')[
    ['is_churn']].mean().reset_index().sort_values('is_churn',ascending=False)
weeks 
weeks.head(10)


# In[24]:


sns.barplot(x='registration_init_weeks', y='is_churn', data=weeks.head(10))
plt.show()


# In[25]:


mask = (composite['registration_init_weeks']>=0)&(composite['registration_init_weeks']<=12)
composite[mask]['registration_init_weeks'].value_counts()


# In[26]:


composite['registration_init_weeks'].value_counts(ascending=False).head(10)


# In[27]:


fig, ax = plt.subplots(figsize=(10,6))
ax.plot_date(composite['registration_init_time'], composite['is_churn', color="blue", linestyle="-")

ax.set(xlabel='date', ylabel='churn',title='Click Rate Over Time')
ax.legend()

plt.show()


# In[28]:


#composite['registration_init_year'] = composite.registration_init_time.dt.year
#composite['registration_init_month'] = composite.registration_init_time.dt.month
#composite['registration_init_day'] = composite.registration_init_time.dt.day
#composite = composite.drop('registration_init_time', axis=1)


# In[29]:


#year = churn_rate(composite, 'registration_init_year')


# In[30]:


#year


# In[31]:


#composite.registration_init_year.value_counts()


# In[32]:


#month = churn_rate(composite, 'registration_init_month')


# In[33]:


#month


# In[34]:


#day = churn_rate(composite, 'registration_init_day')


# In[35]:


#composite.isnull().sum()


# In[36]:


composite.shape


# In[37]:


for col in transactions.columns:
    print(col + ' has ' + str(transactions[col].nunique()) + ' unique_values.')


# In[38]:


#transactions = transactions.drop_duplicates()
#transactions['payment_method_id', 'payment_plan_days']


# In[39]:


#def churn_rate(df, col):
#    col_rate = df.groupby(col)[['is_churn']].mean().reset_index().sort_values(
#        'is_churn', ascending=False)
#    sns.barplot(x=col, y='is_churn', data=col_rate)
#    print(plt.show())
#    return col_rate


# In[40]:





# In[40]:


#pm_id = churn_rate(transactions, 'payment_method_id')


# In[41]:


#pm_id.dtypes


# In[42]:


#members = pd.merge(members, train, on='msno', how='left')

transactions.msno.nunique()
# In[43]:


#members.shape


# In[44]:




