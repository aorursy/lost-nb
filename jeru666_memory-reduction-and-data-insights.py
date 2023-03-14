#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.csv')
df_members = pd.read_csv('../input/members.csv')
df_transactions = pd.read_csv('../input/transactions.csv')
df_sample = pd.read_csv('../input/sample_submission_zero.csv')

#df_user_logs_1 = pd.read_csv('../input/user_logs.csv', chunksize = 500)
#df = pd.concat(df_user_logs_1, ignore_index=True)

# Any results you write to the current directory are saved as output.


# In[2]:


#--- Displays memory consumed by each column ---
print(df_members.memory_usage())

#--- Displays memory consumed by entire dataframe ---
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[3]:


#--- Check whether it has any missing values ----
print(df_members.isnull().values.any())


# In[4]:


#--- check which columns have Nan values ---
columns_with_Nan = df_members.columns[df_members.isnull().any()].tolist()
print(columns_with_Nan)


# In[5]:


#--- Check the datatypes of each of the columns in the dataframe ---
print(df_members.dtypes)


# In[6]:


print (df_members.head())


# In[7]:


print(np.max(df_members['city']))
print(np.min(df_members['city']))


# In[8]:


df_members['city'] = df_members['city'].astype(np.int8)


# In[9]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[10]:


print(np.max(df_members['bd']))
print(np.min(df_members['bd']))


# In[11]:


df_members['bd'] = df_members['bd'].astype(np.int16)


# In[12]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[13]:


print(np.max(df_members['registered_via']))
print(np.min(df_members['registered_via']))


# In[14]:


df_members['registered_via'] = df_members['registered_via'].astype(np.int8)


# In[15]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[16]:


df_members['registration_init_year'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[:4]))
df_members['registration_init_month'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
df_members['registration_init_date'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[-2:]))


# In[17]:


df_members['expiration_date_year'] = df_members['expiration_date'].apply(lambda x: int(str(x)[:4]))
df_members['expiration_date_month'] = df_members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
df_members['expiration_date_date'] = df_members['expiration_date'].apply(lambda x: int(str(x)[-2:]))


# In[18]:


print(df_members.head())


# In[19]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[20]:


df_members['registration_init_year'] = df_members['registration_init_year'].astype(np.int16)
df_members['registration_init_month'] = df_members['registration_init_month'].astype(np.int8)
df_members['registration_init_date'] = df_members['registration_init_date'].astype(np.int8)

df_members['expiration_date_year'] = df_members['expiration_date_year'].astype(np.int16)
df_members['expiration_date_month'] = df_members['expiration_date_month'].astype(np.int8)
df_members['expiration_date_date'] = df_members['expiration_date_date'].astype(np.int8)


# In[21]:


#--- Now drop the unwanted date columns ---
df_members = df_members.drop('registration_init_time', 1)
df_members = df_members.drop('expiration_date', 1)


# In[22]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[23]:


print(df_train.head())


# In[24]:


print(df_train.isnull().values.any())


# In[25]:


print(df_train.dtypes)


# In[26]:


mem = df_train.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[27]:


df_train['is_churn'] = df_train['is_churn'].astype(np.int8)

mem = df_train.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[28]:


print(df_transactions.head())


# In[29]:


mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[30]:


print(df_transactions.isnull().values.any())


# In[31]:


print(df_transactions.dtypes)


# In[32]:



df_transactions['payment_method_id'] = df_transactions['payment_method_id'].astype(np.int8)
df_transactions['payment_plan_days'] = df_transactions['payment_plan_days'].astype(np.int16)
df_transactions['plan_list_price'] = df_transactions['plan_list_price'].astype(np.int16)
df_transactions['actual_amount_paid'] = df_transactions['actual_amount_paid'].astype(np.int16)
df_transactions['is_auto_renew'] = df_transactions['is_auto_renew'].astype(np.int8)
df_transactions['is_cancel'] = df_transactions['is_cancel'].astype(np.int8)


# In[33]:



df_transactions['transaction_date_year'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[:4]))
df_transactions['transaction_date_month'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[4:6]))
df_transactions['transaction_date_date'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[-2:]))

df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))
df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))


# In[34]:



df_transactions['transaction_date_year'] = df_transactions['transaction_date_year'].astype(np.int16)
df_transactions['transaction_date_month'] = df_transactions['transaction_date_month'].astype(np.int8)
df_transactions['transaction_date_date'] = df_transactions['transaction_date_date'].astype(np.int8)

df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date_year'].astype(np.int16)
df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date_month'].astype(np.int8)
df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date_date'].astype(np.int8)


# In[35]:


#--- Now drop the unwanted date columns ---
df_transactions = df_transactions.drop('transaction_date', 1)
df_transactions = df_transactions.drop('membership_expire_date', 1)


# In[36]:


print(df_transactions.head())


# In[37]:


mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[38]:


print('DONE!!')


# In[39]:


#print(df_members.head())
#print(df_train.head())

df_train_members = pd.merge(df_train, df_members, on='msno', how='inner')
df_merged = pd.merge(df_train_members, df_transactions, on='msno', how='inner')
print(df_merged.head())


# In[40]:


df_train_members.hist(column='is_churn')


# In[41]:


#--- Check whether new dataframe has any missing values ----
print(df_train_members.isnull().values.any())


# In[42]:


#--- check which columns have Nan values ---
columns_with_Nan = df_train_members.columns[df_train_members.isnull().any()].tolist()
print(columns_with_Nan)


# In[43]:


df_train_members['gender'].isnull().sum()


# In[44]:


churn_vs_gender = pd.crosstab(df_train_members['gender'], df_train_members['is_churn'])

churn_vs_gender_rate = churn_vs_gender.div(churn_vs_gender.sum(1).astype(float), axis=0) # normalize the value
churn_vs_gender_rate.plot(kind='barh', , stacked=True)


# In[45]:


churn_registered_via = pd.crosstab(df_train_members['registered_via'], df_train_members['is_churn'])

churn_vs_registered_via_rate = churn_registered_via.div(churn_registered_via.sum(1).astype(float), axis=0) # normalize the value
churn_vs_registered_via_rate.plot(kind='barh', stacked=True)


# In[46]:


churn_vs_city = pd.crosstab(df_train_members['city'], df_train_members['is_churn'])

churn_vs_city_rate = churn_vs_city.div(churn_vs_city.sum(1).astype(float),  axis=0) # normalize the value
churn_vs_city_rate.plot(kind='bar', stacked=True)


# In[47]:


#eliminating extreme outliers
df_train_members = df_train_members[df_train_members['bd'] >= 1]
df_train_members = df_train_members[df_train_members['bd'] <= 80]

import seaborn as sns
sns.violinplot(x=df_train_members["is_churn"], y=df_train_members["bd"], data=df_train_members)


# In[48]:


print (df_train_members['city'].unique())


# In[49]:


data = df_train_members.groupby('city').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='city', y='msno', data=data)


# In[50]:


print (df_train_members['bd'].nunique())


# In[51]:


df_train_members.plot(x=df_train_members.index, y='bd')


# In[52]:


print (df_train_members['registered_via'].unique())


# In[53]:


data = df_train_members.groupby('registered_via').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registered_via', y='msno', data=data)


# In[54]:


print (df_train_members['registration_init_year'].unique())


# In[55]:


data = df_train_members.groupby('registration_init_year').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_year', y='msno', data=data)


# In[56]:


data = df_train_members.groupby('registration_init_month').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_month', y='msno', data=data)


# In[57]:


data = df_train_members.groupby('registration_init_date').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_date', y='msno', data=data)


# In[58]:


data = df_merged.groupby('payment_method_id').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='payment_method_id', y='msno', data=data)


# In[59]:


from matplotlib import pyplot
data = df_merged.groupby('payment_plan_days').aggregate({'msno':'count'}).reset_index()
a4_dims = (11, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='payment_plan_days', y='msno', data=data)


# In[60]:


data = df_merged.groupby('plan_list_price').aggregate({'msno':'count'}).reset_index()
a4_dims = (20, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='plan_list_price', y='msno', data=data)


# In[61]:


data = df_merged.groupby('actual_amount_paid').aggregate({'msno':'count'}).reset_index()
a4_dims = (20, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='actual_amount_paid', y='msno', data=data)


# In[62]:


data = df_merged.groupby('is_cancel').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='is_cancel', y='msno', data=data)


# In[63]:


#print(df_merged.columns)


# In[64]:


corr_matrix = df_merged.corr()
f, ax = plt.subplots(figsize=(20, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#--- For positive high correlation ---
high_corr_var = np.where(corr_matrix > 0.8)
high_corr_var = [(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr = []
for i in range(0,len(high_corr_var)):
    high_corr.append(high_corr_var[i][0])
    high_corr.append(high_corr_var[i][1])
high_corr = list(set(high_corr))

#--- For negative high corrlation ---
high_neg_corr_var = np.where(corr_matrix < -0.8)
high_neg_corr_var = [(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_neg_corr_var) if x!=y and x<y]
high_neg_corr = []
for i in range(0,len(high_neg_corr_var)):
    high_corr.append(high_neg_corr_var[i][0])
    high_corr.append(high_neg_corr_var[i][1])
high_neg_corr = list(set(high_neg_corr))  

#--- Merge both these lists avoiding duplicates ---
high_corr_list = list(set(high_corr + high_neg_corr))


# In[65]:


print(high_corr_list)

