#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


from subprocess import check_output
print(check_output(['ls','../input']).decode('utf8'))
#print(check_output)


# In[67]:


df_aisles = pd.read_csv('../input/aisles.csv',low_memory=False)
df_departments = pd.read_csv('../input/departments.csv', low_memory=False)
df_order_product_prior = pd.read_csv('../input/order_products__prior.csv',low_memory=False)
df_order_product_train = pd.read_csv('../input/order_products__train.csv',low_memory=False)
df_orders = pd.read_csv('../input/orders.csv',low_memory=False)
df_products = pd.read_csv('../input/products.csv',low_memory=False)


# In[68]:


df_orders.head()


# In[69]:


crs = df_orders.groupby('user_id')[['order_number','order_dow']].agg({'order_number':'max','order_dow':'sum'})
crs.head()


# In[70]:


df_order_product_prior.head()


# In[71]:


df_order_product_train.head()


# In[72]:


order_src = df_orders['eval_set'].value_counts()
order_src


# In[73]:


order_src = df_orders['eval_set'].value_counts()
#order_src.head()
sns.barplot(order_src.index,order_src.values)
plt.title('EDA on order dataset to count rows in each orders dataset',fontsize=14)
plt.xlabel('eval set',fontsize=12)
plt.ylabel('Number of Occurrences',fontsize=12)
plt.show()


# In[74]:


plt.figure(figsize=(10,5))
sns.countplot(x='order_dow',data=df_orders)
plt.title('1. Order by week day',fontsize=14)
plt.xlabel('week day',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.show()


# In[75]:


plt.figure(figsize=(10,5))
sns.countplot(x='order_hour_of_day',data=df_orders)
plt.title('2. Order by time of the day',fontsize=14)
plt.xlabel('Time of the day',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.show()


# In[76]:


grouped_df_groupby = df_orders.groupby(["order_dow", "order_hour_of_day"])
grouped_df = grouped_df_groupby["order_number"].aggregate("count").reset_index()
print(grouped_df)
print('-----------------------------AFTER PIVOT-------------------------------------')
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
print(grouped_df)
plt.figure(figsize=(10,5))
sns.heatmap(grouped_df)
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


# In[77]:


plt.figure(figsize=(17,5))
sns.countplot(x='days_since_prior_order',data=df_orders)
plt.title('DAY SINCE THE PRIOR ORDER',fontsize=14)
plt.xlabel('No of days since the prior order',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.show()


# In[78]:


df_order_product_prior.head()


# In[79]:


df_order_product_train.head()


# In[80]:


df_order_product_prior['reordered'].sum()


# In[81]:


df_order_product_prior.shape


# In[82]:


df_order_product_prior['reordered'].sum() /df_order_product_prior.shape[0] * 100


# In[83]:


df_order_product_train['reordered'].sum()/df_order_product_train.shape[0] * 100


# In[84]:


df_order_product_prior.head()


# In[91]:


grouped_df = df_order_product_train.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()
cnt_srs = grouped_df.add_to_cart_order.value_counts()

plt.figure(figsize=(20,8))
sns.barplot(cnt_srs.index,cnt_srs.values,alpha=0.5)
plt.title('Number of products bought in each order',fontsize=14)
plt.xlabel('No of products',fontsize=12)
plt.ylabel('No of occurance',fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




