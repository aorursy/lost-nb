#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


import pandas as pd

train = pd.read_csv('../input/instacart-market-basket-analysis/order_products__train.csv')
prior = pd.read_csv('../input/instacart-market-basket-analysis/order_products__prior.csv')
orders = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')
products = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')
test = pd.read_csv('../input/instacart-market-basket-analysis/sample_submission.csv')
departments = pd.read_csv('../input/instacart-market-basket-analysis/departments.csv')
aisles = pd.read_csv('../input/instacart-market-basket-analysis/aisles.csv')


# In[3]:


order_product_count = prior.groupby('order_id').count()[['product_id']]


# In[ ]:





# In[4]:


order_product_count


# In[ ]:





# In[5]:


order_product_count.columns = ['product_count']


# In[6]:


order_product_count


# In[7]:


orders2 = orders.merge(order_product_count, left_on='order_id', right_index=True)


# In[8]:


index_day = "Sun Mon Tue Wen Thu Fri Sat".split()


# In[9]:


def drawWeekHour(ds, values,  aggfunc=len, title=None, figsize=(18,5) , cmap=None):
    weekhour_ds = ds.pivot_table(index='order_dow', columns='order_hour_of_day', values=values, aggfunc=aggfunc).fillna(0)
    weekhour_ds.index =  [index_day[index] for index in weekhour_ds.index]
    
    plt.figure(figsize=figsize)
    f = sns.heatmap(weekhour_ds, annot=True, fmt="1.1f", linewidths=.5, cmap=cmap) 
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    if title:
        plt.title(title, fontsize=15)


# In[10]:



drawWeekHour(orders, values='days_since_prior_order',aggfunc=lambda x: np.mean(x), title="prior orders", cmap='YlGn')


# In[11]:


orders.head()


# In[12]:


sns.set(style="whitegrid", palette="colorblind", font_scale=1.5)

orders2.groupby('order_number').agg({'days_since_prior_order':np.mean, 'product_count':np.mean})    .plot(figsize=(16,6), title="order_number, prior_order", marker='o' )
plt.ylabel('days since prior order ')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




