#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc


# In[ ]:


#engine='c' is used to faster read our .csv files
aisles = pd.read_csv('../input/aisles.csv' , engine='c')
departments = pd.read_csv('../input/departments.csv', engine='c')
products = pd.read_csv('../input/products.csv', engine='c')

#merge info of aisles & departments to products with a single request
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')

#fix names to a clear format [replace spaces with underscores]
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower()
goods.department = goods.department.str.replace(' ', '_').str.lower()
goods.aisle= goods.aisle.str.replace(' ', '_').str.lower()

goods.head()


# In[ ]:


#load orders
orders = pd.read_csv('../input/orders.csv', engine='c' )
orders.head()


# In[ ]:


#load prior
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c')
#load train                       
op_train = pd.read_csv('../input/order_products__train.csv', engine='c')
                       
#concatenate rows of train below prior:           
log= pd.concat([op_prior,op_train], ignore_index=1)                
log.tail()


# In[ ]:


# !--! runtime: 1m:14s
log.sort_values(['order_id', 'add_to_cart_order'], inplace=True)
log.reset_index(drop=1, inplace=True)
log = pd.merge(log, goods, on='product_id', how='left')
log = pd.merge(log, orders, on='order_id', how='left')
log['order_number_rev'] = log.groupby('user_id').order_number.transform(np.max) - log.order_number
gc.collect()
log.head()


# In[ ]:


log.head()


# In[ ]:


#we indicate our desired groups (in this case, we create info for each product)
#we will reuse this grouping also in the next features
gr = log.groupby('product_id')


# In[ ]:


#pro (for products) will be a hyper-DF to store all new features for products
#.to_frame converts the aggegated values for each product into a DF
pro = gr.product_id.count().to_frame()
pro.columns = ['total_purchases']

pro.head()


# In[ ]:


#mean position in the add_to_cart of order

#we calculate the mean value of add to cart order for each product
#we chain .to_frame() to create a DF with info for each product
pro['item_mean_pos_cart'] = gr.add_to_cart_order.mean()

#now we create other metrics for each product [we use all known aggregation functions]:

#sum of orders
pro['item_sum_pos_cart'] = gr.add_to_cart_order.sum()
#min value [the best place appeared on a cart order]
pro['item_min_pos_cart'] = gr.add_to_cart_order.min()
#max [the worst place appeared on a cart order]
pro['item_max_pos_cart'] = gr.add_to_cart_order.max()
#median
pro['item_median_pos_cart'] = gr.add_to_cart_order.median()
#standard deviation - how dispersed is the order 
pro['item_std_pos_cart'] = gr.add_to_cart_order.std()

pro.head(10)


# In[ ]:


pro.reset_index(level=0, inplace=True)
pro.head()


# In[ ]:


#dropna removes first order of a user
#runtime : 3m:16s
#average
dslo = log.dropna(axis=0).groupby('product_id').days_since_prior_order.mean().to_frame()
dslo.columns = ['days_since_last_order_product_mean']
#max
dslo['days_since_last_order_product_max'] = log.dropna(axis=0).groupby('product_id').days_since_prior_order.max().to_frame()
#min
dslo['days_since_last_order_product_min'] = log.dropna(axis=0).groupby('product_id').days_since_prior_order.min().to_frame()
#sum
dslo['days_since_last_order_product_sum'] = log.dropna(axis=0).groupby('product_id').days_since_prior_order.sum().to_frame()
#median
dslo['days_since_last_order_product_median'] = log.dropna(axis=0).groupby('product_id').days_since_prior_order.median().to_frame()
#standard deviation
dslo['days_since_last_order_product_std'] = log.dropna(axis=0).groupby('product_id').days_since_prior_order.std().to_frame()
dslo.reset_index(level=0, inplace=True)
dslo.head()


# In[ ]:


#we merge the features with hyper-DF "pro"
pro = pd.merge(pro, dslo, on='product_id', how='left')
pro.head()


# In[ ]:


#runtime : 35s
item_users = log.groupby(['product_id', 'user_id']).size().reset_index()
item_users.columns = ['product_id', 'user_id', 'total']
item_users[item_users.total==1].head()


# In[ ]:


# how many times an item bought and it was the first in the card list
item_one = item_users[item_users.total==1].groupby('product_id').size().reset_index()
item_one.columns = ['product_id', 'item_only_one_user_total']
item_one.head()


# In[ ]:


#define a ratio by dividing with the total number of purchases [already calculated in the pro hyper-DF]
item_one['ratio_firstorder_to_all'] = item_one['item_only_one_user_total']/ pro['total_purchases']
item_one.head()


# In[ ]:


#merge to the hyper-DF
pro = pd.merge(pro, item_one, how='left')
pro.head()


# In[ ]:


product_hour1 = log.groupby(['product_id', 'order_hour_of_day']).size().reset_index()
product_hour1.columns = ['product_id', 'order_hour_of_day', 'item_hour_cnt']
product_hour1.head(25)


# In[ ]:


product_hour1['item_hour_ratio'] = product_hour1.item_hour_cnt / product_hour1.groupby('product_id').transform(np.sum).item_hour_cnt
product_hour1.head()


# In[ ]:


### Total unique orders of a product for a given hour. (drop orders of the same hour from same users)


# In[ ]:


product_hour2 = log.drop_duplicates(['user_id', 'product_id', 'order_hour_of_day']).groupby(['product_id', 'order_hour_of_day']).size().reset_index()
product_hour2.columns = ['product_id', 'order_hour_of_day', 'item_hour_cnt_unq']
product_hour2['item_hour_ratio_unq'] = product_hour2.item_hour_cnt_unq / product_hour2.groupby('product_id').transform(np.sum).item_hour_cnt_unq
product_hour2.head()


# In[ ]:


product_hour= pd.merge(product_hour1, product_hour2)
product_hour.head()


# In[ ]:


product_day1 = log.groupby(['product_id', 'order_dow']).size().reset_index()
product_day1.columns = ['product_id', 'order_dow', 'item_dow_cnt']
product_day1['item_dow_ratio'] = product_day1.item_dow_cnt / product_day1.groupby('product_id').transform(np.sum).item_dow_cnt


# In[ ]:


product_day2 = log.drop_duplicates(['user_id', 'product_id', 'order_dow']).groupby(['product_id', 'order_dow']).size().reset_index()
product_day2.columns = ['product_id', 'order_dow', 'item_dow_cnt_unq']
product_day2['item_dow_ratio_unq'] = product_day2.item_dow_cnt_unq / product_day2.groupby('product_id').transform(np.sum).item_dow_cnt_unq    


# In[ ]:


product_day= pd.merge(product_day1, product_day2)
product_day.head()


# In[ ]:


gro= log.groupby('order_id')
order_size = gro.size().reset_index()
order_size.columns = ['order_id', 'total_products_of_order']
order_size.head()


# In[ ]:


order_product= log.groupby(['user_id', 'product_id']).size().reset_index()
order_product.columns= ['user_id', 'product_id', 'times']


# In[ ]:


order_product.head()


# In[ ]:


order_product_choice = order_product.groupby('product_id').times.max().to_frame()
order_product_choice.head()

