#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # dataframes
import numpy as np # algebra & calculus
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style


# In[2]:


#prior dataset
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})

print('Total ordered products(prior): {}'.format(op_prior.shape[0]))
op_prior.head()


# In[3]:


# orders
orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))
print(orders.info())
orders.head()


# In[4]:


# test dataset (submission)
test_orders = orders[orders.eval_set == 'test']
test_orders.head()


# In[5]:


# combine order details

order_details = pd.merge(op_prior, orders, on = 'order_id', how = 'left')
print(order_details.head())
print(order_details.dtypes)


# In[6]:


test_history = order_details[(order_details.user_id.isin(test_orders.user_id))].groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()
test_history.columns = ['user_id', 'products']

test_history = pd.merge(left=test_history, 
                        right=test_orders, 
                        how='right', 
                        on='user_id')[['order_id', 'products']]

test_history.fillna('None')

test_history.to_csv('dumb_submission.csv', encoding='utf-8', index=False)


# In[7]:


get_ipython().run_cell_magic('time', '', "\ntest_history = order_details[(order_details.user_id.isin(test_orders.user_id)) \n                             & (order_details.reordered == 1)]\\\n.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()\ntest_history.columns = ['user_id', 'products']\n\ntest_history = pd.merge(left=test_history, \n                        right=test_orders, \n                        how='right', \n                        on='user_id')[['order_id', 'products']]\n\ntest_history.to_csv('dumb2_subm.csv', encoding='utf-8', index=False)")


# In[8]:


get_ipython().run_cell_magic('time', '', "test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\n# This is assuming that order number is ordered. The max number of the order_number is the last order.\nlast_orders = test_history.groupby('user_id')['order_number'].max().reset_index()\n\nlast_ordered_reordered_only = pd.merge(\n            left=pd.merge(\n                    left=last_orders,\n                    right=test_history[test_history.reordered == 1],\n                    how='left',\n                    on=['user_id', 'order_number']\n                )[['user_id', 'product_id']],\n            right=test_orders[['user_id', 'order_id']],\n            how='left',\n            on='user_id'\n        )\n\nt = last_ordered_reordered_only.fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) \n                                              ).reset_index().replace(to_replace='-1', \n                                                                      value='None')\nt.columns = ['order_id', 'products']\n\n# save submission\nt.to_csv('less_dumb_subm_last_order_reordered_only.csv', \n                         encoding='utf-8', \n                         index=False)")

