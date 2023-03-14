#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To begin the competition, I tried to submit the average per item and per store.
# I was motivated to do so since this is the top basic benchmark which was posted...
# I used some tricks (e.g. if a item/store wasn't in the train - 
#         take the average of the store/item which is already known)

# I get s score of 1.323.
# Anything simpler got a worse score...

# How come I can't get close to the posted benchmark (0.726)?
# Isn't this weird?


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime


# In[3]:


# reading the data and turning negative numbers of unit sales to 0
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df['unit_sales'] = train_df['unit_sales'].apply(lambda x: max(0,x))


# In[4]:


# creating dictionaries of 3 types:
# 1. mean by store and item
# 2. mean by item only
# 3. mean by store only

# also calculating average number of sales per item (into mean_all)

mean_per_both = train_df[['item_nbr', 'store_nbr', 'unit_sales']]    .groupby(['item_nbr', 'store_nbr'], as_index = False)    .agg('mean')
mean_dict_both = dict(zip(zip(mean_per_both['store_nbr'],mean_per_both['item_nbr']),                           mean_per_both['unit_sales']))

mean_per_item = train_df[['item_nbr', 'unit_sales']].groupby(['item_nbr'], as_index = False)    .agg('mean')
mean_dict_item = dict(zip(mean_per_item['item_nbr'], mean_per_item['unit_sales']))

mean_per_store = train_df[['store_nbr', 'unit_sales']].groupby(['store_nbr'], as_index = False)    .agg('mean')
mean_dict_store = dict(zip(mean_per_store['store_nbr'], mean_per_item['unit_sales']))

mean_all = np.mean(list(mean_dict_item.values()))


# In[5]:


# function which receives an instance and returns the most "specific" mean 
# which can be calculated... (using dictionaries from previous cell)

def calc_mean(x):
    try:
        return mean_dict_both[(x['store_nbr'],x['item_nbr'])]
    except:
        try:
            return mean_dict_item[x['item_nbr']]
        except:
            try:
                mean_dict_item[x['store_nbr']]
            except:
                return mean_all


# In[6]:


# creating "predictions" using the function from the previous cell
test_df['unit_sales'] = test_df[['store_nbr','item_nbr']].apply(calc_mean, axis=1)


# In[7]:


# saving to file
today = '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().month) +     '_' + str(datetime.datetime.now().year)
test_df[['id', 'unit_sales']].to_csv('mean_by_item_and_store' + today + '.csv', index = False)

