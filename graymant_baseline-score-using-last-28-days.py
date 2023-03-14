#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd

import os


# In[2]:


input_path = "../input/m5-forecasting-accuracy"

def get_salesval_coltypes():
    keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] +         [f"d_{i}" for i in range(1, 1914)]
    values = ['object', 'category', 'category', 'category', 'category', 'category'] +        ["uint16" for i in range(1, 1914)]
    return dict(zip(keys, values))

submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
sales_train_val = pd.read_csv(os.path.join(input_path, 'sales_train_validation.csv'), 
                              dtype=get_salesval_coltypes())


# In[3]:


preds = sales_train_val.iloc[:, -28:]


# In[4]:


all_preds = pd.concat([preds, preds])


# In[5]:


all_preds.reset_index(inplace=True, drop=True)
all_preds['id'] = submission.id
all_preds = all_preds.reindex(
        columns=['id'] + [c for c in all_preds.columns if c != 'id'], copy=False)
all_preds.columns = ['id'] + [f"F{i}" for i in range(1, 29)]

all_preds.to_csv('submission.csv', index=False)


# In[ ]:




