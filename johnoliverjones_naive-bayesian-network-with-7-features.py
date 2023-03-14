#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#mdf = 'c:/Users/John/Documents/Kaggle/Porto Seguro/'
mdf = '../input/'

train = pd.read_csv(mdf + "train.csv", usecols = ['target', 'ps_car_07_cat',
    'ps_car_02_cat', 'ps_car_13','ps_reg_02', 'ps_ind_06_bin', 'ps_ind_16_bin', 'ps_ind_17_bin'])

test = pd.read_csv(mdf + "test.csv", usecols = ['id', 'ps_car_07_cat', 'ps_car_02_cat',
    'ps_car_13','ps_reg_02', 'ps_ind_06_bin', 'ps_ind_16_bin', 'ps_ind_17_bin'])
bins = [0.0, 0.639, 0.784, 1.093, 4.4]
train['ps_car_13_d'] = pd.cut(train['ps_car_13'], bins)
test['ps_car_13_d'] = pd.cut(test['ps_car_13'], bins)
bins2 = [-0.1, 0.25, 0.75, 2.0]
train['ps_reg_02_d'] = pd.cut(train['ps_reg_02'], bins2)
test['ps_reg_02_d'] = pd.cut(test['ps_reg_02'], bins2)
train.head(4)


# In[2]:


# Now calculate the each factor associated with each feature, (fi): p( feature_i | target)

f1 = pd.DataFrame()
f2 = pd.DataFrame()
f3 = pd.DataFrame()
f4 = pd.DataFrame()
f5 = pd.DataFrame()
f6 = pd.DataFrame()
f7 = pd.DataFrame()

f1 = train.groupby('ps_car_13_d')['target'].agg([('p_f1','mean')]).reset_index()
f2 = train.groupby('ps_reg_02_d')['target'].agg([('p_f2','mean')]).reset_index()
f3 = train.groupby(['ps_car_07_cat'])['target'].agg([('p_f3','mean')]).reset_index()
f4 = train.groupby(['ps_car_02_cat'])['target'].agg([('p_f4','mean')]).reset_index()
f5 = train.groupby('ps_ind_06_bin')['target'].agg([('p_f5','mean')]).reset_index()
f6 = train.groupby('ps_ind_16_bin')['target'].agg([('p_f6','mean')]).reset_index()
f7 = train.groupby('ps_ind_17_bin')['target'].agg([('p_f7','mean')]).reset_index()
f3.head(10)


# In[3]:


sol1 = pd.DataFrame()
sol1 = test.merge(f1, on = 'ps_car_13_d')
sol2 = pd.DataFrame()
sol2 = sol1.merge(f2, on = 'ps_reg_02_d')
del sol1
sol3 = pd.DataFrame()
sol3 = sol2.merge(f3, on = 'ps_car_07_cat')
del sol2
sol4 = pd.DataFrame()
sol4 = sol3.merge(f4, on = 'ps_car_02_cat')
del sol3
sol5 = pd.DataFrame()
sol5 = sol4.merge(f5, on = 'ps_ind_06_bin')
del sol4
sol6 = pd.DataFrame()
sol6 = sol5.merge(f6, on = 'ps_ind_16_bin')
del sol5
sol = pd.DataFrame()
sol = sol6.merge(f7, on = 'ps_ind_17_bin')
del sol6
sol.head(5)


# In[4]:


# f is the product of factors of feaures
sol.loc[:,'f'] = sol.loc[:,'p_f1'] * sol.loc[:,'p_f2'] * sol.loc[:,'p_f3'] * sol.loc[:,'p_f4']                 * sol.loc[:,'p_f5'] * sol.loc[:,'p_f6'] * sol.loc[:,'p_f7'] 

z = sol.f.sum() / len(sol.f)
# z is the normalizing factor
sol['target'] = 0.03645 * sol.loc[:,'f'] / z
sol[['id', 'target']].to_csv('bn_5_output_7_nodes.csv', index = False, float_format='%.4f')
sol.shape


# In[5]:


# thanks to cpmpml for : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
from numba import jit

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

sol1 = pd.DataFrame()
sol1 = train.merge(f1, on = 'ps_car_13_d')
sol2 = pd.DataFrame()
sol2 = sol1.merge(f2, on = 'ps_reg_02_d')
del sol1
sol3 = pd.DataFrame()
sol3 = sol2.merge(f3, on = 'ps_car_07_cat')
del sol2
sol4 = pd.DataFrame()
sol4 = sol3.merge(f4, on = 'ps_car_02_cat')
del sol3
sol5 = pd.DataFrame()
sol5 = sol4.merge(f5, on = 'ps_ind_06_bin')
del sol4
sol6 = pd.DataFrame()
sol6 = sol5.merge(f6, on = 'ps_ind_16_bin')
del sol5
sol = pd.DataFrame()
sol = sol6.merge(f7, on = 'ps_ind_17_bin')
del sol6
sol.loc[:,'f'] = sol.loc[:,'p_f1'] * sol.loc[:,'p_f2'] * sol.loc[:,'p_f3'] * sol.loc[:,'p_f4']                 * sol.loc[:,'p_f5'] * sol.loc[:,'p_f6'] * sol.loc[:,'p_f7'] 
z = sol.f.sum() / len(sol.f)
sol['exp_target'] = 0.03645 * sol.loc[:,'f'] / z

# Calculate GINI score
eval_gini(sol['target'], sol['exp_target'])

