#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as datetime

#mdf = 'c:/Users/John/Documents/Kaggle/zillow/data/'
mdf = '../input/'

train = pd.read_csv(mdf + 'properties_2016.csv')
train_label = pd.read_csv(mdf + 'train_2016_v2.csv', parse_dates = ['transactiondate'])

train.shape


# In[2]:


# Build features for model. Based on early BN of clusters
train.loc[:,'No_Structure'] = train.structuretaxvaluedollarcnt.isnull()
train.loc[:,'M_Age'] = 2017 - train['yearbuilt']
common = train.merge(train_label,on=['parcelid'])

model = pd.DataFrame()
model.loc[:,'logerror'] = common.logerror
model.loc[:,'Tax_Value'] = common.taxvaluedollarcnt
model.loc[:,'Structure_Age'] = common.M_Age

# the other variables are in buckets that maximize mutual information w logerror
bins1 = [0,184421,850293,1360000,500000000]
cat1 = ['<185k','<850k','<1360k','>=1360k']
model.loc[:,'Tax_Value_b'] = pd.cut(model.Tax_Value, bins1, labels = cat1)
bins2 = [0, 23, 41, 57, 80, 300]
cat2 = ['<23', '<41', '<57', '<80', '>=80']
model.loc[:,'Age_b'] = pd.cut(model.Structure_Age, bins2, labels = cat2)


# calculate the probability distribution p(e |Age,Tax)
# 
# the expected value of e is the weighted average for each state of logerror
p = pd.DataFrame()
p = model.groupby(['Tax_Value_b','Age_b'])['logerror'].agg([('exp_logerror','mean')])
pd.set_option('display.float_format', lambda x: '%.3f' % x)
p.reset_index(inplace = True)

# calculate p (e|Tax) when Age is nan.
ps = pd.DataFrame()
ps = model[model.Age_b.isnull() == True].groupby(['Tax_Value_b'])['logerror'].agg([('exp_logerror','mean')])
pd.set_option('display.float_format', lambda x: '%.3f' % x)
ps.reset_index(inplace = True)

# calculate p (e|Age) when there is no Tax_Value
pss = pd.DataFrame()
pss = model[model.Tax_Value.isnull() == True].        groupby(['Age_b'])['logerror'].agg([('exp_logerror','mean')])

p.head(10)


# In[3]:


#Use the above tables to look up the expected logerror for each property in train
df = pd.DataFrame()
df.loc[:,'parcelid'] = train.parcelid
df.loc[:,'Tax_Value'] = train.taxvaluedollarcnt
df.loc[:,'Age'] = 2017 - train.yearbuilt
df.loc[:,'Tax_Value_b'] = pd.cut(df.Tax_Value, bins1, labels = cat1)
df.loc[:,'Age_b'] = pd.cut(df.Age, bins2, labels = cat2)
df = df.drop('Tax_Value', 1)
df = df.drop('Age', 1)

df.head(5)


# In[4]:


s0 = pd.DataFrame()
s1 = pd.DataFrame()
df0 = pd.DataFrame()
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

df0 = df[(df['Tax_Value_b'].isnull() == False) & (df['Age_b'].isnull() == False)]
s0 = pd.merge(df0, p, on = ['Tax_Value_b', 'Age_b'], how = 'left')

df1 = df[(df['Tax_Value_b'].isnull() == False) & (df['Age_b'].isnull() == True)]
s1 = pd.merge(df1, ps, on = 'Tax_Value_b', how = 'left')

df2 = df[(df['Tax_Value_b'].isnull() == True) & (df['Age_b'].isnull() == True)]
df2['exp_logerror'] = 0.004

df3 = df[(df['Tax_Value_b'].isnull() == True) & (df['Age_b'].isnull() == False)]
df3['exp_logerror'] = -0.010

frames = [s0, s1, df2, df3]
df = pd.concat(frames)

df.shape


# In[5]:



df = df.drop('Tax_Value_b', 1)
df = df.drop('Age_b', 1)

df['201610'] = df.exp_logerror
df['201611'] = df.exp_logerror
df['201612'] = df.exp_logerror
df['201710'] = df.exp_logerror
df['201711'] = df.exp_logerror
df['201712'] = df.exp_logerror
df = df.drop('exp_logerror', 1)
df = df.rename(columns={'parcelid': 'ParcelId'})

df.sort_values('ParcelId')

df.to_csv('mean_logerror6.csv', index = False)
df.shape


# In[6]:




