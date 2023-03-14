#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
df_sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
df_price = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')


# In[3]:


# Check for missing entries
df_cal['date'] = pd.to_datetime(df_cal['date'])

assert (df_cal['date'].max() - df_cal['date'].min()).days + 1 == df_cal.shape[0], 'Missing Dates in the data'


# In[4]:


df_cal['weekday'] = pd.Categorical(df_cal['weekday'], 
                                   categories=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday'], 
                                   ordered=True)


# In[5]:


df_event_types = pd.wide_to_long(df_cal[['event_name_1','event_type_1','event_name_2','event_type_2']].reset_index(), 
                stubnames=['event_name_','event_type_'],
                i='index',j='num')\
    .dropna()\
    .reset_index(level=-1)\
    .groupby(['event_type_','event_name_'], as_index=False)\
    .count()\
    .rename(columns={'event_type_':'Event Type','event_name_':'Event Name', 'num':'Counts'})


# In[6]:


fig, ax = plt.subplots(1,1,figsize=(10,12))
ax = sns.barplot(x='Counts', y='Event Name', hue='Event Type', data=df_event_types, orient='h', dodge=False)
ax.set_title('Total No. of Events (By type and name)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5))
plt.show()


# In[7]:


df_cal[['year','weekday','snap_CA','snap_TX','snap_WI']]    .groupby(['year','weekday'])    .sum()    .plot(kind='bar', 
          stacked=True, 
          figsize=(20,6), 
          title='SNAP Purchases (Year and Days)')


# In[8]:


snap_by_month = df_cal[['date','snap_CA','snap_TX','snap_WI']]    .resample(rule='M',on='date')    .sum()

snap_by_month.index = snap_by_month.index.strftime('%b-%Y')

fig, ax = plt.subplots(1,1,figsize=(20,6))
snap_by_month.plot(kind='bar', stacked=True, title='SNAP Purchases Across Months', ax=ax)

ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5))

plt.show()


# In[9]:


df_cat_dept = df_sales[['dept_id','cat_id','id']].groupby(['cat_id','dept_id']).count().reset_index()

df_cat_dept.rename(columns={'id':'Count of Unique IDs'}, inplace=True)

fig, ax = plt.subplots(1,1,figsize=(12,8))

sns.barplot(data=df_cat_dept, x='Count of Unique IDs', y='dept_id', hue='cat_id', orient='h', dodge=False, ax=ax)
ax.set_title('Total No. of Unique IDs (By Department and Category)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), title='cat_id')
plt.show()


# In[10]:


df_sales['Total_sales'] = df_sales.iloc[:, 6:].sum(axis=1)

df_total_item_sales = df_sales[['cat_id','dept_id','Total_sales']].groupby(['cat_id','dept_id'], as_index=False).sum()

fig, ax = plt.subplots(1,1,figsize=(12,8))

sns.barplot(data=df_total_item_sales, x='Total_sales', y='dept_id', hue='cat_id', orient='h', dodge=False, ax=ax)
ax.set_title('Total Items Sales (By Department and Category)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), title='cat_id')
plt.show()


# In[11]:


fig, ax = plt.subplots(1,1, figsize=(12,6))

df_item_by_store = df_sales[['state_id','store_id','item_id']]    .groupby(['state_id','store_id']).count()    .reset_index()    .rename(columns={'item_id':'Unique Items'})
    
sns.barplot(data=df_item_by_store, x='store_id', y='Unique Items', hue='state_id', dodge=False, ax=ax)
 
ax.set_title('Count of Unique Items across Stores')    
ax.set_ylabel('Count of Unique Items')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), title='state_id')
plt.show()


# In[12]:


df_total_store_sales = df_sales[['state_id','store_id','Total_sales']].groupby(['state_id','store_id'], as_index=False).sum()

fig, ax = plt.subplots(1,1,figsize=(12,6))

sns.barplot(data=df_total_store_sales, x='store_id', y='Total_sales', hue='state_id', dodge=False, ax=ax)
ax.set_title('Total Sales Volume (By Store)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), title='state_id')
plt.show()


# In[13]:


d_cols = list(df_sales.columns[df_sales.columns.str.startswith('d_')])

df_days_long = df_sales[['item_id','store_id'] + d_cols].set_index(['item_id','store_id'])    .stack()    .rename('sale_unit')    .reset_index(-1)    .query('sale_unit > 0')

d_to_wmyrwk_mapping = df_cal[['wm_yr_wk','d']].set_index('d').to_dict()['wm_yr_wk']

df_days_long['wm_yr_wk'] = df_days_long['level_2'].map(d_to_wmyrwk_mapping)
df_days_long = df_days_long.set_index('wm_yr_wk', append=True).groupby(level=[0,1,2])['sale_unit'].sum().to_frame()

df_merged = df_days_long.merge(df_price.set_index(['item_id','store_id','wm_yr_wk']), left_index=True, right_index=True, how='left')
df_merged['sale_value'] = df_merged['sale_unit'] * df_merged['sell_price']


# In[14]:


df_sales_value_by_store = (df_merged['sale_value'].groupby(level=[1]).sum() / 1_000_000)    .round(1).rename('Sales Value ($ millions)')    .to_frame()    .reset_index()

df_sales_value_by_store['state_id'] = df_sales_value_by_store['store_id'].str[:2]

fig, ax = plt.subplots(1,1,figsize=(12,6))

sns.barplot(data=df_sales_value_by_store, x='store_id', y='Sales Value ($ millions)', hue='state_id', dodge=False, ax=ax)
ax.set_title('Total Sales Value (By Store)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), title='state_id')
plt.show()

