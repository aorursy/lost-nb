#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import missingno as msno 
import seaborn as sns 
import plotly.graph_objects as go
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")

# Display all columns
pd.set_option('display.max_columns', None)

# List files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/m5-forecasting-accuracy'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Load data
files = ['/kaggle/input/m5-forecasting-accuracy/calendar.csv', 
         '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv',
         '/kaggle/input/m5-forecasting-accuracy/sell_prices.csv']

data = [pd.read_csv(f) for f in files]
dt_calendar, dt_sales, dt_prices = data

# Merge calendar and prices
dt_prices_s = shuffle(dt_prices, n_samples = 1000000)
dt_complementary = dt_prices_s.merge(dt_calendar, how='left', on='wm_yr_wk')

# Shuffle data (it is originally ordered) and take n rows (if you don't have enough RAM)
dt_sales_s = shuffle(dt_sales, n_samples = 1000)


# In[3]:


# Melt sales data
indicators = [f'd_{i}' for i in range(1,1914)]

dt_sales_melt = pd.melt(dt_sales_s, 
                        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                        value_vars = indicators, var_name = 'day_key', value_name = 'sales_day')

dt_sales_melt['day'] = dt_sales_melt['day_key'].apply(lambda x: x[2:]).astype(int)


# In[4]:


# Data to work with
columns = ['store_id','item_id','sell_price','date','year','d','event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
dt_work = dt_sales_melt.merge(dt_complementary[columns], how = 'left', left_on=['item_id','store_id','day_key'], right_on=['item_id','store_id','d'])
print(dt_work.shape)


# In[5]:


#General glimpse
msno.matrix(dt_work)


# In[6]:


# NA's as bar plot
msno.bar(dt_work)


# In[7]:


# NA's by department
msno.matrix(dt_work.sort_values(by=['dept_id']))


# In[8]:


# Dendogram of NA's values
msno.dendrogram(dt_work)


# In[9]:


# Distribution of d variable
sns.distplot(dt_work[dt_work.d.isna()].sales_day, color='b')


# In[10]:


print(dt_work.shape)
dt_work = dt_work.dropna(how='any', subset=['d'])
print(dt_work.shape)


# In[11]:


def timeseries_global(data:'pd.DataFrame',level:str):
    
    # Group data by level 
    dt_level = data.groupby([level, 'day']).agg({'sales_day':'sum','sell_price':'mean'})                                                          .reset_index()                                                          .sort_values(by=[level,'day'])
    
    # Visualize by components of level 
    levels = dt_level[level].unique()

    for l in levels: 
        df = dt_level[(dt_level[level] == l)]
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(16,8))
        plt.style.use('ggplot')
        ax1.plot(df['sales_day'], color = 'blue')
        ax1.set_xticklabels(labels = df.day.values, rotation=70)
        ax1.grid(False)
        ax1.set_title(f"Time series of sales for {l}")
        ax2.plot(df['sell_price'], color = 'red')
        ax2.set_xticklabels(labels = df.day.values, rotation=70)
        ax2.grid(False)
        ax2.set_title(f"Time series of price for {l}")
        break


# In[12]:


# State level
timeseries_global(dt_work,'state_id')


# In[13]:


# Store level
timeseries_global(dt_work,'store_id')


# In[14]:


def timeseries_particular(data:'pd.DataFrame',level_1:str, level_2:str):
    
    # Group data by level 
    dt_level = data.groupby([level_1,level_2, 'day']).agg({'sales_day':'sum','sell_price':'mean'})                                                                .reset_index()                                                                .sort_values(by=[level_1,level_2,'day'])
    
    # Visualize by components of level 
    l1, l2 = dt_level[level_1].unique(), dt_level[level_2].unique() 
    iterables = [(a, b) for a in l1 for b in l2]

    for i in iterables: 
        a, b = i
        df = dt_level[(dt_level[level_1] == a) & (dt_level[level_2] == b)]
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(16,8))
        plt.style.use('ggplot')
        ax1.plot(df['sales_day'], color = 'blue')
        ax1.set_xticklabels(labels = df.day.values, rotation=70)
        ax1.grid(False)
        ax1.set_title(f"Time series of sales for {a} and {b}")
        ax2.plot(df['sell_price'], color = 'red')
        ax2.set_xticklabels(labels = df.day.values, rotation=70)
        ax2.grid(False)
        ax2.set_title(f"Time series of price for {a} and {b}")
        break


# In[15]:


# Level: Categories of each store 
timeseries_particular(dt_work,'store_id', 'cat_id')


# In[16]:


# Level: departments by store
timeseries_particular(dt_work, 'store_id', 'dept_id')


# In[17]:


def timeseries_range_slider(data:'pd.DataFrame',level_1:str, level_2:str):
     
    # Group data by level 
    dt_level = data.groupby([level_1,level_2, 'date']).agg({'sales_day':'sum','sell_price':'mean'})                                                       .reset_index()                                                       .sort_values(by=[level_1,level_2,'date'])
    
    # Visualize by components of level 
    l1, l2 = dt_level[level_1].unique(), dt_level[level_2].unique() 
    iterables = [(a, b) for a in l1 for b in l2]

    for i in iterables: 
        a, b = i
        df = dt_level[(dt_level[level_1] == a) & (dt_level[level_2] == b)]
        
        mxm = max(df['sales_day'])
        if mxm < 50:
            weight = 5
        elif mxm < 100:
            weight = 10
        elif mxm < 200: 
            weight = 25  
        else:
            weight = 35
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sales_day'], 
                                 name='Sales',
                                 line_color='deepskyblue'))

        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sell_price']*weight, 
                                 name='Price average',
                                 line_color='dimgray'))

        fig.update_layout(title_text=f'Time Series of sales with Rangeslider for {a} and {b}',
                          xaxis_rangeslider_visible=True)
        fig.show()
        break


# In[18]:


# Timeseries with a range slider 
timeseries_range_slider(dt_work, 'store_id', 'dept_id')


# In[19]:


def timeseries_range_slider_event(data:'pd.DataFrame',level_1:str, level_2:str, level_3:str, level_4:str):
     
    # Group data by level 
    dt_level = data.groupby([level_1,level_2, level_3]).agg({'sales_day':'sum','sell_price':'mean'})                                                       .reset_index()                                                       .sort_values(by=[level_1,level_2,level_3])
    
    dt_ant = data.groupby([level_1, level_2, level_3, level_4]).agg({'sales_day':'sum','sell_price':'mean'})                                                                                .reset_index()                                                                                .sort_values(by=[level_1,level_2,level_3])
    
    
    # Visualize by components of level 
    l1, l2 = dt_level[level_1].unique(), dt_level[level_2].unique() 
    iterables = [(a, b) for a in l1 for b in l2]

    for i in iterables: 
        a, b = i
        df = dt_level[(dt_level[level_1] == a) & (dt_level[level_2] == b)]
        df_ant = dt_ant[(dt_ant[level_1] == a) & (dt_ant[level_2] == b)]
        
        # Annotations
        events = df_ant[['date', 'event_name_1']]
        events = events.set_index('date')
        ants = [dict(x = date, y = 10, 
                       xref = 'x', yref = 'y', 
                       textangle = 45,
                       font=dict(color = 'black', size = 8),
                       text = f'{value[0]}')
                       for date, value in zip(events.index, 
                                              events.values)]
        
        # Weights for price
        mxm = max(df['sales_day'])
        if mxm < 50:
            weight = 5
        elif mxm < 100:
            weight = 10
        elif mxm < 200: 
            weight = 25  
        else:
            weight = 35
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sales_day'], 
                                 name='Sales',
                                 line_color='deepskyblue'))

        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sell_price']*weight, 
                                 name='Price average',
                                 line_color='dimgray'))

        fig.update_layout(title_text=f'Time Series of sales with Rangeslider for {a} and {b}',
                          xaxis_rangeslider_visible=True,
                          annotations = ants,
                          height=800,
                          width=1100)
        
        
        fig.show()
        break


# In[20]:


# Mark special events as annotation in the time series
timeseries_range_slider_event(dt_work,'store_id','dept_id','date','event_name_1')


# In[21]:


def timeseries_range_slider_snap(data:'pd.DataFrame',level_1:str, level_2:str, level_3:str):
     
    # Group data by level 
    dt_level = data.groupby([level_1,level_2, level_3]).agg({'sales_day':'sum','sell_price':'mean'})                                                        .reset_index()                                                        .sort_values(by=[level_1,level_2,level_3])

    # Visualize by components of level 
    l1, l2 = dt_level[level_1].unique(), dt_level[level_2].unique() 
    iterables = [(a, b) for a in l1 for b in l2]
    
    states = ['WI','CA','TX']

    for i in iterables: 
        a, b = i
        df = dt_level[(dt_level[level_1] == a) & (dt_level[level_2] == b)]

        # Annotations
        for s in states:
            if a.startswith(s):
                col = 'snap_' + s
                df_ant = data[(data[col] == 1) & (data[level_1] == a)].groupby([level_1, level_2, level_3, col])                                                                               .agg({'sales_day':'sum','sell_price':'mean'})                                                                               .reset_index()                                                                               .sort_values(by=[level_1, level_2, level_3])    
    
        events = df_ant[['date', col]]
        events = events.set_index('date')

        ants = [dict(x = date, 
                     y = 10, 
                     xref = 'x', 
                     yref = 'y', 
                     textangle = 45,
                     font=dict(color = 'black', size = 6),
                               text = col)
                     for date, value in zip(events.index, 
                                            events.values)]
        
        
        # Weights for price
        mxm = max(df['sales_day'])
        if mxm < 50:
            weight = 5
        elif mxm < 100:
            weight = 10
        elif mxm < 200: 
            weight = 25  
        else:
            weight = 35
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sales_day'], 
                                 name='Sales',
                                 line_color='deepskyblue'))

        fig.add_trace(go.Scatter(x=df.date, 
                                 y=df['sell_price']*weight, 
                                 name='Price average',
                                 line_color='dimgray'))

        fig.update_layout(title_text=f'Time Series of sales with Rangeslider for {a} and {b}',
                          xaxis_rangeslider_visible=True,
                          annotations = ants,
                          height=600,
                          width=800)
        
        
        fig.show()
        break


# In[22]:


# Timeseries annotated with SNAP days
timeseries_range_slider_snap(dt_work, 'store_id', 'dept_id', 'date')

