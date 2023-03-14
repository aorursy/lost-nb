#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # search in texts as scrap
from collections import Counter
from fbprophet import Prophet
import matplotlib.pyplot as plt # graph
from plotly.offline import init_notebook_mode, iplot # graph
from plotly import graph_objs as go # graph

# Initialize plotly
init_notebook_mode(connected=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# importing and seeing the kaggle's file
k1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_1.csv')
k1.head()


# In[3]:


# importing and seeing the kaggle's file
k2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_2.csv')
k2.head()


# In[4]:


# importing and seeing the kaggle's file
t2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_2.csv')
t2.head()


# In[5]:


# importing and seeing the kaggle's file
t1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_1.csv')
t1.head()


# In[6]:


# importing and seeing the kaggle's file
ss2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_2.csv')
ss2.head()


# In[7]:


# importing and seeing the kaggle's file
ss1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv')
ss1.head()


# In[8]:


# view shapes of files 

print('shape t1')
print(t1.shape)
print('shape t2')
print(t2.shape)
print('shape k1')
print(k1.shape)
print('shape k2')
print(k2.shape)
print('shape ss1')
print(ss1.shape)
print('shape ss2')
print(ss2.shape)


# In[9]:


# Verify general behavior

#Total sum per column: 
t1.loc['Total',:]= t1.mean(axis=0)


# In[10]:


x = t1.tail(1)
x = x.T
x


# In[11]:


x.drop('Page', axis=0, inplace=True)
x.drop('lang', axis=0, inplace=True)


# In[12]:


x


# In[13]:


def plotly_df(df, title=''):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    iplot(fig, show_link=False)


# In[14]:


plotly_df(x, 'Total behavior')


# In[15]:


# TO use prophet is necessary 2 columns ds (with date ) and y with information
df = x.reset_index()
df.columns = ['ds', 'y']
df.tail(n=3)


# In[16]:


# ds MUST be date
df['ds'] = pd.to_datetime(df['ds']).dt.date


# In[17]:


# to try, last month will be predict 

prediction_size = 30
train_df = df[:-prediction_size]
train_df.tail(n=3)


# In[18]:


# It's time to predict with prophet 

m = Prophet()
m.fit(train_df)
future = m.make_future_dataframe(periods=prediction_size)
forecast = m.predict(future)


# In[19]:


forecast.tail(n=3)


# In[20]:


m.plot(forecast)


# In[21]:


m.plot_components(forecast)


# In[ ]:





# In[22]:


# minimizing outliers effects

outliers = []

for i in range(0,len(df)-1):
    if (df['y'][i] > forecast['yhat_upper'][i]):
        outliers.append({'info':'max', 'index':i, 'date':df['ds'][i] ,'val':df['y'][i], 'forecast val': forecast['yhat_upper'][i], 'factor': df['y'][i]/forecast['yhat'][i] })
    if (df['y'][i] < forecast['yhat_lower'][i]):
        outliers.append({'info':'min', 'index':i, 'date':df['ds'][i] ,'val':df['y'][i], 'forecast val': forecast['yhat_lower'][i], 'factor': df['y'][i]/forecast['yhat'][i]})
        
outliers = pd.DataFrame(outliers)


# In[23]:


df_new = df.copy()
df_new


# In[24]:


for i in range (0, len(outliers)-1 ):    
    df_new['y'][outliers['index'][i]] = df_new['y'][outliers['index'][i]] / outliers['factor'][i]


# In[25]:


# Prophet AGAIN with NEW df

# ds MUST be date
df_new['ds'] = pd.to_datetime(df_new['ds']).dt.date


# In[26]:


# to try, last month will be predict 

prediction_size = 30
train_df = df_new[:-prediction_size]
train_df.tail(n=3)


# In[27]:


# It's time to predict with prophet 

m = Prophet()
m.fit(train_df)
future = m.make_future_dataframe(periods=prediction_size)
forecast = m.predict(future)


# In[28]:


m.plot(forecast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


# Function to find page language

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page) # search text like mask
    if res:
        return res[0][0:2] # 2 first letters
    return 'na'

t1['lang'] = t1.Page.map(get_language) # new collumn
# see: map () relation between series = https://www.geeksforgeeks.org/python-pandas-map/


# In[30]:


plt.figure(figsize=(12, 6))
plt.title("Number of sites by languages", fontsize="18")
t1['lang'].value_counts().plot.bar(rot=0);


# In[31]:


t1.values[0]


# In[32]:


t1.T.index.values


# In[ ]:





# In[ ]:





# In[33]:


t1['mean'] = 


# In[34]:



#str_ = '2NE1_zh.wikipedia.org_all-access_spider'
str_[0:str_.find('.wikipedia.org')-3]

def get_subject(page):
    res = page[0:page.find('.wikipedia.org')-3]
    if res:
        return res
    return 'na'


# In[35]:


t1['subject'] = t1.Page.map(get_subject) # new collumn


# In[36]:


t1.Page


# In[37]:


#Split datasets in languages

lang_sets = {}
lang_sets['en'] = t1[t1.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = t1[t1.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = t1[t1.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = t1[t1.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = t1[t1.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = t1[t1.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = t1[t1.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = t1[t1.lang=='es'].iloc[:,0:-1]


# In[38]:


# access means for languages
sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]


# In[39]:


days = [r for r in range(sums['en'].shape[0])]


# In[40]:


################################ TESTS 



#re.search('{}.wikipedia.org','2NE1_zh.wikipedia.org_all-access_spider')
#[0][0:2]
# t1.Page.map(get_language)
# lang_sets
#lang_sets['es']
#len(lang_sets)
#japan = pd.DataFrame()
#japan['date_']= ja.index
#japan['count_'] = list(sums['ja'])
#japan
#pd.DataFrame(sums['ja'])

#pd.DataFrame(sums['ja']).index



# lang_sets['en'].iloc[:,1:].sum(axis=0)
#lang_sets['en'].shape[0]


# In[ ]:




