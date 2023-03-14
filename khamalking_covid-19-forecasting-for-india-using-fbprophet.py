#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import fbprophet as fbp
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
url = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv', 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv']
confirmed = pd.read_csv(url[0], error_bad_lines = False)
deaths = pd.read_csv(url[1], error_bad_lines = False)
recovered = pd.read_csv(url[2], error_bad_lines = False)
#print(confirmed.head(), deaths.head(), recovered.head())
print(confirmed.shape, recovered.shape, deaths.shape)


# In[4]:


row = confirmed.iloc[:, -1].idxmax() 
index = confirmed.index[confirmed.iloc[:, 1] == 'India']
confirmed.loc[index]


# In[5]:


conCopy = confirmed
rCopy = recovered
deaCopy = deaths


# In[6]:


dropColumns = ['Province/State', 'Country/Region', 'Lat', 'Long']
conCopy = conCopy.drop(dropColumns,axis=1)


# In[7]:


lockdown0 = pd.DataFrame({"holiday" : "lockdown-0", "ds" : pd.to_datetime(pd.DataFrame({'year': [2020], 'month':  [3], 'day': [22]}))}) 
lockdown1 = pd.DataFrame({"holiday" : "lockdown-1", "ds" : pd.date_range(start = '2020-03-25', periods = 21)})
holidays = pd.concat([lockdown0, lockdown1])
holidays = holidays.reset_index(drop=True)


# In[ ]:


lockdown0 = pd.DataFrame({"holiday" : "lockdown-0", "ds" : pd.to_datetime(pd.DataFrame({'year': [2020], 'month':  [3], 'day': [22]}))}) 
lockdown1 = pd.DataFrame({"holiday" : "lockdown-1", "ds" : pd.date_range(start = '2020-03-25', periods = 21)})
lockdown2 = pd.DataFrame({"holiday" : "lockdown-2", "ds" : pd.date_range(start = '2020-04-20', periods = 28)})
holidays = pd.concat([lockdown0, lockdown1, lockdown2])
holidays = holidays.reset_index(drop=True)


# In[ ]:


lockdown0 = pd.DataFrame({"holiday" : "lockdown-0", "ds" : pd.to_datetime(pd.DataFrame({'year': [2020], 'month':  [3], 'day': [22]}))}) 
lockdown1 = pd.DataFrame({"holiday" : "lockdown-1", "ds" : pd.date_range(start = '2020-03-25', periods = 21)})
lockdown2 = pd.DataFrame({"holiday" : "lockdown-2", "ds" : pd.date_range(start = '2020-04-20', periods = 28)})
lockdown3 = pd.DataFrame({"holiday" : "lockdown-3", "ds" : pd.date_range(start = '2020-05-25', periods = 15)})
holidays = pd.concat([lockdown0, lockdown1, lockdown2, lockdown3])
holidays = holidays.reset_index(drop=True)


# In[ ]:


lockdown0 = pd.DataFrame({"holiday" : "lockdown-0", "ds" : pd.to_datetime(pd.DataFrame({'year': [2020], 'month':  [3], 'day': [22]}))}) 
lockdown1 = pd.DataFrame({"holiday" : "lockdown-1", "ds" : pd.date_range(start = '2020-03-25', periods = 49)})
holidays = pd.concat([lockdown0, lockdown1])
holidays = holidays.reset_index(drop=True)


# In[8]:


changepoints = ['3/31/20']


# In[10]:


data = pd.DataFrame(data = [conCopy.loc[index[0]]])
data = data.T
data = data.reset_index()
data.rename(columns={'index': 'ds', index[0]: 'y'}, inplace=True)
#print(data.describe())
model = fbp.Prophet(growth = "linear", holidays = holidays, seasonality_mode = "multiplicative", changepoints = changepoints, changepoint_prior_scale = 30, seasonality_prior_scale = 10, holidays_prior_scale = 20, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,).add_seasonality(name = "daily", period = 1, fourier_order = 5).add_seasonality(name = "weekly", period = 14, fourier_order = 5)
model.fit(data)
future = model.make_future_dataframe(periods = 70)
print(future.tail())
forecast = model.predict(future)
#print(data[60:74])
print(forecast[['ds', 'yhat']].tail())
plt.plot(forecast['ds'][74:143], forecast['yhat'][74:143])


# In[17]:


type(forecast['yhat'][74:143])
#result = pd.concat([df1, forecast[['yhat']][74:143]], ignore_index=True)


# In[ ]:


df_cv = cross_validation(model, initial='20 days', period='10 days', horizon = '10 days')
df_cv


# In[ ]:


df_p = performance_metrics(df_cv)
df_p.head()


# In[ ]:


model.plot_components(forecast)


# In[ ]:


predictions = []
for i in range(len(conCopy)):
    data = pd.DataFrame(data = [conCopy.loc[i]])
    data = data.T
    data = data.reset_index()
    data.rename(columns={'index': 'ds', i: 'y'}, inplace=True)
    #print(data.describe())
    model = fbp.Prophet(growth = "linear", seasonality_mode = "multiplicative", changepoint_prior_scale = 30, seasonality_prior_scale = 20, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,).add_seasonality(name = "daily", period = 1, fourier_order = 5).add_seasonality(name = "weekly", period = 14, fourier_order = 5)
    model.fit(data)
    future = model.make_future_dataframe(periods = 70)
    #print(future.tail())
    forecast = model.predict(future)
    #print(forecast[['ds', 'yhat']].tail())
    if i == 0:
        predictions = forecast[['yhat']][74:143]
    else:
        predictions = pd.concat([predictions, forecast[['yhat']][74:143]], axis = 1, ignore_index = True)
    plt.plot(forecast['ds'][74:143], forecast['yhat'][74:143])
    print(i)
    
#error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])


# In[ ]:


from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')


# In[ ]:


data = pd.DataFrame(data = [conCopy.loc[0]])
data = data.T
data = data.reset_index()
data.rename(columns={'index': 'ds', 0: 'y'}, inplace=True)


# In[ ]:


from tqdm.notebook import tqdm as tqdm
for row in tqdm(conCopy.loc[0:3]):
    print(row)


# In[ ]:





# In[ ]:





# In[ ]:


import pyramid.arima as pa
auto_arima(df['Monthly beer production'], seasonal=True).summary()
pip install pmdarima
#from pmdarima import auto_arima

