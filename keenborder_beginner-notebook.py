#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




# Memory reduction helper function:
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns
        col_type = df[col].dtypes
        if col_type in numerics: #numerics
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




data_pass = '/kaggle/input/m5-forecasting-accuracy/'

# Sales quantities:
sales_train_validation = pd.read_csv(data_pass+'sales_train_validation.csv')

# Calendar to get week number to join sell prices:
calendar = pd.read_csv(data_pass+'calendar.csv')
calendar = reduce_mem_usage(calendar)

# Sell prices to calculate sales in USD:
sales_prices= pd.read_csv(data_pass+'sell_prices.csv')
sales_prices = reduce_mem_usage(sales_prices)




sales_train_validation.head()




sales_prices.head()




calendar.head()




item_sold=sales_prices.groupby('item_id').count()
item_sold=item_sold.rename(columns={'store_id':'item_sold'}).drop(['wm_yr_wk','sell_price'],1).sort_values(by='item_sold').reset_index()
item_sold['category']=item_sold['item_id'].apply(lambda x:re.findall("[a-zA-Z]+",x)[0])
item_sold.groupby('category')['item_sold'].count().plot(kind='bar')




overall_pattern_sales=sales_prices.groupby('wm_yr_wk').count()
overall_pattern_sales=overall_pattern_sales.reset_index().rename(columns={'wm_yr_wk':'Time','store_id':'Sales'}).drop(['item_id','sell_price'],1)
sns.lineplot(x="Time", y="Sales", data=overall_pattern_sales)




revenue_per_item=sales_prices.groupby('item_id')['sell_price'].agg({'Price':np.mean})
revenue_per_item['item_sold']=sales_prices.groupby('item_id').count()['sell_price']
revenue_per_item['revene']=revenue_per_item['item_sold']*revenue_per_item['Price']
revenue_per_item.reset_index(inplace=True)
revenue_per_item['category']=revenue_per_item['item_id'].apply(lambda x:re.findall("[a-zA-Z]+",x)[0])
revenue_per_item.groupby('category')['revene'].sum().plot(kind='bar')




sales_count_d=sales_train_validation.drop(['id','dept_id','cat_id','store_id','state_id'],1).groupby('item_id').sum().T
sales_count_d.head()




sales_unit_dates=sales_count_d.merge(calendar.set_index('d'),left_index=True,right_index=True,how='inner')




import random
import matplotlib.dates as mdates

def random_product_plot():
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    items_picked=random.sample(list(set(sales_prices['item_id'].tolist())),6)
    fig=plt.figure(figsize=(20,20))
    color=['r','b','g','m','k','y']
    
    for i in range(6):
        ax=fig.add_subplot(3,2,i+1)
        item_to_plot=items_picked[i]
        c_o=color[i]
        ax.plot(mdates.date2num(pd.to_datetime(sales_unit_dates['date'])),sales_unit_dates[item_to_plot],c=c_o)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
                
        datemin = np.datetime64(sales_unit_dates['date'][0], 'Y')
        datemax = np.datetime64(sales_unit_dates['date'][-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
        ax.set_xlabel(item_to_plot)
        ax.grid(True)




random_product_plot()




random_product_plot()




random_product_plot()




states_sales=sales_train_validation.drop(['id','dept_id','cat_id','store_id','item_id'],1).groupby('state_id').sum().T




states_sales=states_sales.merge(calendar.set_index('d'),left_index=True,right_index=True,how='inner')




states_sales.head()




def state_sales_plot():
    states=states_sales.columns.tolist()
    fig=plt.figure(figsize=(20,20))
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    color=['r','b','g']
    
    for i in range(3):
        ax=fig.add_subplot(3,1,i+1)
        c_o=color[i]
        ax.plot(mdates.date2num(pd.to_datetime(states_sales['date'])),states_sales[states[i]],c=c_o)
        
        
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
                
        datemin = np.datetime64(states_sales['date'][0], 'Y')
        datemax = np.datetime64(states_sales['date'][-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
        ax.set_xlabel(states[i])
        ax.set_ylabel('Total--Sales')
        ax.grid(True)
        




state_sales_plot()




mean_prices=sales_prices.groupby('item_id')['sell_price'].mean()
mean_prices.head()




revene_data_product=sales_unit_dates.drop(sales_unit_dates.columns.tolist()[3049:],1).T.merge(mean_prices,right_index=True,left_index=True
                                                                         ,how='inner')




columns_d_list=revene_data_product.columns.tolist()
for column_index in range(len(columns_d_list)-1):
    revene_data_product['{}'.format(columns_d_list[column_index])]=revene_data_product['sell_price']*revene_data_product[columns_d_list[column_index]]




revene_data_product.head()




columns=revene_data_product.T.columns.tolist()
columns.append('date')
revene_data_product=revene_data_product.drop('sell_price',1).T.merge(calendar.set_index('d'),right_index=True,left_index=True,how='inner')[columns]




revene_data_product.head()




def random_product_revenue_plot():
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    items_picked=random.sample(list(set(sales_prices['item_id'].tolist())),6)
    fig=plt.figure(figsize=(20,20))
    fig.suptitle('Revenue Generated by Different Products', fontsize=16)
    color=['r','b','g','m','k','y']
    
    for i in range(6):
        ax=fig.add_subplot(3,2,i+1)
        item_to_plot=items_picked[i]
        c_o=color[i]
        ax.plot(mdates.date2num(pd.to_datetime(revene_data_product['date'])),revene_data_product[item_to_plot],c=c_o)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
                
        datemin = np.datetime64(revene_data_product['date'][0], 'Y')
        datemax = np.datetime64(revene_data_product['date'][-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
        ax.set_xlabel(item_to_plot)
        ax.set_ylabel('Revenue')
        ax.grid(True)




random_product_revenue_plot()




random_product_revenue_plot()




random_product_revenue_plot()




#Create date index
date_index = calendar['date']
dates = date_index[0:1913]
dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]



# Create a data frame for items sales per day with item ids (with Store Id) as columns names  and dates as the index 
sales_train_validation['item_store_id'] = sales_train_validation.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)
DF_Sales = sales_train_validation.loc[:,'d_1':'d_1913'].T
DF_Sales.columns = sales_train_validation['item_store_id'].values

#Set Dates as index 
DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
DF_Sales.index = pd.to_datetime(DF_Sales.index)
DF_Sales.head()




series=np.array(DF_Sales.loc[:,'FOODS_3_825_WI_3'].values.tolist())
time=np.array(DF_Sales.index.tolist())
series.shape




sns.distplot(series, kde=False, rug=True);




split_time=1850
time_train=time[:]
x_train=series[:]
time_valid=time[split_time:]
x_valid=series[split_time:]

window_size=28
batch_size=32
shuffle_buffer_size=1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)




tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 28
batch_size = 32

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[28,1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
# optimizer = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule]) 




def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast




rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[:, -1, 0]
rnn_forecast.shape




rnn_forecast_valid = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast_valid = rnn_forecast_valid[split_time - window_size:-1, -1, 0]
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast_valid).numpy()




dates=dates_list[window_size-1:]
orginal_sales=series[window_size-1:]




figure=plt.figure(figsize=(20,20))
ax=figure.add_subplot(111)
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


ax.plot(mdates.date2num(pd.to_datetime(dates)),orginal_sales,c='r',label='Real')
ax.plot(mdates.date2num(pd.to_datetime(dates)),rnn_forecast,c='b',label='Predicted')
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

ax.grid(True)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
ax.set_title('Real Vs Predicted Sales Over Time')











