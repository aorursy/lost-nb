#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt
from statsmodels.robust import mad

import warnings
warnings.filterwarnings('ignore')


# In[2]:


input_dir = '../input/m5-forecasting-accuracy/'

sales_train_validation = pd.read_csv(input_dir+'sales_train_validation.csv')
calendar = pd.read_csv(input_dir+'calendar.csv')
sales_prices = pd.read_csv(input_dir+'sell_prices.csv')


# In[3]:


sales_train_validation.head()


# In[4]:


calendar.head()


# In[5]:


sales_prices.head()


# In[6]:


def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # 计算当前占用的内存 
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    # 循环每一列
    for col in df.columns:

        # 获取每一列的数据类型
        col_type = df[col].dtypes

        # 如果数据类型属于上面定义的类型之
        if col_type in numerics:

            # 计算该列数据的最小值和最大值 用于我们指定相应的数据类型 
            c_min = df[col].min()
            c_max = df[col].max()

            # 如果 该列的数据类型属于 int 类型，然后进行判断
            if str(col_type)[:3] == 'int':
                # 如果 该列最小的值 大于int8类型的最小值，并且最大值小于int8类型的最大值，则采用int8类型 
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                # 同上
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                # 同上
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

                # 同上
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # 否则 则采用 float 的处理方法       
            else:

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[7]:


sales_bd = np.round(sales_train_validation.memory_usage().sum()/(1024*1024),1)
calendar_bd = np.round(calendar.memory_usage().sum()/(1024*1024),1)
prices_bd = np.round(sales_prices.memory_usage().sum()/(1024*1024),1)


# In[8]:


sales_train_validation = reduce_mem_usage(sales_train_validation)
sales_prices = reduce_mem_usage(sales_prices)
calendar = reduce_mem_usage(calendar)


# In[9]:


sales_ad = np.round(sales_train_validation.memory_usage().sum()/(1024*1024),1)
calendar_ad = np.round(calendar.memory_usage().sum()/(1024*1024),1)
prices_ad = np.round(sales_prices.memory_usage().sum()/(1024*1024),1)


# In[10]:


# memory   # no melt


# In[11]:


# memory  # melt


# In[12]:


memory = pd.DataFrame({
    'DataFrame':['sale_train_val','sale_prices','calendar'],
    'Before memory reducing':[sales_bd,prices_bd,calendar_bd],
    'After memory reducing':[sales_ad,prices_ad,calendar_ad],

})

memory = pd.melt(memory,id_vars='DataFrame',var_name='Status',value_name='Memory(MB)')
memory.sort_values('Memory(MB)',inplace=True)
fig = px.bar(memory,x='DataFrame',y='Memory(MB)',color='Status',barmode='group',text='Memory(MB)')
fig.update_traces(textposition='outside')
fig.update_layout(template='seaborn',title='Effect of Downcasting')
fig.show()


# In[13]:


ids = sorted(list(sales_train_validation['id']))

d_cols = [c for c in sales_train_validation.columns if 'd_' in c ]


# In[14]:


x_1 = sales_train_validation.loc[sales_train_validation['id'] == ids[2]].set_index('id')[d_cols].values[0] 
x_2 = sales_train_validation.loc[sales_train_validation['id'] == ids[6]].set_index('id')[d_cols].values[0]
x_3 = sales_train_validation.loc[sales_train_validation['id'] == ids[7]].set_index('id')[d_cols].values[0]

fig = make_subplots(rows=3,cols=1)

fig.add_traces(go.Scatter(x=np.arange(len(x_1)),y=x_1,showlegend=False,mode='lines',name='First sample',marker=dict(color=next(color_cycle)))
               ,rows=1,cols=1)

fig.add_traces(go.Scatter(x=np.arange(len(x_2)),y=x_2,showlegend=False,mode='lines',name='Second sample',marker=dict(color=next(color_cycle)))
                ,rows=2,cols=1)

fig.add_traces(go.Scatter(x=np.arange(len(x_3)),y=x_3,showlegend=False,mode='lines',name='Third sample',marker=dict(color=next(color_cycle)))
                ,rows=3,cols=1)

fig.update_layout(height =1200,width=800,title_text='Sample sales')
fig.show()


# In[15]:


x_1 = sales_train_validation.loc[sales_train_validation['id'] == ids[2]].set_index('id')[d_cols].values[0][0:90] 
x_2 = sales_train_validation.loc[sales_train_validation['id'] == ids[6]].set_index('id')[d_cols].values[0][90:180] 
x_3 = sales_train_validation.loc[sales_train_validation['id'] == ids[7]].set_index('id')[d_cols].values[0][1800:] 

fig = make_subplots(rows=3,cols=1)

fig.add_traces(go.Scatter(x=np.arange(len(x_1)),y=x_1,showlegend=False,mode='lines',name='First sample',marker=dict(color=next(color_cycle)))
               ,rows=1,cols=1) # 指定绘制的子图

fig.add_traces(go.Scatter(x=np.arange(len(x_2)),y=x_2,showlegend=False,mode='lines',name='Second sample',marker=dict(color=next(color_cycle)))
                ,rows=2,cols=1)

fig.add_traces(go.Scatter(x=np.arange(len(x_3)),y=x_3,showlegend=False,mode='lines',name='Third sample',marker=dict(color=next(color_cycle)))
                ,rows=3,cols=1)

fig.update_layout(height =1200,width=800,title_text='Sample sales')
fig.show()


# In[16]:


# from statsmodels.tsa.seasonal import seasonal_decompose


# In[17]:


# new_x1=sales_train_validation.loc[sales_train_validation['id'] == ids[2]].set_index('id')[d_cols][0:90]
# new_x1.T


# In[18]:


# decomposition = seasonal_decompose(new_x1,freq=12)
# trend = decomposition.trend #趋势效应
# seasonal = decomposition.seasonal #季节效应
# residual = decomposition.resid #随机效应
# plt.subplot(411)
# plt.plot(new_x1, label=u'原始数据')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label=u'趋势')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label=u'季节性')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label=u'残差')
# plt.legend(loc='best')
# plt.tight_layout()


# In[19]:


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


# In[20]:


y_w1 = denoise_signal(x_1)
y_w2 = denoise_signal(x_2)
y_w3 = denoise_signal(x_3)


# In[21]:


fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(30,20))

ax[0,0].plot(x_1,color='seagreen',marker='o')
ax[0,0].set_title('Original Sales',fontsize=24)
ax[0,0].plot(y_w1,color='red',marker='.')
ax[0,0].set_title('Original Sales',fontsize=24)
ax[0,1].plot(y_w1,color='red',marker='.')
ax[0,1].set_title('Original Sales',fontsize=24)

ax[1,0].plot(x_2,color='seagreen',marker='o')
ax[1,0].set_title('Original Sales',fontsize=24)
ax[1,0].plot(y_w2,color='red',marker='.')
ax[1,0].set_title('Original Sales',fontsize=24)
ax[1,1].plot(y_w2,color='red',marker='.')
ax[1,1].set_title('Original Sales',fontsize=24)

ax[2,0].plot(x_3,color='seagreen',marker='o')
ax[2,0].set_title('Original Sales',fontsize=24)
ax[2,0].plot(y_w3,color='red',marker='.')
ax[2,0].set_title('Original Sales',fontsize=24)
ax[2,1].plot(y_w3,color='red',marker='.')
ax[2,1].set_title('Original Sales',fontsize=24)

fig.show()


# In[22]:


def average_smoothing(signal, kernel_size=3, stride=1):
    sample = [0]*(kernel_size-stride) # 通过 len(y_a1) 可以发现与原始数据同长度
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend([np.mean(signal[start:end])])
    return np.array(sample)


# In[23]:


y_a1 = average_smoothing(x_1)
y_a2 = average_smoothing(x_2)
y_a3 = average_smoothing(x_3)

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_a2, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_a3, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=3, col=1
)


fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()


# In[24]:


fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(30,20))

ax[0,0].plot(x_1,color='seagreen',marker='o')
ax[0,0].set_title('Original Sales',fontsize=24)
ax[0,1].plot(y_a1,color='red',marker='x')
ax[0,1].set_title('Original Sales',fontsize=24)

ax[1,0].plot(x_2,color='seagreen',marker='o')
ax[1,0].set_title('Original Sales',fontsize=24)
ax[1,1].plot(y_a2,color='red',marker='.')
ax[1,1].set_title('Original Sales',fontsize=24)


ax[2,0].plot(x_3,color='seagreen',marker='o')
ax[2,0].set_title('Original Sales',fontsize=24)
ax[2,1].plot(y_a3,color='red',marker='.')
ax[2,1].set_title('Original Sales',fontsize=24)

fig.show()


# In[25]:


sales_train_validation.head()


# In[26]:


sales_prices.head()


# In[27]:


calendar.head()


# In[28]:


past_sales = sales_train_validation.set_index('id')[d_cols]             .T             .merge(calendar.set_index('d')['date'],left_index =True,right_index=True,).set_index('date')

store_list = sales_prices['store_id'].unique()
means = []

fig = go.Figure()

for s_id in  store_list:
    store_items = [s for s in past_sales.columns if s_id in s]
    data = past_sales[store_items].sum(axis=1).rolling(90).mean()
    means.append(np.mean(past_sales[store_items].sum(axis=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data, name=s_id))
                 
fig.update_layout(yaxis_title='Sales',xaxis_title='Time',title='Rolling average Sales vs Time (per store)')


# In[29]:


fig = go.Figure()
for s_id in  store_list:
    store_items = [s for s in past_sales.columns if s_id in s]
    data = past_sales[store_items].sum(axis=1).rolling(90).mean()
    fig.add_trace(go.Box(x=[s_id]*len(data), y=data, name=s_id))
fig.update_layout(yaxis_title='Sales',xaxis_title='Store',title='Rolling average Sales vs Store (per store)')


# In[30]:


fig = go.Figure()

# method 1
df = pd.DataFrame(np.transpose([means,store_list]))
df.columns=['Mean sales','Store name']
px.bar(df, y="Mean sales", x="Store name", color="Store name", title="Mean sales vs. Store name")

# method 2
# for i in np.transpose([means, store_list]):
#     fig.add_trace(go.Bar(y=[i[0]], x=[i[1]], marker = dict(color = next(color_cycle)),name=i[1]))
# fig.update_layout(yaxis_title="Mean Sales", xaxis_title="Store name", title="Mean Sales vs. Store name")


# In[31]:


cat_id_list = sales_train_validation['cat_id'].unique()

fig= go.Figure()

for cat_id in  cat_id_list:
    store_items = [s for s in past_sales.columns if cat_id in s]
    data = past_sales[store_items].sum(axis=1)
    fig.add_trace(go.Scatter(x=data.index,y=data,name=cat_id))
fig.update_layout(xaxis_title='time',yaxis_title='cat_id sales',title='cat_id Sales vs Time (per cat_id)')


# In[32]:


past_sales_clipped = past_sales.clip(0, 1)

fig = go.Figure()
for cat_id in  cat_id_list:
    store_items = [s for s in past_sales_clipped.columns if cat_id in s]
    data = past_sales_clipped[store_items].mean(axis=1) * 100
    fig.add_trace(go.Scatter(x=data.index,y=data,name=cat_id,mode='markers'))
fig.update_layout(xaxis_title='time',yaxis_title='% of Inventory with at least 1 sale',title='Inventory Sale Percentage by Date')


# In[33]:


print('The lowest sale date was:', past_sales.sum(axis=1).sort_values().index[0],
     'with', past_sales.sum(axis=1).sort_values().values[0], 'sales')
print('The lowest sale date was:', past_sales.sum(axis=1).sort_values(ascending=False).index[0],
     'with', past_sales.sum(axis=1).sort_values(ascending=False).values[0], 'sales')


# In[34]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels

from tqdm import tqdm


# In[35]:


store_sum = sales_train_validation.groupby(['store_id']).sum().T.reset_index(drop = True)
store_sum.head()


# In[36]:


train_datasets= store_sum.iloc[0:70]
val_datasets= store_sum.iloc[70:100]


# In[37]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[38]:


weeks_per_year = 365

time_series = store_sum["CA_1"]
sj_sc = seasonal_decompose(time_series, freq= weeks_per_year)
sj_sc.plot()

plt.show()


# In[39]:


fig = make_subplots(3,1)

fig.add_trace(go.Scatter(x=np.arange(70),y=train_datasets.iloc[:,0],marker=dict(color='seagreen'),name='Train'),row=1,col=1)
fig.add_trace(go.Scatter(x=np.arange(70,100),y=val_datasets.iloc[:,0],marker=dict(color='red'),name='Val'),row=1,col=1)

fig.add_trace(go.Scatter(x=np.arange(70),y=train_datasets.iloc[:,4],marker=dict(color='seagreen'), showlegend=False),row=2,col=1)
fig.add_trace(go.Scatter(x=np.arange(70,100),y=val_datasets.iloc[:,4],marker=dict(color='red'), showlegend=False),row=2,col=1)

fig.add_trace(go.Scatter(x=np.arange(70),y=train_datasets.iloc[:,7],marker=dict(color='seagreen'), showlegend=False),row=3,col=1)
fig.add_trace(go.Scatter(x=np.arange(70,100),y=val_datasets.iloc[:,7],marker=dict(color='red'), showlegend=False),row=3,col=1)

fig.update_layout(title='Sales volume of a commodity')


# In[40]:


store_col = [0,4,7]


# In[41]:


def sarima_train_test(t_series, p = 2, d = 1, r = 2, NUM_TO_FORECAST = 56, do_plot_results = True):
    
    NUM_TO_FORECAST = NUM_TO_FORECAST  # Similar to train test splits.
    dates = np.arange(t_series.shape[0])
    
    model = SARIMAX(t_series, order = (p, d, r), trend = 'c')
    results = model.fit()
#     results.plot_diagnostics(figsize=(12, 8))
#     plt.show()

    forecast = results.get_prediction(start = - NUM_TO_FORECAST)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    print(mean_forecast.shape)

    # Plot the forecast
#     plt.figure(figsize=(6,4))
#     plt.plot(dates[-NUM_TO_FORECAST:],
#             mean_forecast.values,
#             color = 'red',
#             label = 'forecast')


#     plt.plot(dates[-NUM_TO_FORECAST:],
#             t_series.iloc[-NUM_TO_FORECAST:],
#             color = 'blue',
#             label = 'actual')
#     plt.legend()
#     plt.title('Predicted vs. Actual Values')
#     plt.show()
    
    residuals = results.resid # 模型的残差
    mae_sarima = np.mean(np.abs(residuals))
    print('Mean absolute error: ', mae_sarima)
    print(results.summary())
    return mean_forecast


# In[42]:


predictions = []

for col in store_col:
    predictions.append(sarima_train_test(train_datasets.iloc[:,col],NUM_TO_FORECAST=28))

predictions = np.array(predictions).reshape((-1, 28))


# In[43]:


predictions


# In[44]:


pred_1 = predictions[0]
pred_2 = predictions[1]
pred_3 = predictions[2]

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(70), mode='lines', y=train_datasets.iloc[:,0].values, marker=dict(color="dodgerblue"),
               name="Train"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_datasets.iloc[:,0].values, mode='lines', marker=dict(color="darkorange"),
               name="Val"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
               name="Pred"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70), mode='lines', y=train_datasets.iloc[:,4].values, marker=dict(color="dodgerblue"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_datasets.iloc[:,4].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70), mode='lines', y=train_datasets.iloc[:,7].values, marker=dict(color="dodgerblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_datasets.iloc[:,7].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="ARIMA")
fig.show()

