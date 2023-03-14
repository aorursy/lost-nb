#!/usr/bin/env python
# coding: utf-8



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




import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.optimize import curve_fit
import datetime
import matplotlib.dates as mdates
import json
import pandas as pd

def sigmoid_sqrt_func(x, a, b, c, d, e):
    return c + d / (1.0 + np.exp(-a*x+b)) + e*x**0.5

def sigmoid_linear_func(x, a, b, c, d, e):
    return c + d / (1.0 + np.exp(-a*x+b)) + e*0.1*x

def sigmoid_quad_func(x, a, b, c, d, e, f):
    return c + d / (1.0 + np.exp(-a*x+b)) + e*0.1*x + f*0.001*x*x

def sigmoid_func(x, a, b, c, d):
    return c + d / (1.0 + np.exp(-a*x+b))

def exp_func(x, a, b, c, d):
    return c + d * np.exp(a*x+b)

def func_fitting(y, func=sigmoid_func, x_scale=50.0, y_scale=10000.0, start_pred=8, AN=0, MAXN=60, PN=15, b=5):
    # y = [549, 730, 1058, 1423, 2714, 3554, 4903, 5806, 7153, 9074, 11177, 13522, 16678, 19665, 22112, 24953]
    # start_date = '01/24/2020'
    #自定义函数 e指数形式

    #定义x、y散点坐标
    x = range(len(y))
    x_real = np.array(x)/x_scale
    
    y_real = np.array(y)/y_scale

    x_train = x_real
    y_train = y_real

    def next_day_pred(AN, BN):
        x_train = x_real[AN:BN]
        y_train = y_real[AN:BN]

        popt, pcov = curve_fit(func, x_train, y_train, 
                               method='trf', 
                               maxfev=20000, 
                               p0=(1, 0, 0, 1),
                               #sigma=(np.arange(1, len(x_train)+1)) / (len(x_train)),
                               #sigma=np.arange(16 - len(x_train), 16) / 15,
                               bounds=[(-b, -np.inf, -np.inf, -b), (b, np.inf, np.inf, b)],
                              )
        #print(popt)
        #print(np.arange(16 - len(x_train), 16) / 15)
        #print((np.arange(1, len(x_train)+1)) / (len(x_train)))
        
        x_pred = np.array(range(MAXN))/x_scale
        y_pred = func(x_pred, *popt)

        return x_pred, y_pred

    NP = start_pred
    y_pred = [np.nan]*NP #y_real[:NP].tolist()
    y_pred_list = []
    for BN in range(NP, len(y_real)):
        #x_pred, y_pred_ = next_day_pred(AN, BN)
        x_pred, y_pred_ = next_day_pred(BN-PN, BN)
        y_pred.append(y_pred_[BN])
        y_pred_list.append(y_pred_)
    for BN in range(len(y_real), len(y_pred_)):
        y_pred.append(y_pred_[BN])

    y_pred = np.array(y_pred)
    y_pred_list = np.array(y_pred_list)
    y_pred_std = np.std(y_pred_list[-2:], axis=0)
    
    return x_real*x_scale, y_real*y_scale, x_train*x_scale, y_train*y_scale,             x_pred*x_scale, y_pred*y_scale, y_pred_std*y_scale

def draw_figure(start_date, title, x_real, y_real, x_train, y_train, x_pred, y_pred, y_pred_std):
    def to_date(idx):
        idx = np.round(idx)
        return datetime.datetime.strptime(start_date, '%m/%d/%Y').date() + datetime.timedelta(days=idx)
    #绘图
    
    fig, ax1 = plt.subplots(figsize=[14, 7])

    plot1 = ax1.plot(list(map(to_date, x_real)), y_real, 'gs',label='original')
    plot2 = ax1.plot(list(map(to_date, x_pred)), y_pred, 'r',label='predict')
    plot3 = ax1.fill_between(list(map(to_date, x_pred)), 
                             np.maximum(0, (y_pred-y_pred_std)), 
                             (y_pred+y_pred_std),
        alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848')

    plot0 = ax1.plot(list(map(to_date, x_train)), y_train, 'y.',label='history')

    ax2=ax1.twinx()
    ax2.plot(list(map(to_date, x_real))[1:], (y_real[1:]-y_real[:-1]), '-s',label='original add')
    ax2.plot(list(map(to_date, x_pred))[1:], (y_pred[1:]-y_pred[:-1]), '-',label='pred add')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    plt.xlabel('x')
    plt.ylabel('y')
    fig.legend(loc=2) #指定legend的位置右下角
    plt.title(title)
    plt.savefig('{}.pdf'.format(title))
    plt.show()
    
    date = list(map(to_date, x_pred))
    pred = y_pred
    real = y_real
    for i in range(len(pred)):
        if i < len(real):
            print('{}\t{:.0f}\t{:.0f}\t{:.3f}'.format(date[i], real[i], pred[i], np.abs(pred[i]-real[i])/real[i]*100))
        else:
            print('{}\t-\t{:.0f}'.format(date[i], pred[i]))
            
    return pred




train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
pred_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train_data = train_data.fillna(value='NULL')
test_data = test_data.fillna(value='NULL')




test_data




train_date_list = train_data.iloc[:, 3].unique()
print(len(train_date_list))
print(train_date_list)

test_date_list = test_data.iloc[:, 3].unique()
print(len(test_date_list))
print(test_date_list)

len(train_data.groupby(['Province_State', 'Country_Region']))
len(test_data.groupby(['Province_State', 'Country_Region']))




start_date = '01/22/2020'
start_pred = 74
start_submit = 64
len_pred = 30

test_date_list = test_data.iloc[:, 3].unique()
#print(test_date_list)

test_data_filled = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
test_data_filled = test_data_filled.fillna(value='NULL')
test_data_filled['ConfirmedCases'] = pred_data['ConfirmedCases']
test_data_filled['Fatalities'] = pred_data['Fatalities']

for idx, (k, v) in enumerate(train_data.groupby(['Province_State', 'Country_Region'])):
    #if k != ('NULL', 'Italy') and \
    #   k != ('Curacao', 'Netherlands') and \
    #   k != ('Guangdong', 'China') and idx != 9 and \
    #   k[1] != 'Germany' and \
    #   k[1] != 'Spain':
    #    continue
    #if k[1] != 'US':
    #    continue
    #if k[0] != 'Hong Kong' and 'Tai' not in k[1]:
    #    continue
    #if k[1] != 'India':
    #    continue
    #if k[1] != 'Afghanistan':
    #    continue
        
    print(idx, k, end=' ')
    
    b_cc, b_f = 5, 3
    
    if k[1] == 'China':
        b_cc = 1
        b_f = 1
        if k[0] == 'Hong Kong':
            b_cc = 3
    elif k[1] == 'Italy':
        b_cc = 10
        b_f = 3
    elif k[1] == 'US':
        b_cc = 5
        b_f = 3
    elif k[1] == 'Spain':
        b_cc = 4
        b_f = 3
    
    hist_num = v.loc[:,'ConfirmedCases'].tolist()
    #print(hist_num)
    ret = func_fitting(hist_num, y_scale=max(1000, np.max(hist_num)), b=b_cc,
                       start_pred=start_pred, PN=10, MAXN=len(hist_num)+len_pred)
    ret = list(ret)
    real_cc = np.round(np.array(ret[1]))
    pred_cc = np.round(np.array(ret[5]))
    for i in range(len(real_cc)):
        pred_cc[i] = real_cc[i]
    pred_cc = pred_cc[start_submit:]
    
    print(pred_cc)
    
    hist_num = v.loc[:,'Fatalities'].tolist()
    #print(hist_num)
    ret = func_fitting(hist_num, y_scale=max(1000, np.max(hist_num)), b=b_f,
                       start_pred=start_pred, PN=10, MAXN=len(hist_num)+len_pred)
    ret = list(ret)
    real_f = np.round(np.array(ret[1]))
    pred_f = np.round(np.array(ret[5]))
    for i in range(len(real_f)):
        pred_f[i] = real_f[i]
    pred_f = pred_f[start_submit:]
    
    print(pred_f)
    
    for i in range(14, len(pred_cc)):
        if pred_f[i] < 20 and pred_cc[i] > 200 and k[1] != 'China':
            pred_f[i] = pred_cc[i] * 0.01 * np.log10(pred_cc[i])
        #elif k[1] == 'China':
        #    print(k)
            
    #print(pred_cc)
    print(pred_f)
    print(pred_cc[-1], pred_f[-1])
    
    for i in range(len(pred_cc)):
        index = (test_data_filled['Province_State'] == k[0]) &                 (test_data_filled['Country_Region'] == k[1]) &                 (test_data_filled['Date'] == test_date_list[i])
        test_data_filled.loc[index, 'ConfirmedCases'] = pred_cc[i]
        test_data_filled.loc[index, 'Fatalities'] = pred_f[i]




submission = test_data_filled.loc[:,['ForecastId', 'ConfirmedCases', 'Fatalities']]




submission.to_csv("submission.csv", index=False)
submission.head(500)

