#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os


print("show data file")

for f in os.listdir("../input"):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
        
df_train = pd.read_csv('../input/train.csv', nrows=100000)
df_test = pd.read_csv('../input/test.csv', nrows=100000)

print("Head of train:")
df_train.head(20)





print("Head of test:")
df_test.head(20)




print(df_train.nunique())
df_train.describe()




print(df_test.nunique())
df_test.describe()




app = df_train.app.value_counts(normalize=True)
device = df_train.device.value_counts(normalize=True)
os = df_train.os.value_counts(normalize=True)

plt.figure(figsize=(12,15))

plt.subplot(311)
g1 = sns.barplot(x=app.index[:20], y =app.values[:20])
g1.set_title("App Dist", fontsize=15)
g1.set_xlabel("App ID")
g1.set_ylabel("count normalize", fontsize=12)

plt.subplot(312)
g2 = sns.barplot(x=device.index[:20], y=device.values[:20])
g2.set_title("Device Dist", fontsize=15)
g2.set_xlabel("Device ID")
g2.set_ylabel("count normalize", fontsize=12)

plt.subplot(313)
g3 = sns.barplot(x=os.index[:20], y=os.values[:20])
g3.set_title("OS Dist", fontsize=15)
g3.set_xlabel("OS ID")
g3.set_ylabel("count normalize", fontsize=12)

plt.subplots_adjust(hspace=0.3)
plt.show()

print(app.loc[app.values > 0.05])
print(device[:3])
print(os[:5])




channel = df_train.channel.value_counts(normalize=True)
click_t = df_train.click_time.value_counts(normalize=True)

plt.figure(figsize=(12,10))

plt.subplot(211)
g1 = sns.barplot(x=channel.index[:20], y=channel.values[:20])
g1.set_title("channel dist", fontsize=15)
g1.set_xlabel("channel id")
g1.set_ylabel("count normalize")

plt.subplot(212)
g2 = sns.barplot(x=click_t.index[:20], y=click_t.values[:20])
g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
g2.set_title("click time dist", fontsize=15)
g2.set_xlabel("click time")
g2.set_ylabel("count normalize")

plt.subplots_adjust(hspace=0.3)
plt.show()




df_train['click_time'] = pd.to_datetime(df_train['click_time'])

df_train['click_time_D'] = df_train['click_time'].dt.to_period("H")
print("click_time limit to hour: ")
print(df_train['click_time_D'].unique())
#print(df_train['click_time_D'][:10])
df_train['click_time_m'] = df_train['click_time'].dt.to_period("min")
# print(df_train['click_time_m'][:10])
print("click_time limit to min, size: ", len(df_train['click_time_m'].unique()))

df_test['click_time'] = pd.to_datetime(df_test['click_time'])

df_test['click_time_D'] = df_test['click_time'].dt.to_period("H")
print("click_time limit to hour: ")
print(df_test['click_time_D'].unique())
#print(df_train['click_time_D'][:10])
df_test['click_time_m'] = df_test['click_time'].dt.to_period("min")
# print(df_train['click_time_m'][:10])
print("click_time limit to min, size: ", len(df_test['click_time_m'].unique()))




dw = df_train.loc[df_train['is_attributed'] == 1]
app_dw = dw.app.value_counts(normalize=True)
device_dw = dw.device.value_counts(normalize=True)
os_dw = dw.os.value_counts(normalize=True)

plt.figure(figsize=(12,15))

plt.subplot(311)
g1 = sns.barplot(x=app_dw.index[:20], y =app_dw.values[:20])
g1.set_title("App Dist", fontsize=15)
g1.set_xlabel("App ID")
g1.set_ylabel("count normalize", fontsize=12)

plt.subplot(312)
g2 = sns.barplot(x=device_dw.index[:20], y=device_dw.values[:20])
g2.set_title("Device Dist", fontsize=15)
g2.set_xlabel("Device ID")
g2.set_ylabel("count normalize", fontsize=12)

plt.subplot(313)
g3 = sns.barplot(x=os_dw.index[:20], y=os_dw.values[:20])
g3.set_title("OS Dist", fontsize=15)
g3.set_xlabel("OS ID")
g3.set_ylabel("count normalize", fontsize=12)

plt.subplots_adjust(hspace=0.3)
plt.show()

print(app[:3])
print(app_dw[:3])
print(device[:3])
print(device_dw[:3])
print(os[:3])
print(os_dw[:3])




plt.figure(figsize=(12,20))
plt.subplot(411)
ax1 = sns.distplot(df_train['app'][df_train['is_attributed']==0], color='r', label='no download')
ax2 = sns.distplot(df_train['app'][df_train['is_attributed']==1], color='b', label='download')

plt.subplot(412)
ax1 = sns.distplot(df_train['device'][df_train['is_attributed']==0], color='r', label='no download')
ax2 = sns.distplot(df_train['device'][df_train['is_attributed']==1], color='b', label='download')

plt.subplot(413)
ax1 = sns.distplot(df_train['os'][df_train['is_attributed']==0], color='r', label='no download')
ax2 = sns.distplot(df_train['os'][df_train['is_attributed']==1], color='b', label='download')

plt.subplot(414)
ax1 = sns.distplot(df_train['channel'][df_train['is_attributed']==0], color='r', label='no download')
ax2 = sns.distplot(df_train['channel'][df_train['is_attributed']==1], color='b', label='download')
plt.show()




通过分析可以发现可用的5维数据中，click_time由于数据原因基本是没有贡献度的，剩下的几维数据中，app和device和channel会是相对比较有贡献度的

而ip由于经过了编码，不能做分段处理，贡献度应该也很低。

基本上是一个很少特征的二分类问题，直观上使用xgboost处理。






