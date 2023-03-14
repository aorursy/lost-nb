#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

import category_encoders as ce
import sklearn
from xgboost import XGBRegressor
import xgboost as xgb

import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd


# In[ ]:


#%qtconsole


# In[ ]:


#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)


# In[ ]:


trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")


# In[ ]:


trainData.head()


# In[ ]:


trainData.describe()


# In[ ]:


testData.head()


# In[ ]:


print(trainData.shape)
print(testData.shape)


# In[ ]:


trainData["testflag"]=0
testData["testflag"]=1
fullData = trainData.append(testData)


# In[ ]:



print(fullData.shape)
print(fullData.columns)


# In[ ]:


data = fullData.copy()
data.reset_index(inplace=True)


# In[ ]:





# In[ ]:


data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.datetime.strptime(dateString,"%Y-%m-%d").month)
data["day"] = data.date.apply(lambda dateString : datetime.datetime.strptime(dateString,"%Y-%m-%d").day)


# In[ ]:


featcols = testData.columns.values
featcols = list(set(featcols))
featcols.append("date")
featcols.append("hour")
featcols.append("year") 
featcols.append("weekday")
featcols.append("month")
featcols.append("day")
featcols= list(set(featcols))
featcols.remove("date")
featcols.remove("datetime")
featcols.remove("testflag")
featcols


# In[ ]:


data[data.testflag==1].datetime.describe()


# In[ ]:


data[data.testflag==0].datetime.describe()


# In[ ]:


data.head()


# In[ ]:


from matplotlib import pyplot as pp 
A = data[data.testflag==0].sample(100)
B = data[data.testflag==1].sample(100)


# In[ ]:





# In[ ]:


#pp.plot(data[data.testflag==0].datetime, np.ones(data[data.testflag==0].shape[0]))
f=pp.figure(figsize=(8,8))
ax=f.add_subplot(421)
ax.set_title("year")
pp.plot(np.arange(A.shape[0]),A.year,  ". r")
pp.plot(np.arange(B.shape[0]),B.year, ". b")
ax=f.add_subplot(422)
ax.set_title("month")
pp.plot(np.arange(A.shape[0]),A.month,  ". r")
pp.plot(np.arange(B.shape[0]),B.month, ". b")
ax=f.add_subplot(423)
ax.set_title("day")
pp.plot(np.arange(A.shape[0]),A.day,  ". r")
pp.plot(np.arange(B.shape[0]),B.day, ". b")
ax=f.add_subplot(424)
ax.set_title("weekday")
pp.plot(np.arange(A.shape[0]),A.weekday,  ". r")
pp.plot(np.arange(B.shape[0]),B.weekday, ". b")


# In[ ]:


A.day.describe()


# In[ ]:


B.day.describe()


# In[ ]:


f=pp.figure(figsize=(16,3))
data[data.testflag==0].plot("datetime", "count", marker=".")
f.tight_layout()
f=pp.figure(figsize=(16,3))
a1=f.add_subplot(131)
data[data.testflag==0].loc[:500].plot("datetime", "count", ax=a1)
a2=f.add_subplot(132)
data[data.testflag==0].loc[2000:2500].plot("datetime", "count", ax=a2)
a3=f.add_subplot(133)
data[data.testflag==0].tail(500).plot("datetime", "count", ax=a3)


# In[ ]:


#fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=1)
fig,ax1= plt.subplots(nrows=1)
#fig.set_size(12,20)
hourAggregated = pd.DataFrame(data.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
hourAggregated
import seaborn as sns 
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax1)
#ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')


# In[ ]:


hourAggregatedWd = pd.DataFrame(data.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
hourAggregatedWd


# In[ ]:


fig,ax1= plt.subplots(nrows=1)
sns.pointplot(x=hourAggregatedWd["hour"], y=hourAggregatedWd["count"],hue=hourAggregatedWd["weekday"], data=hourAggregatedWd, join=True,ax=ax1)


# In[ ]:


fig,ax1= plt.subplots(nrows=1)
weekendWd = hourAggregatedWd[hourAggregatedWd["weekday"]>=5]
sns.pointplot(x=weekendWd["hour"], y=weekendWd["count"],hue=weekendWd["weekday"], data=weekendWd, join=True,ax=ax1)


# In[ ]:


hourAggregatedWth = pd.DataFrame(data.groupby(["hour","weather"],sort=True)["count"].mean()).reset_index()
hourAggregatedWth
fig,ax1= plt.subplots(nrows=1)
sns.pointplot(x=hourAggregatedWth["hour"], y=hourAggregatedWth["count"],hue=hourAggregatedWth["weather"], data=hourAggregatedWth, join=True,ax=ax1)


# In[ ]:


monthAggregatedWd = pd.DataFrame(data.groupby(["month","weekday"],sort=True)["count"].mean()).reset_index()
monthAggregatedWd
fig,ax1= plt.subplots(nrows=1)
sns.pointplot(x=monthAggregatedWd["month"], y=monthAggregatedWd["count"],hue=monthAggregatedWd["weekday"], data=monthAggregatedWd, join=True,ax=ax1)


# In[ ]:


monthAggregatedWth = pd.DataFrame(data.groupby(["month","weather"],sort=True)["count"].mean()).reset_index()
fig,ax1= plt.subplots(nrows=1)
sns.pointplot(x=monthAggregatedWth["month"], y=monthAggregatedWth["count"],hue=monthAggregatedWth["weather"], data=monthAggregatedWth, join=True,ax=ax1)

# for  plots see https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile

# @Joana: nice variable plots: https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
# please insert!

# xgboost :https://www.kaggle.com/miteshyadav/comprehensive-eda-with-xgboost-top-10-percentile


# In[ ]:


from matplotlib import pyplot as pp
f=pp.figure(figsize=(12,6))
ax1=f.add_subplot(221)
data["count"].hist(bins=101, ax=ax1)
pp.title("Count, lin hist")
ax2=f.add_subplot(222)
pp.hist(data["count"].dropna(), bins=101, log=True)
pp.title("Count, semilogy hist")
ax3=f.add_subplot(223)
l1 = np.log(data["count"].dropna().values)
pp.hist(l1, bins=101, log=False)
pp.title("Count, semilogy hist")


# In[ ]:





# In[ ]:


import seaborn as sns
f = pp.figure(figsize=(16,4))
a1 = f.add_subplot(131)
sns.boxplot(data=data,y="count",x="month")
a2 = f.add_subplot(132)
sns.boxplot(data=data,y="count",x="weekday")
a3 = f.add_subplot(133)
sns.boxplot(data=data,y="count",x="hour")


# In[ ]:


data.weather.unique()


# In[ ]:


# Thanks to Flavia

data["weekday2"] = np.cos(data["weekday"]/6. *(np.pi) ) #runs originally from 0..6
data["month2"] = np.cos((data["month"]-1.)/11. *(np.pi) ) # runs or. from 1..12
data["hour2"] = np.cos((data["hour"])/23. *(np.pi) ) # runs or. from 1..12

featcols.append("weekday2")
featcols.append("month2")
featcols.append("hour

desc = data.describe(include="all")
for c in featcols+["count",]:
    desc.loc["nnan", c] = data[pd.isnull(data[c])][c].shape[0]
    desc.loc["unique", c] = data[c].unique().shape[0]
    desc.loc["is_feature", c] = 1
    desc.loc["is_numeric", c] = 1
    #print(c, data[pd.isnull(data[c])][c].shape[0])
c="count"
desc.loc["is_feature", c] = 0
# cat features:
for c in( "holiday", "season","weather", "workingday", "year" ):
    desc.loc["is_numeric", c] = 0
    data[c] = data[c].astype("category")
#desc.loc["is_numeric", ["date", "datetime"]] = 0
#desc.loc["is_feature", ["date", "datetime"]] = 0
# special: sine "weekday", "month", "year" has only 2 values, so cat.

desc


# In[ ]:


X_cat=[]
feat_num = desc.T[(desc.loc["is_numeric"]==1) & (desc.loc["is_feature"]==1)].T.columns.values

feat_cat = desc.T[desc.loc["is_numeric"]==0 & (desc.loc["is_feature"]==1) ].T.columns.values

import category_encoders as ce

for i,c in enumerate(feat_cat):
    #ci = ce.OneHotEncoder()
    ci = ce.OneHotEncoder(cols=[c], impute_missing=False)
    if i==0:
        X_cat = ci.fit_transform(data[[c]])
    else:
        X_cat = pd.concat([X_cat , ci.fit_transform(data[[c]])], axis=1)
X_cat


# In[ ]:


X_num = data[feat_num]
X_num


# In[ ]:


X = pd.concat([X_num, X_cat], axis=1)


# In[ ]:


X.shape


# In[ ]:


y = np.log(data["count"])
#y = data["count"]
y


# In[ ]:


X_train  = X[data.testflag==0].values
y_train = y[data.testflag==0]


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


X_test  = X[data.testflag==1].values
y_test = y[data.testflag==1]


# In[ ]:


print(X_test.shape)
print(y_test.shape)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 2000 decision trees
rf = RandomForestRegressor(n_estimators = 2000, random_state = 42)

# Train the model on training data
rf.fit(X_train, y_train);


# In[ ]:


# # We only have labeled train data (so far)

predictions = np.exp(rf.predict(X_train))

# Calculate the absolute errors
errors = np.abs(predictions - np.exp(y_train))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', np.mean(errors) )


# In[ ]:


submission_y = np.exp(rf.predict(X_test))


# In[ ]:


submission_y


# In[ ]:


submission= data[data["testflag"]==1][["datetime"]].copy()
submission["count"] = submission_y

#submission.head(10)


# In[ ]:


submission.to_csv("submission_04.csv.gz", index=False, sep=",", compression="gzip") # 0.43


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
gbm.fit(X_train,y_train)


# In[ ]:


# We only have labeled train data (so far)
predictions = np.exp(gbm.predict(X_train))

# Calculate the absolute errors
errors = np.abs(predictions - np.exp(y_train))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', np.mean(errors) )


# need to create CV set


# In[ ]:




