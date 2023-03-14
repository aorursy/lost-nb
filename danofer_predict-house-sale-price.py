#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


house_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

#preview data
house_df.head()


# In[3]:


house_df.info()
print("------------------")
test_df.info()


# In[4]:


# fill missing values
print(house_df["full_sq"].mode())
print(house_df["full_sq"].isnull().values.any())
print(house_df["full_sq"].isnull().sum())
print("------------------")
#house_df["full_sq"] = house_df["full_sq"].fillna()
# find the price-segment with max number of houses
figure, ax = plt.subplots(2,2,figsize=(12,10)) # 2 rows, 2 columns
min_price = house_df["price_doc"].min()
max_price = house_df["price_doc"].max()
avg_price = house_df["price_doc"].mean()

min_area = house_df["full_sq"].min()
max_area = house_df["full_sq"].max()
avg_area = house_df["full_sq"].mean()
print(min_price, max_price, avg_price)
print(min_area, max_area, avg_area)
print("------------------")
# find rows given a column value
# print(house_df.loc[house_df["full_sq"] == 0]['id'])
ax[0,0].scatter(house_df["full_sq"], house_df["price_doc"])
ax[0,0].set(title = "scatter plot below mean", xlabel = "area", ylabel = "price",xlim = [0, 50], ylim = [min_price, 0.3*max_price] )
ax[0,1].scatter(house_df["full_sq"], house_df["price_doc"])
ax[0,1].set(title = "scatter plot below mean", xlabel = "area", ylabel = "price",xlim = [50, 100], ylim = [min_price, max_price/2] )
ax[1,0].scatter(house_df["full_sq"], house_df["price_doc"])
ax[1,0].set(title = "scatter plot below mean", xlabel = "area", ylabel = "price",xlim = [100, 150], ylim = [min_price, 0.7*max_price] )
ax[1,1].scatter(house_df["full_sq"], house_df["price_doc"])
ax[1,1].set(title = "scatter plot below mean", xlabel = "area", ylabel = "price",xlim = [150, 200], ylim = [min_price, 0.9*max_price] )

#ax[1].scatter(house_df["full_sq"], house_df["price_doc"])
#ax[1].set(title = "scatter plot above mean", xlabel = "area", ylabel = "price", xlim = [0, 300], ylim = [avg_price, max_price + avg_price])


# In[5]:


# model
LinearRegression.fit(house_df["full_sq"],house_df["price_doc"])
RMSQE = r2_score.(test_df["price_doc"]predict(test_df["full_sq"])

