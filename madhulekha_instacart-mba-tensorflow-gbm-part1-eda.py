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




import matplotlib.pyplot as plt
import seaborn as sns
import warnings

color = sns.color_palette()
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:,.2f}'.format
plt.rcParams.update({'font.size': 18})




##changing directory
os.chdir('../input/')




###  INVENTORY TERMS

#### AISLES
all_aisles = pd.read_csv('aisles.csv') #134
all_aisles.head(5)
all_aisles.describe(include = 'all')




###  INVENTORY TERMS

#### DEPARTMENTS
all_depts = pd.read_csv('departments.csv') #21
all_depts.head(5)
all_depts.describe(include = 'all')




###  INVENTORY TERMS

#### PRODUCTS
all_products = pd.read_csv('products.csv') #49688
all_products.head(5)
all_products.describe(include = 'all')
##134 aisle_ids, 21 dept_ids, 49688 products, ids - All descriptions are unique




###  ORDER TERMS
all_orders = pd.read_csv('orders.csv')
print(all_orders.shape) # (3421083, 7)
all_orders.head(5)

print("No. of first orders : ", all_orders.days_since_prior_order.isnull().sum())
print("No. of users : ",all_orders.user_id.nunique())
#Orders of all users from their first purchase is given, max ? - dig down to details on eda of orders

all_orders.describe()




### PRIOR ORDERS
prior_orders = pd.read_csv('order_products__prior.csv')
print(prior_orders.shape)
prior_orders.head(5)




### TRAINING ORDERS
train_set = pd.read_csv('order_products__train.csv')
print(train_set.shape)
train_set.head(5)




sample = pd.read_csv('sample_submission.csv')
sample.head()




all_orders.head(5)




all_orders.eval_set.value_counts()




all_orders[['user_id','eval_set']].groupby('eval_set').nunique('user_id')['user_id']




set(all_orders.loc[all_orders['eval_set'] == 'train','user_id']) & set(all_orders.loc[all_orders['eval_set'] == 'test','user_id'])




t = all_orders[['user_id','order_id']].groupby('user_id').nunique('order_id')['order_id']
print(t.max())
print(t.min())




plt.figure(figsize=(12,8))
sns.distplot(t,norm_hist=False,kde=False,axlabel='No. of orders')
plt.xticks(range(min(t), max(t)+1, 3))
plt.show()




t1 = all_orders.loc[all_orders['eval_set']=='prior',['user_id','order_id']].groupby('user_id').nunique('order_id')['order_id']
print(t1.max())
print(t1.min())
print("No. of users in prior with maximum number of orders :" + str(sum(t1==t1.max())))
print("No. of users in prior with minimum number of orders :" + str(sum(t1==t1.min())))




all_orders.head(5)




plt.figure(figsize=(12,5))
sns.countplot(x="order_dow", data=all_orders,palette = sns.color_palette("ch:2.5,-.2,dark=.3"))




plt.figure(figsize=(12,4))
sns.countplot(x="order_hour_of_day", data=all_orders)




sns.catplot(x='order_hour_of_day',col="order_dow", data=all_orders,kind="count")




t = all_orders.groupby(['order_hour_of_day',"order_dow"])["order_id"].count().reset_index()
x = t.pivot("order_dow","order_hour_of_day","order_id")
plt.figure(figsize=(12,6))
sns.heatmap(x,cmap="YlGnBu")




plt.figure(figsize=(12,4))
sns.countplot(x="days_since_prior_order", data=all_orders)




first_orders = all_orders[all_orders.days_since_prior_order != all_orders.days_since_prior_order]
print(first_orders.shape)
print(first_orders.order_number.unique())
print(first_orders.eval_set.unique())
first_orders.head()




print(prior_orders.columns) # Just for reference




t = prior_orders.groupby('order_id')['product_id'].count().reset_index()
plt.figure(figsize=(20,6))
ax = sns.countplot(x='product_id',data=t)
ax.set(xlabel='No. of products per order - prior',ylabel='count of orders')
plt.show()
print(t['product_id'].max())




t = train_set.groupby('order_id')['product_id'].count().reset_index()
plt.figure(figsize=(20,6))
ax = sns.countplot(x='product_id',data=t)
ax.set(xlabel='No. of products per order - train',ylabel='count of orders')
plt.show()
print(t['product_id'].max())




fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,6))
sns.countplot(x= 'reordered',data = prior_orders, ax=ax[0],palette = sns.color_palette("Set2"))
sns.countplot(x= 'reordered',data = train_set, ax=ax[1],palette = sns.color_palette("Set2"))
print("Reordered distribution in prior orders : \n",prior_orders.reordered.value_counts(normalize=True))
print("Reordered distribution in train orders : \n",train_set.reordered.value_counts(normalize=True))




t = prior_orders.groupby('order_id')['reordered'].sum().reset_index()
plt.figure(figsize=(20,6))
#plt.xlabel('No. of re-ordered products per order')
#plt.ylabel('count of orders')
ax = sns.countplot(x='reordered',data=t)
ax.set(xlabel='No. of re-ordered products per order', ylabel='count of orders')
plt.show()
print("No. of orders with no reordered products in prior: ", t.loc[t['reordered'] == 0,'order_id'].size)
print("% of orders with no reordered products in prior: ",(t.loc[t['reordered'] == 0,'order_id'].size*100)/t['order_id'].size)




t = train_set.groupby('order_id')['reordered'].sum().reset_index()
print("No. of orders with no reordered products in train: ",t.loc[t['reordered'] == 0,'order_id'].size)
print("% of orders with no reordered products in train: ",(t.loc[t['reordered'] == 0,'order_id'].size*100)/t['order_id'].size)




prior_orders_extended = prior_orders.merge(all_products[['product_id','aisle_id','department_id']], on='product_id', how='left').    merge(all_orders,on='order_id',how='left')
#    merge(all_aisles, on='aisle_id', how='left').\ # Removing as data too heavy \
#    merge(all_depts, on='department_id', how='left').\ # Removing as data too heavy \

prior_orders_extended.head()




tab = pd.crosstab(prior_orders_extended['department_id'],prior_orders_extended['reordered'],values=prior_orders_extended['order_id'],aggfunc='count')
 # Mapping the names of dept here
tab = tab.merge(all_depts,how='left',left_index=True,right_on='department_id').drop('department_id',axis=1).set_index('department')
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,6))
tab.sum(axis=1).plot(kind='bar',ax=ax[0])
tab_prop.plot(kind="bar", stacked=True, ax = ax[1],color = ['lightcoral','lightgreen'] )




tab = pd.crosstab(prior_orders_extended['department_id'],prior_orders_extended['reordered'],values=prior_orders_extended['order_id'],aggfunc='nunique')
 # Mapping the names of dept here
tab = tab.merge(all_depts,how='left',left_index=True,right_on='department_id').drop('department_id',axis=1).set_index('department')
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,6))
tab.sum(axis=1).plot(kind='bar',ax=ax[0])
tab_prop.plot(kind="bar", stacked=True, ax = ax[1],color = ['lightcoral','lightgreen'] )




tab = pd.crosstab(prior_orders_extended['aisle_id'],prior_orders_extended['reordered'],values=prior_orders_extended['order_id'],aggfunc='count')
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,6))
tab.sum(axis=1).plot(kind='bar',ax=ax[0])
tab_prop.plot(kind="bar", stacked=True, ax = ax[1],color = ['lightcoral','lightgreen'] )




prod_repeatability = prior_orders_extended.groupby('product_id').agg(    {'add_to_cart_order':'mean','reordered':['count','sum']})
prod_repeatability.columns = prod_repeatability.columns.map('_'.join)
prod_repeatability = prod_repeatability.reset_index().rename(columns=    {'add_to_cart_order_mean':'avg_cart_position','reordered_count':'Total_Purchases','reordered_sum':'Repeat_Purchases'})
prod_repeatability['%_repeated'] = prod_repeatability['Repeat_Purchases']/prod_repeatability['Total_Purchases']




plt.figure(figsize=(20,6))
ax = sns.distplot(prod_repeatability['%_repeated'],kde=False)
ax.set(xlabel='Repeatability of product', ylabel='Number of products')
plt.show()




prod_cart_pos = prior_orders_extended.groupby('add_to_cart_order').agg({'reordered':['count','sum']}).reset_index()
prod_cart_pos.columns = prod_cart_pos.columns.map('_'.join)
prod_cart_pos = prod_cart_pos.reset_index().rename(columns=    {'reordered_count':'Total_Purchases','reordered_sum':'Repeat_Purchases'})
prod_cart_pos['%_repeated'] = prod_cart_pos['Repeat_Purchases']/prod_cart_pos['Total_Purchases']




prod_cart_pos.head()




fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(20,10))
sns.lineplot(data =prod_cart_pos,x='add_to_cart_order_' ,y='%_repeated',ax=ax[0] )
ax[0].set(ylabel='Reorder percentage')
sns.barplot(data =prod_cart_pos,x='add_to_cart_order_' ,y='Total_Purchases',ax=ax[1])
ax[1].set(xlabel='Position of product in cart', ylabel='Number of orders')
plt.show()




product_frequency = prior_orders_extended.groupby('product_id').agg({'order_id':'count','reordered':'sum'})
product_frequency = product_frequency.reset_index().rename(columns={'order_id':'Total_Orders','reordered':'No_reorders'})
product_frequency = product_frequency.merge(all_products,how='left',on='product_id').    merge(all_aisles,how='left',on='aisle_id').    merge(all_depts,how='left',on='department_id')
product_frequency['reorder_%'] = product_frequency['No_reorders']/product_frequency['Total_Orders']




product_frequency.sort_values(by=['Total_Orders','reorder_%'],ascending=False).head(15)




from wordcloud import WordCloud
wordcloud = WordCloud(width=600, height=600,background_color="skyblue",colormap="Spectral").generate(' '.join(product_frequency['product_name']))
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




aisle_frequency = product_frequency.groupby(['department_id','aisle_id'])['Total_Orders','No_reorders'].sum().reset_index()
aisle_frequency = aisle_frequency.merge(all_aisles,how='left',on='aisle_id').    merge(all_depts,how='left',on='department_id')
aisle_frequency['reorder_%'] = aisle_frequency['No_reorders']/aisle_frequency['Total_Orders']




aisle_frequency.sort_values(by=['Total_Orders','reorder_%'],ascending=False).head(10)




department_frequency = product_frequency.groupby(['department_id'])['Total_Orders','No_reorders'].sum().reset_index()
department_frequency = department_frequency.merge(all_depts,how='left',on='department_id')




import squarify
plt.figure(figsize=(15,15))
squarify.plot(sizes=department_frequency['Total_Orders'], label=department_frequency['department'], alpha=.7 )
plt.axis('off')
plt.show()

