#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# In[2]:


store = pd.read_csv("../input/store.csv")
train = pd.read_csv("../input/train.csv",index_col = "Date",parse_dates = ['Date'],low_memory=False)
test = pd.read_csv("../input/test.csv",index_col = "Date",parse_dates = ['Date'],low_memory=False)
sample_submission = pd.read_csv("../input/sample_submission.csv")


# In[3]:


train.sample(5)


# In[4]:


store.sample(5)


# In[5]:


store.index = store.Store


# In[6]:


print ("Shape of data set is ",train.shape)


# In[7]:


train.head(5).sort_values('Date')


# In[8]:


train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear
train['DayOfWeek'] = train.index.dayofweek
train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].fillna(0)
train['SalePerCustomer'].describe()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
year_records = train['Year'].value_counts()

sns.barplot(x=year_records.index,y=year_records.values)
plt.show()


# In[10]:


#train = train.loc[ (train.Year == 2013) | (train.Year == 2014)]
#test = train.loc[ (train.Year == 2015)]


# In[11]:


store[( store.Store == 322 )]


# In[12]:


store.sample(5)


# In[13]:


store_id = store.sample(1).Store.values
print ("Store id is ",store_id[0])

store_data = train[train['Store'] == store_id[0]]
print ("Number of entries of store ",store_id[0]," is ",store_data.shape[0])


# In[14]:


plt.figure(figsize=(20,6))
plt.plot(store_data[store_data.Open == 1].Sales)
plt.title("Sales for store id {} excluding store closed days".format(store_id[0]))
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = seasonal_decompose(store_data[store_data.Open == 1].Sales, model='multiplicative',freq=365)
plt.figure(figsize=(20,6))
fig = decomposition.plot()
plt.show()


# In[16]:


sale_per_customer = train[train.Open == 1].groupby('StateHoliday')['SalePerCustomer'].sum() /                     train[train.Open == 1].groupby('StateHoliday')['SalePerCustomer'].count()
print (sale_per_customer)
plt.figure(figsize=(5,3))
sns.barplot(x=sale_per_customer.index,y=sale_per_customer.values)
plt.ylabel('Sale Per Customer')
plt.show()


# In[17]:


data_by_day_of_week = train[train.Open == 1].groupby('DayOfWeek').mean()


# In[18]:


mean_customer_count = data_by_day_of_week.Customers
mean_sales_value = data_by_day_of_week.Sales
mean_sale_per_customer = data_by_day_of_week.SalePerCustomer
promo = data_by_day_of_week.Promo

mean_sales_value = mean_sales_value.sort_values(ascending=True)
mean_customer_count = mean_customer_count.sort_values(ascending=True)
mean_sale_per_customer = mean_sale_per_customer.sort_values(ascending=True)
promo = promo.sort_values(ascending=True)

fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(20,11))
sns.barplot(x=mean_customer_count.index, y=mean_customer_count, ax=axs[0][0])
sns.barplot(x=mean_sales_value.index, y=mean_sales_value, ax=axs[0][1])
sns.barplot(x=mean_sale_per_customer.index, y=mean_sale_per_customer, ax=axs[1][0])
sns.barplot(x=promo.index, y=promo, ax=axs[1][1])

axs[0][0].set_title('Customer Count')
axs[0][1].set_title('Sales Value')
axs[1][0].set_title('Average Sales Per Customer')
axs[1][1].set_title('Promo')

axs[0][0].set_xlabel('')
axs[0][1].set_xlabel('')
axs[1][0].set_xlabel('')
axs[1][1].set_xlabel('')

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
axs[0][0].set_xticklabels(days,rotation = 10)
axs[0][1].set_xticklabels(days,rotation = 10)
axs[1][0].set_xticklabels(days,rotation = 10)
axs[1][1].set_xticklabels(days,rotation = 10)

plt.show()


# In[19]:


sales_by_store = train[['Sales','Store','Open']]
total_sales_by_store = sales_by_store.groupby('Store').sum()
total_sales_by_store['SalePerDay'] = total_sales_by_store['Sales']/total_sales_by_store['Open']
total_sales_by_store.sort_values(['SalePerDay'],inplace=True)

plt.figure(figsize=(20,5))
sns.boxplot(x=total_sales_by_store['SalePerDay'])
plt.title('Box Plot of stores sale per day')
plt.xlabel('Per Day Sales')
plt.show()


# In[20]:


#Making of top 3 and bottom 3
total_sales_by_store.sort_values(['SalePerDay'],inplace=True,ascending=True)
bottom_3 = total_sales_by_store[0:3]
total_sales_by_store.sort_values(['SalePerDay'],inplace=True,ascending=False)
top_3 = total_sales_by_store[0:3]
frames = [top_3, bottom_3]
top_3_bottom_3 = pd.concat(frames)

top_3_bottom_3['Store'] = top_3_bottom_3.index
top_3_bottom_3 = top_3_bottom_3.sort_values(['SalePerDay']).reset_index(drop=True)

#Plotting bar plot of top 3 and bottom 3
plt.figure(figsize=(8,6))
ax = sns.barplot(top_3_bottom_3.index, top_3_bottom_3.SalePerDay)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Store", ylabel='SalePerDay')
# adding proper x labels
ax.set_xticklabels(top_3_bottom_3.Store)
for item in ax.get_xticklabels(): 
    item.set_rotation(0)
for i, v in enumerate(top_3_bottom_3["SalePerDay"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(round(v[1],2)), color='m', va ='bottom', rotation=45)
plt.tight_layout()
plt.show()


# In[21]:


store_and_competitor_distance = store.loc[top_3_bottom_3.Store]
store_and_competitor_distance = store_and_competitor_distance[['Store','CompetitionDistance']]

store_and_competitor_distance['Store'] = store_and_competitor_distance.index
store_and_competitor_distance = store_and_competitor_distance.sort_values(['CompetitionDistance']).reset_index(drop=True)

#Plotting bar plot of top 3 and bottom 3
plt.figure(figsize=(8,6))
ax = sns.barplot(store_and_competitor_distance.index, store_and_competitor_distance.CompetitionDistance)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Store", ylabel='CompetitionDistance')
# adding proper x labels
ax.set_xticklabels(store_and_competitor_distance.Store)
for item in ax.get_xticklabels(): 
    item.set_rotation(0)
for i, v in enumerate(store_and_competitor_distance["CompetitionDistance"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(round(v[1],2)), color='m', va ='bottom', rotation=45)
plt.tight_layout()
plt.show()

Store 198 has the competitor at the farthest distance and it also has the third least sale per day.

Store 543 has got the competitor at the closest distance ( 250 ) and it has the second least sale  per day.

Store 307 has third least competitor distance and it has the least sale per day.

Store 817 which has highest sale per day also has competitor at second closes distance.

Let us make a correlation plot between competitor distance and daily sales to understand this a bit further.
# In[22]:


total_sales_by_store['Store'] = total_sales_by_store.index
total_sales_by_store = total_sales_by_store[["SalePerDay","Store"]]
store_and_competitor_distance = store[["Store","CompetitionDistance"]]
joined_df = pd.merge(store_and_competitor_distance, total_sales_by_store, on='Store', how='inner')
sns.jointplot(x="SalePerDay",y="CompetitionDistance",data=joined_df , kind="reg")
plt.show()


# In[23]:


corr = joined_df['SalePerDay'].corr(joined_df['CompetitionDistance'])
print ("Correlation between sale per day and competition distance is ",corr)

