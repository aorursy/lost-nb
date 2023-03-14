#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# import zipfile
# with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/order_products__prior.csv.zip","r") as zipf:
# zipf.extractall(".")


# In[3]:


#  Pick a Dataset you might be interested in.
#  Say, all airline-safety files...
prior = "order_products__prior.csv"
train = "order_products__train.csv"
orders = "orders.csv"
products = "products.csv"
aisles = "aisles.csv"
depart = "departments.csv"


# In[4]:


import zipfile
# Will unzip the files so that you can see them..
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+prior+".zip","r") as z:
    z.extractall(".")
    
from subprocess import check_output
print(check_output(["ls", prior]).decode("utf8"))

from pandas import DataFrame as df
prior = pd.read_csv("order_products__prior.csv")
prior.head()


# In[5]:


with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+train+".zip","r") as z:
    z.extractall(".")
train = pd.read_csv("order_products__train.csv")


# In[6]:


with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+orders+".zip","r") as z:
    z.extractall(".")
orders = pd.read_csv("orders.csv")


# In[7]:


with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+products+".zip","r") as z:
    z.extractall(".")
products = pd.read_csv("products.csv")


# In[8]:


with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+aisles+".zip","r") as z:
    z.extractall(".")
aisles = pd.read_csv("aisles.csv")


# In[9]:


with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+depart+".zip","r") as z:
    z.extractall(".")
depart = pd.read_csv("departments.csv")


# In[10]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[11]:


np.max(orders['user_id']) 
#order_id from orders = 3421083
#order_id from train.csv= 3421070
#order_id from prior.csv = 3421083


# In[12]:


dow=list(set(orders['order_dow']))
print(dow)
# dow : 요일. 0 - 일요일


# In[13]:


oid_prior=list(set(prior['order_id']))
print(len(oid_prior))
oid_train=list(set(train['order_id']))
print(len(oid_train))
oid=list(set(orders['order_id']))
print(len(oid))

# 131209 + 3214874 =/= 3421083 (75,000차이) -> sample_submission 행 수와 일치 : test set


# In[14]:


len(orders)


# In[15]:


print(100*((len(oid)-len(oid_train)-len(oid_prior))/len(oid)),'%')


# In[16]:


for col in orders.columns:
    null_or = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100* (orders[col].isnull().sum() / orders[col].shape[0]))
    print(null_or)
    
    # orders['days_since_prior_order']에만 6.03% NaN


# In[17]:


for col in products.columns:
    null = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (products[col].isnull().sum() / products[col].shape[0]))
    print(null)


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5) 
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

msno.matrix(df=orders.iloc[:, :], figsize=(6, 6), color=(0.8, 0.5, 0.2))


# In[19]:


msno.bar(df=orders.iloc[:, :], figsize=(6, 6), color=(0.8, 0.5, 0.2))


# In[20]:


f, ax = plt.subplots(1, 2, figsize=(15, 5))

prior['reordered'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - reordered')
ax[0].set_ylabel('')
sns.countplot('reordered', data=prior, ax=ax[1])
ax[1].set_title('Count plot - reordered')

plt.show()


# In[21]:


f, ax = plt.subplots(1, 2, figsize=(10, 5))

train['reordered'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - reordered')
ax[0].set_ylabel('')
sns.countplot('reordered', data=prior, ax=ax[1])
ax[1].set_title('Count plot - reordered')

plt.show()


# In[22]:


print(len(products))


# In[23]:


# sub = pd.read_csv('/kaggle/input/instacart-market-basket-analysis/sample_submission.csv')
# sub.head()

sub = "sample_submission.csv"
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+sub+".zip","r") as z:
    z.extractall(".")
sub = pd.read_csv("sample_submission.csv")
sub.head()


# In[24]:


o2 = orders.set_index("user_id")
o2.head()


# In[25]:


o_dict = dict(list(orders[:][['order_id', 'eval_set', 'user_id' , 'order_number', 'order_dow' , 'order_hour_of_day' , 'days_since_prior_order']].groupby('user_id')))


# In[26]:


o_dict[1]


# In[27]:


o_num = list(orders.groupby('user_id').size())
# print(o_num)
np.max(o_num)


# In[28]:


from collections import Counter

result = Counter(o_num)
print(result)


# In[29]:


for k in result.keys():
    print(k,result[k])
print()


# In[30]:


se_result = pd.Series(result, name='count')


# In[31]:


se_result = se_result.sort_values(ascending=False)


# In[32]:


se_result.index.name='o_num'
# se_result.reset_index()


# In[33]:


df_result = pd.DataFrame(se_result)
df_result.reset_index(level=['o_num'], inplace = True)
df_result = df_result.sort_values(by=['o_num'], ascending=True)
df_result.tail()


# In[34]:


## from matplotlib import pyplot as plt
plt.style.use('seaborn')
sns.set(font_scale=2.5)

get_ipython().run_line_magic('matplotlib', 'inline')
 
# y = se_result[:]
# x = range(len(y))
# plt.bar(x, y, width = 0.8, color="blue")
# plt.show()


# In[35]:


sns.set(font_scale=0.9)
plt.figure(figsize=(30,10)) # 위치가 sns.plot 위에 있어야만 작동하나..?

ax = sns.barplot(data=df_result, x="o_num", y="count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
plt.title("customer number per order_num")
plt.show()


# In[36]:


# sns.distplot(df_result["count"], hist=False)
# plt.show()

sns.set(font_scale=1.0)
plt.figure(figsize=(15,8)) # 위치가 sns.plot 위에 있어야만 작동하나..?
sns.countplot(x="days_since_prior_order", data=orders)
plt.show()


# In[37]:


sns.countplot(x="order_hour_of_day", data=orders)
plt.show()


# In[38]:


# print(len(train_copy))
# print(train_copy['reordered'][1:2]==0)
# (train_copy['reordered'][0:1]==0).bool()==True
# train_copy.loc[1:2]


# In[39]:


train_copy = train


# In[40]:


train_copy = train_copy.sort_values(['reordered'], ascending=[True])


# In[41]:


tr_re = train_copy[int(555793):]  #tr_re = reordered products in train set
tr_re.head()


# In[42]:


tr_nore = train_copy[:int(555793)]
tr_nore.tail()


# In[43]:


from collections import Counter as cc

pro_name = cc(tr_re['product_id'])
print(pro_name.most_common()[:10])


# In[44]:


pro_name2 = cc(tr_nore['product_id'])
print(pro_name2.most_common()[:10])


# In[45]:


pro_name3 = cc(train['product_id'])
print(pro_name3.most_common()[:10])


# In[46]:


prior_copy = prior


# In[47]:


prior_copy = prior_copy.sort_values(['reordered'], ascending=[True])


# In[48]:


# pr_re = prior_copy  #pr_re = reordered products in prior set
# prior_copy[127000:127100]
# print(prior_copy[16434489:17000000])


# In[49]:


order_product_count = prior.groupby('order_id').count()[['product_id']]
order_product_count.columns = ['product_count']


# In[50]:


orders2 = orders.merge(order_product_count, left_on='order_id', right_index=True)


# In[51]:


index_day = "Sun Mon Tue Wen Thu Fri Sat".split()


# In[52]:


def drawWeekHour(ds, values,  aggfunc=len, title=None, figsize=(18,5) , cmap=None):
    weekhour_ds = ds.pivot_table(index='order_dow', columns='order_hour_of_day', values=values, aggfunc=aggfunc).fillna(0)
    weekhour_ds.index =  [index_day[index] for index in weekhour_ds.index]
    
    plt.figure(figsize=figsize)
    f = sns.heatmap(weekhour_ds, annot=True, fmt="1.1f", linewidths=.5, cmap=cmap) 
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    if title:
        plt.title(title, fontsize=15)


# In[53]:


# 시간과 요일간에 재주문이 걸리는 시간에 관한 상관관계
drawWeekHour(orders, values='days_since_prior_order',aggfunc=lambda x: np.mean(x), title="prior orders", cmap='YlGn')


# In[54]:


# 주문 횟수가 많아질 수록 reorder까지의 날짜가 짧아진다.
sns.set(style="whitegrid", palette="colorblind", font_scale=1.5)

orders2.groupby('order_number').agg({'days_since_prior_order':np.mean, 'product_count':np.mean})    .plot(figsize=(16,6), title="order_number, prior_order", marker='o' )
plt.ylabel('days since prior order ')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




