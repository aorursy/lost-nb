#!/usr/bin/env python
# coding: utf-8

# In[1]:


Exploratory Data Analysis 


# In[2]:





# In[2]:





# In[2]:


#Import the required packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk #Ml models
import seaborn as sns #visualization
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

#Let's check the avilable files for this problem

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


#Load the required files using read_csv 
Aisles=pd.read_csv('../input/aisles.csv')
Departments=pd.read_csv('../input/departments.csv')
OrderProductsPrior=pd.read_csv('../input/order_products__prior.csv')
OrderProductsTrain=pd.read_csv('../input/order_products__train.csv')
Orders=pd.read_csv('../input/orders.csv')
Products=pd.read_csv('../input/products.csv')


# In[4]:


#View sample aisles data
Aisles.sample(n=5)


# In[5]:


#View sample department data
Departments.sample(n=5)


# In[6]:


#View sample products data
Products.sample(n=5)


# In[7]:


#View sample OrdersProductPrior data
OrderProductsPrior.sample(n=5)


# In[8]:


#View sample OrdersProductTrain data
OrderProductsTrain.sample(n=5)


# In[9]:


#View sample Orders data
Orders.sample(n=5)


# In[10]:


#Get the total count of orders based on eval_set
Orders.groupby('eval_set').size().sort_values(ascending=False)


# In[11]:


#create data frame for the test set
OrderProductTest=Orders[Orders['eval_set']=='test']


# In[12]:


#view sample test set
OrderProductTest.sample(n=5)


# In[13]:


#Dig more into Orders_prior
OrderProductsPrior.sample(n=10)


# In[14]:


pro=OrderProductsPrior.groupby('product_id')['reordered'].aggregate({'tor':'count','tos':sum}).reset_index()
pro=pd.merge(pro,Products,on=['product_id'])
pro=pd.merge(pro,Departments,on=['department_id'])
pro=pd.merge(pro,Aisles,on=['aisle_id'])


# In[15]:


req=pro.sort_values(by='tor',ascending=False).head(10)
req[['product_id','tor','tos','product_name','department','aisle']]


# In[16]:




