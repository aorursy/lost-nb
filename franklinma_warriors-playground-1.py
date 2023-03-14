#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


len(df['id'].unique())


# In[6]:


id_number = df['id'].unique()


# In[7]:


plt.plot(id_number)


# In[8]:


id_number[800:900]


# In[9]:


lenID = {}
for id_no in id_number:
    lenID[id_no] = len(df[df['id'] == id_no])


# In[10]:


lenID


# In[11]:


print(id_10['y'].std(axis=0))
plt.plot(id_10['y'])
plt.show()


# In[12]:


plt.hist(id_10['y'])


# In[13]:


id_11 = df[df['id'] == 11]
print(id_11['y'].std(axis=0))
plt.plot(id_11['y'])
plt.show()
id_11.shape


# In[14]:


plt.hist(id_11['y'])


# In[15]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.cross_validation import cross_val_score
import matplotlib


# In[16]:


#id_10.head()
x_train_id_10 = id_10.loc[:, "derived_0":"technical_44"]
x_train_id_10.head()


# In[17]:


id_10["technical"]


# In[18]:


adsf


# In[19]:


model_lasso = LassoCV(0.5).fit(x_train, y_target)


# In[20]:


mean_values = x_train.mean(axis=0)
x_train.fillna(mean_values, inplace=True).head()


# In[21]:


y_mean = df['y'].mean(axis=0)
y_target = df['y']
y_target.fillna(y_mean, inplace=True)


# In[22]:


model_lasso = LassoCV(0.5).fit(x_train, y_target)


# In[23]:


coef = pd.Series(model_lasso.coef_, index = x_train.columns)


# In[24]:


imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


# In[25]:


get_ipython().run_line_magic('pinfo', 'Lasso')


# In[26]:





# In[26]:




