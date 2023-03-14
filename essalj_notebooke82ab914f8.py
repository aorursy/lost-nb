#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



# In[2]:




# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()


# In[3]:


# happy customers have TARGET==0, unhappy custormers have TARGET==1
# A little less then 4% are unhappy => unbalanced dataset
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df


# In[4]:


#ASSUMING VAR3 = NATIONALITY
train.var3.value_counts()[:10]

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape


# In[5]:


#Add feature that counts the number of zeros in a row
X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']


# In[6]:


# num_var4 : number of bank products
# According to dmi3kno (see https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
# num_var4 is the number of products. Let's plot the distribution:
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()


# In[7]:


# Let's look at the density of the of happy/unhappy customers in function of the number of bank products
sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var4")    .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()


# In[8]:


train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');


# In[9]:





# In[9]:


print(train.var38.describe())

# How is var38 looking when customer is unhappy ?
print(train.loc[train['TARGET']==1, 'var38'].describe())



#train.var38.value_counts() #spike on 117.310,97799016
# what if we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()

#train.var38.hist(bins=1000)
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);



# In[10]:



train['TARGET'].describe()
sns.pairplot(data=train[['num_var4','TARGET','var38']], hue='TARGET', dropna=True)
plt.show()


# In[11]:


#data suggests splitting up var38 in most_common_value and rest
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0

#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())
  


# In[12]:



#var15
#The most important feature for XGBoost is var15. According to a Kaggle form post var15 is the age of the customer. Let's explore var15
print(train['var15'].describe())
print(train['var15'].value_counts())

train['var15'].hist(bins=100);


# In[13]:


# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Unhappy customers are slightly older');


# In[14]:


#var_30

train.saldo_var30.hist(bins=100)
plt.xlim(0, train.saldo_var30.max());



# In[15]:


#improve by log_saldo_var30
train['log_saldo_var30'] = train.saldo_var30.map(np.log)

# Let's look at the density of the age of happy/unhappy customers for saldo_var30
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "log_saldo_var30")    .add_legend();



# In[16]:


#Explore the interaction between var15 (age) and var38
sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "var38", "var15")    .add_legend();


# In[17]:


#seems unhappy are has less variation in var38 and in age
sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]); # Age must be positive ;-)


# In[18]:


# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120])


# In[19]:


# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend();


# In[20]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


# In[21]:


X_sel = train[features+['TARGET']]


# In[22]:


#var36
print(X_sel['var36'].value_counts())


sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36")    .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');
#
plt.xlim([-50,180])


# In[23]:


# var36 in function of var38 (most common value excluded) 

sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()
plt.title('If var36==0, only happy customers');


# In[24]:


# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38")    .add_legend();


# In[25]:


train.num_var5.value_counts()


# In[26]:



sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var5")    .add_legend();


# In[27]:


sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde");


# In[28]:


train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6));


# In[29]:


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET");


# In[30]:




