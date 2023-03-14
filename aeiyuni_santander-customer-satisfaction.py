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




train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
train.head()




## from discussion
## var15: age, num_var4: number of product, var38: customer lifetime value, 




train.shape




train.describe().transpose()




train.columns









import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(train['TARGET'], palette='Set3')
## very imbalance dataset. 
## 0: Satisfies, 1: Unsatistfies




df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df




X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']




sns.distplot(train['var15'])
## maybe it is age.




sns.distplot(train['var3'])
# discussion said it was nationality




f = pd.DataFrame(train.columns)
f




## number of product purchased by each user. 
train['num_var4']




train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()




sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var4")    .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()




train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');




sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "n0")    .add_legend()
plt.title('Unhappy customers has lesser feature');

