#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:



from shutil import copyfile

copyfile(src = "../input/thinkstats/thinkstats2.py", dst = "../working/thinkstats2.py")
copyfile(src = "../input/thinkstats/thinkplot.py", dst = "../working/thinkplot.py")

from thinkstats2 import *
from thinkplot import *


# In[3]:


import thinkplot
import thinkstats2


# In[4]:


train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.shape, test.shape


# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# In[10]:


X_train = train.iloc[:, 2:].values
y_train = train.target.values


# In[11]:


X_train


# In[12]:


y_train


# In[13]:


X_test = test.iloc[:, 1:].values


# In[14]:


X, y = train.iloc[:,2:], train.iloc[:,1]


# In[15]:


X_test


# In[16]:


train.info()


# In[17]:


train.count().tail()


# In[18]:


train.describe()


# In[19]:


test.describe()


# In[20]:


train.nunique()


# In[21]:


test.nunique()


# In[22]:


def isnull_data(dataset):
    nulls = dataset.isnull().sum() 
    tot = pd.concat([nulls], axis=1, keys=['Nulls']) 
    types = []
    for columns in dataset.columns:
        dtype = str(dataset[columns].dtype)
        types.append(dtype)
    tot['Types'] = types
    return(np.transpose(tot))


# In[23]:


isnull_data(train)


# In[24]:


isnull_data(test)


# In[25]:


sns.countplot(train.target)


# In[26]:


col = train.columns.values[2:202]


# In[ ]:





# In[27]:


thinkplot.Scatter(train[col[0]],test[col[0]], alpha=1)
thinkplot.Config(legend=False)


# In[28]:


thinkplot.Scatter(train[col[1]],test[col[1]], alpha=1)
thinkplot.Config(legend=False)


# In[29]:


thinkplot.Scatter(train[col[2]],test[col[2]], alpha=1)
thinkplot.Config(legend=False)


# In[30]:


thinkplot.Scatter(train[col[3]],test[col[3]], alpha=1)
thinkplot.Config(legend=False)


# In[31]:


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[col].mean(axis=0),color="green", kde=True,bins=120, label='train')
sns.distplot(test[col].mean(axis=0),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[32]:


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per column in the train and test set")
sns.distplot(train[col].min(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[col].min(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[33]:


targetEqZero = train.loc[train['target'] == 0][col]
targetEqOne = train.loc[train['target'] == 1][col]


# In[34]:


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per column in the train and test set")
sns.distplot(train[col].max(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[col].max(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[35]:


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per row in the train set")
sns.distplot(targetEqZero.min(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sns.distplot(targetEqOne.min(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# In[36]:


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per row in the train set")
sns.distplot(targetEqZero.max(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sns.distplot(targetEqOne.max(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# In[37]:


unique_max_train = []
unique_max_test = []
for col in col:
    values = train[col].value_counts()
    unique_max_train.append([col, values.max(), values.idxmax()])
    values = test[col].value_counts()
    unique_max_test.append([col, values.max(), values.idxmax()])


# In[38]:


np.transpose((pd.DataFrame(unique_max_train, columns=['Column', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False).head(10))


# In[39]:


features = train.columns.values[2:202]


# In[40]:


train_standart = pd.DataFrame()
test_standart = pd.DataFrame()


# In[41]:


idx = train.columns.values[2:202]

for df in [train]:
    train_standart['target'] = df.target
    train_standart['sum'] = df[idx].sum(axis=1)  
    train_standart['min'] = df[idx].min(axis=1)
    train_standart['max'] = df[idx].max(axis=1)
    train_standart['mean'] = df[idx].mean(axis=1)
    train_standart['std'] = df[idx].std(axis=1)
 


# In[42]:


train_standart.head


# In[43]:


idx = train.columns.values[2:202]
for df in [test]:
    test_standart['sum'] = df[idx].sum(axis=1)  
    test_standart['min'] = df[idx].min(axis=1)
    test_standart['max'] = df[idx].max(axis=1)
    test_standart['mean'] = df[idx].mean(axis=1)
    test_standart['std'] = df[idx].std(axis=1)
 


# In[44]:


test_standart.head


# In[45]:


t0 = train_standart.loc[train_standart['target'] == 0]
t1 = train_standart.loc[train_standart['target'] == 1]


# In[46]:


pdf_min_t0 = thinkstats2.EstimatedPdf(t0['min'])
pdf_min_t1 = thinkstats2.EstimatedPdf(t1['min'])
thinkplot.Pdf(pdf_min_t0, label='target 0')
thinkplot.Pdf(pdf_min_t1, label='target 1')
thinkplot.Config(xlabel='Min', ylabel='PDF')


# In[47]:


pdf_sum_t0 = thinkstats2.EstimatedPdf(t0['sum'])
pdf_sum_t1 = thinkstats2.EstimatedPdf(t1['sum'])
thinkplot.Pdf(pdf_sum_t0, label='target 0')
thinkplot.Pdf(pdf_sum_t1, label='target 1')
thinkplot.Config(xlabel='Sum', ylabel='PDF')


# In[48]:


pdf_max_t0 = thinkstats2.EstimatedPdf(t0['max'])
pdf_max_t1 = thinkstats2.EstimatedPdf(t1['max'])
thinkplot.Pdf(pdf_max_t0, label='target 0')
thinkplot.Pdf(pdf_max_t1, label='target 1')
thinkplot.Config(xlabel='Max', ylabel='PDF')


# In[49]:


pdf_mean_t0 = thinkstats2.EstimatedPdf(t0['mean'])
pdf_mean_t1 = thinkstats2.EstimatedPdf(t1['mean'])
thinkplot.Pdf(pdf_mean_t0, label='target 0')
thinkplot.Pdf(pdf_mean_t1, label='target 1')
thinkplot.Config(xlabel='Mean', ylabel='PDF')


# In[50]:


pdf_std_t0 = thinkstats2.EstimatedPdf(t0['std'])
pdf_std_t1 = thinkstats2.EstimatedPdf(t1['std'])
thinkplot.Pdf(pdf_std_t0, label='target 0')
thinkplot.Pdf(pdf_std_t1, label='target 1')
thinkplot.Config(xlabel='Std', ylabel='PDF')


# In[51]:


correlation = train.corr(method='pearson', min_periods=1).abs().unstack().sort_values(kind="quicksort").reset_index()
correlation.head(10)


# In[52]:


sns.FacetGrid(train, hue="target", size=5)    .map(plt.scatter, "var_1", "var_2")    .add_legend()


# In[53]:


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


# In[54]:


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


# In[55]:


def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr


# In[56]:


Corr(train['var_0'], train['var_1'])


# In[57]:


np.corrcoef(train['var_0'], train['var_1'])


# In[58]:


def plot_feature_importances(model, columns):
    nr_f = columns.shape[0]
    imp = pd.Series(data = model.feature_importances_, 
                    index=columns).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')


# In[59]:


from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[60]:


import pickle


# In[61]:


xgb_s = XGBClassifier()


# In[62]:


get_ipython().run_cell_magic('time', '', 'xgb_s.fit(X_train,y_train)')


# In[63]:


preds_xgb = xgb.predict(X_test)


# In[64]:


xgb_s_y_preds = xgb_s.predict(X_test)


# In[65]:


pickle.dump(xgb_s, open("pima.xgb_s.dat", "wb"))


# In[66]:


loaded_model = pickle.load(open("pima.xgb_s.dat", "rb"))


# In[67]:


plot_feature_importances(loaded_model, train.drop('Target', axis=1).columns)


# In[68]:


pickle.dump(xgb, open("pima.pickle.dat", "wb"))


# In[69]:


sub_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": xgb_s_y_preds
                      })
sub_df.to_csv("submission_xgb.csv", index=False)


# In[70]:


sub_xbs = pd.read_csv("submission_xgb.csv")


# In[71]:


sns.countplot(sub_xbs.target)


# In[72]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[73]:


get_ipython().run_cell_magic('time', '', 'clf = tree.DecisionTreeClassifier(max_depth=10)\nclf = clf.fit(X_train, y_train)')


# In[74]:


clf_pred = clf.predict(X_test)


# In[75]:


dtc_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": clf_pred
                      })
dtc_df.to_csv("submission_dtc.csv", index=False)


# In[76]:


dtc_data = pd.read_csv("submission_dtc.csv")


# In[77]:


sns.countplot(dtc_data.target)
    


# In[78]:


from sklearn.neighbors import KNeighborsClassifier


# In[79]:


knn = KNeighborsClassifier()


# In[80]:


get_ipython().run_cell_magic('time', '', 'knn.fit(X_train,y_train)')


# In[81]:


knn_preds = knn.predict(X_test)


# In[82]:


knn_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": knn_preds
                      })
knn_df.to_csv("submission_knn.csv", index=False)


# In[83]:


knn_data = pd.read_csv("submission_knn.csv")


# In[84]:


sns.countplot(knn_data.target)


# In[85]:


from sklearn.linear_model import LogisticRegression


# In[86]:


log_reg_cls = LogisticRegression()


# In[87]:


get_ipython().run_cell_magic('time', '', 'log_reg_cls.fit(X_train, y_train)')


# In[88]:


y_preds_log_reg = log_reg_cls.predict(X_test)


# In[89]:


log_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": y_preds_log_reg
                      })
log_df.to_csv("submission_log.csv", index=False)


# In[90]:


log_data = pd.read_csv("submission_log.csv")


# In[91]:


sns.countplot(log_data.target)

