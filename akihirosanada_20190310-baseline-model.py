#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().system('ls -lh ../input')


# In[3]:


from IPython.display import display
def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


# In[4]:


train = pd.read_csv("../input/train.csv", low_memory=False)
test = pd.read_csv("../input/test.csv", low_memory=False)


# In[5]:


display_all(train.head())


# In[6]:


display_all(test.head())


# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


sample = pd.read_csv("../input/sample_submission.csv", low_memory=False)


# In[10]:


sample.head()


# In[11]:


display_all(train.describe())


# In[12]:


display_all(test.describe())


# In[13]:


train.target.hist()


# In[14]:


display_all(pd.DataFrame(train.isnull().sum()).T)


# In[15]:


display_all(pd.DataFrame(train.nunique()).T)


# In[16]:


from sklearn.model_selection import *


# In[17]:


train_X, test_X, train_y, test_y =     train_test_split(train.drop(["ID_code","target"], axis=1), train["target"], 
                     test_size=0.25, random_state=42, stratify=train["target"])


# In[18]:


train_X.reset_index(drop=True, inplace=True)
test_X.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[20]:


rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)


# In[21]:


rf.fit(train_X, train_y)


# In[22]:


from sklearn.metrics import *


# In[23]:


def model_score(m): 
    return {"train":roc_auc_score(train_y, m.predict(train_X)) ,
            "test":roc_auc_score(test_y, m.predict(test_X))}


# In[24]:


model_score(rf)


# In[25]:


rf_2 = RandomForestClassifier(min_samples_split=4, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_2.fit(train_X, train_y)')


# In[26]:


model_score(rf_2)


# In[27]:


lr = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)


# In[28]:


get_ipython().run_line_magic('time', 'lr.fit(train_X, train_y)')


# In[29]:


model_score(lr)


# In[30]:


train_X.shape


# In[31]:


# sampling for speed up
from sklearn.utils.random import sample_without_replacement
selected_index = train_X.index[sample_without_replacement(                                   train_X.shape[0], 20000,random_state=42)]


# In[32]:


train_X_sub = train_X.loc[selected_index,:].reset_index(drop=True)
train_y_sub = train_y.loc[selected_index].reset_index(drop=True)


# In[33]:


print(train_X_sub.shape, train_y_sub.shape)


# In[34]:


lr_2 = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)
get_ipython().run_line_magic('time', 'lr_2.fit(train_X_sub, train_y_sub)')


# In[35]:


model_score(lr_2)


# In[36]:


from sklearn.preprocessing import *


# In[37]:


train_X_sub.head()


# In[38]:


train_X_sub_sc = (train_X_sub - train_X_sub.mean()) / train_X_sub.std()


# In[39]:


# sc = StandardScaler()
# train_X_sub_sc = sc.fit_transform(train_X_sub.reset_index())


# In[40]:


# train_X_sub_sc


# In[41]:


train_X_sub_sc.shape


# In[42]:


lr_3 = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)


# In[43]:


lr_3.fit(train_X_sub_sc, train_y_sub)


# In[44]:


model_score(lr_3)


# In[45]:


rf_3 = RandomForestClassifier(max_depth=5, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)


# In[46]:


rf_3 = RandomForestClassifier(min_samples_split=10, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)


# In[47]:


rf_3 = RandomForestClassifier(min_samples_split=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)


# In[48]:


# https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
import scipy.stats as stat

class LogisticReg(LogisticRegression):
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance 
    in an attribute self.model, and pvalues, z scores and estimated 
    errors for each coefficient in 
    
    self.z_scores
    self.p_values
    self.sigma_estimates
    
    as well as the negative hessian of the log Likelihood (Fisher information)
    
    self.F_ij
    """
    
    def p_fit(self,X,y):
        self.fit(X,y)
        #### Get p-values for the fitted model ####
        denom = (2.0*(1.0+np.cosh(self.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
        
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij
        return self


# In[49]:


lr_2_p = LogisticReg(n_jobs=-1, solver="lbfgs", verbose=1)
get_ipython().run_line_magic('time', 'lr_2_p.p_fit(train_X_sub, train_y_sub)')


# In[50]:


model_score(lr_2_p)


# In[51]:


p_values = pd.DataFrame({"feature": train_X.columns,"p_value":lr_2_p.p_values})


# In[52]:


p_values.sort_values("p_value")[:30].plot(kind="bar")


# In[53]:


p_values[p_values.p_value < 0.05].sort_values("p_value")


# In[54]:


from tqdm import tqdm_notebook as tqdm

def calc_importance(model, df_X, y):
    full_score = roc_auc_score(y, model.predict(df_X))
    importance = {}
    for n in tqdm(df_X.columns):
        df_X_copy = df_X.copy()
        df_X_copy.loc[:,n] = df_X.loc[:,n].sample(frac=1).reset_index(drop=True)
        after_score = roc_auc_score(y, model.predict(df_X_copy))
        importance[n] = full_score - after_score
    return importance


# In[55]:


get_ipython().run_line_magic('time', 'importance = calc_importance(lr_2_p, test_X, test_y)')


# In[56]:


df_importance = pd.DataFrame(list(importance.items()), 
                             columns=["feature", "importance"])
df_importance.head()


# In[57]:


df_importance.sort_values("importance", ascending=False)[:30].plot(kind="bar")


# In[58]:


df_importance.sort_values("importance", ascending=False)    .reset_index(drop=True).plot()


# In[59]:


df_importance.sort_values("importance", ascending=False)[:30]


# In[60]:


sample.head()


# In[61]:


sample.target.hist()


# In[62]:


test.head()


# In[63]:


X_for_predict = test.drop(["ID_code"], axis=1)


# In[64]:


submission = pd.DataFrame({"ID_code":test.ID_code, 
                          "target":lr_2_p.predict(X_for_predict)})


# In[65]:


submission.head()


# In[66]:


submission.target.hist()


# In[67]:


train.target.hist()


# In[68]:


sum(train.target > 0)


# In[69]:


sum(submission.target > 0)


# In[70]:


sorted_prob = np.sort(lr_2_p.predict_proba(X_for_predict)[:,1])[::-1]


# In[71]:


sorted_prob[20000]


# In[72]:


target_proba = lr_2_p.predict_proba(X_for_predict)[:,1]
target_label = np.where(target_proba >= 0.256, 1, 0)
submission_2 = pd.DataFrame({"ID_code":test.ID_code, 
                "target":target_label})


# In[73]:


submission_2.target.hist()


# In[74]:


submission_2.to_csv("submission.csv", index=False)


# In[75]:


submission_3= pd.DataFrame({"ID_code":test.ID_code,
                           "target":lr_2_p.predict_proba(X_for_predict)[:,1]})
submission_3.to_csv("submission_proba.csv", index=False)


# In[76]:




