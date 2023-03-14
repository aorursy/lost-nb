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




get_ipython().system('ls -lh ../input')




from IPython.display import display
def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)




train = pd.read_csv("../input/train.csv", low_memory=False)
test = pd.read_csv("../input/test.csv", low_memory=False)




display_all(train.head())




display_all(test.head())




train.shape




test.shape




sample = pd.read_csv("../input/sample_submission.csv", low_memory=False)




sample.head()




display_all(train.describe())




display_all(test.describe())




train.target.hist()




display_all(pd.DataFrame(train.isnull().sum()).T)




display_all(pd.DataFrame(train.nunique()).T)




from sklearn.model_selection import *




train_X, test_X, train_y, test_y =     train_test_split(train.drop(["ID_code","target"], axis=1), train["target"], 
                     test_size=0.25, random_state=42, stratify=train["target"])




train_X.reset_index(drop=True, inplace=True)
test_X.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)




rf.fit(train_X, train_y)




from sklearn.metrics import *




def model_score(m): 
    return {"train":roc_auc_score(train_y, m.predict(train_X)) ,
            "test":roc_auc_score(test_y, m.predict(test_X))}




model_score(rf)




rf_2 = RandomForestClassifier(min_samples_split=4, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_2.fit(train_X, train_y)')




model_score(rf_2)




lr = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)




get_ipython().run_line_magic('time', 'lr.fit(train_X, train_y)')




model_score(lr)




train_X.shape




# sampling for speed up
from sklearn.utils.random import sample_without_replacement
selected_index = train_X.index[sample_without_replacement(                                   train_X.shape[0], 20000,random_state=42)]




train_X_sub = train_X.loc[selected_index,:].reset_index(drop=True)
train_y_sub = train_y.loc[selected_index].reset_index(drop=True)




print(train_X_sub.shape, train_y_sub.shape)




lr_2 = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)
get_ipython().run_line_magic('time', 'lr_2.fit(train_X_sub, train_y_sub)')




model_score(lr_2)




from sklearn.preprocessing import *




train_X_sub.head()




train_X_sub_sc = (train_X_sub - train_X_sub.mean()) / train_X_sub.std()




# sc = StandardScaler()
# train_X_sub_sc = sc.fit_transform(train_X_sub.reset_index())




# train_X_sub_sc




train_X_sub_sc.shape




lr_3 = LogisticRegression(n_jobs=-1, solver="lbfgs", verbose=1)




lr_3.fit(train_X_sub_sc, train_y_sub)




model_score(lr_3)




rf_3 = RandomForestClassifier(max_depth=5, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)




rf_3 = RandomForestClassifier(min_samples_split=10, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)




rf_3 = RandomForestClassifier(min_samples_split=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'rf_3.fit(train_X, train_y)')
model_score(rf_3)




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




lr_2_p = LogisticReg(n_jobs=-1, solver="lbfgs", verbose=1)
get_ipython().run_line_magic('time', 'lr_2_p.p_fit(train_X_sub, train_y_sub)')




model_score(lr_2_p)




p_values = pd.DataFrame({"feature": train_X.columns,"p_value":lr_2_p.p_values})




p_values.sort_values("p_value")[:30].plot(kind="bar")




p_values[p_values.p_value < 0.05].sort_values("p_value")




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




get_ipython().run_line_magic('time', 'importance = calc_importance(lr_2_p, test_X, test_y)')




df_importance = pd.DataFrame(list(importance.items()), 
                             columns=["feature", "importance"])
df_importance.head()




df_importance.sort_values("importance", ascending=False)[:30].plot(kind="bar")




df_importance.sort_values("importance", ascending=False)    .reset_index(drop=True).plot()




df_importance.sort_values("importance", ascending=False)[:30]




sample.head()




sample.target.hist()




test.head()




X_for_predict = test.drop(["ID_code"], axis=1)




submission = pd.DataFrame({"ID_code":test.ID_code, 
                          "target":lr_2_p.predict(X_for_predict)})




submission.head()




submission.target.hist()




train.target.hist()




sum(train.target > 0)




sum(submission.target > 0)




sorted_prob = np.sort(lr_2_p.predict_proba(X_for_predict)[:,1])[::-1]




sorted_prob[20000]




target_proba = lr_2_p.predict_proba(X_for_predict)[:,1]
target_label = np.where(target_proba >= 0.256, 1, 0)
submission_2 = pd.DataFrame({"ID_code":test.ID_code, 
                "target":target_label})




submission_2.target.hist()




submission_2.to_csv("submission.csv", index=False)




submission_3= pd.DataFrame({"ID_code":test.ID_code,
                           "target":lr_2_p.predict_proba(X_for_predict)[:,1]})
submission_3.to_csv("submission_proba.csv", index=False)






