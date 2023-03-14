#!/usr/bin/env python
# coding: utf-8

# In[1]:


# essential libraries
import numpy as np 
import pandas as pd
# for data visulization
import matplotlib.pyplot as plt
import seaborn as sns


#for data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

# for measuring performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#for tuning parameters
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance

# Misc.
import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# In[2]:


pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train.head()


# In[4]:


train.info()   


# In[5]:


sns.countplot("Target", data=train)


# In[6]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[7]:


sns.countplot(x="v18q",hue="Target",data=train)


# In[8]:


sns.countplot(x="v18q1",hue="Target",data=train)


# In[9]:


sns.countplot(x="tamhog",hue="Target",data=train)


# In[10]:


sns.countplot(x="hhsize",hue="Target",data=train)


# In[11]:


sns.countplot(x="abastaguano",hue="Target",data=train)


# In[12]:


sns.countplot(x="noelec",hue="Target",data=train)


# In[13]:


train.select_dtypes('object').head()


# In[14]:




yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
    
    


# In[15]:


train[["dependency","edjefe","edjefa"]].describe()


# In[16]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[17]:


train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)


# In[18]:


train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)


# In[19]:


train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# In[20]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[21]:


train.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)

test.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)


# In[22]:


train.shape


# In[23]:


test.shape


# In[24]:


y = train.iloc[:,137]
y.unique()


# In[25]:


X = train.iloc[:,1:138]
X.shape


# In[26]:



my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
pca = PCA(0.95)
X = pca.fit_transform(X)


# In[27]:


X.shape, y.shape


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# In[29]:



modelrf = rf()


# In[30]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[31]:


classes = modelrf.predict(X_test)


# In[32]:


(classes == y_test).sum()/y_test.size 


# In[33]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    rf(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[34]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[35]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[36]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[37]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[38]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[39]:


modeletf = ExtraTreesClassifier()


# In[40]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[41]:


classes = modeletf.predict(X_test)

classes


# In[42]:


(classes == y_test).sum()/y_test.size


# In[43]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    ExtraTreesClassifier( ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {   'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
)


# In[44]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[45]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[46]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[47]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[48]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[49]:


modelneigh = KNeighborsClassifier(n_neighbors=4)


# In[50]:


start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[51]:


classes = modelneigh.predict(X_test)

classes


# In[52]:


(classes == y_test).sum()/y_test.size 


# In[53]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=4         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[54]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[55]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[56]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[57]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[58]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[59]:


modelgbm=gbm()


# In[60]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[61]:


classes = modelgbm.predict(X_test)

classes


# In[62]:


(classes == y_test).sum()/y_test.size 


# In[63]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    gbm(
               # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 2                # Number of cross-validation folds
)


# In[64]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[65]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[66]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[67]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[68]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[69]:


modelxgb=XGBClassifier()


# In[70]:


start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[71]:


classes = modelxgb.predict(X_test)

classes


# In[72]:


(classes == y_test).sum()/y_test.size 


# In[73]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[74]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[75]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[76]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[77]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[78]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[79]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[80]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[81]:


classes = modellgb.predict(X_test)

classes


# In[82]:


(classes == y_test).sum()/y_test.size 


# In[83]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    lgb.LGBMClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[84]:



# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[85]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[86]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[87]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[88]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']

