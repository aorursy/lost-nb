#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


#Importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# In[4]:


pd.options.display.max_columns = 150
# reading data from test 
test = pd.read_csv("../input/test.csv")
test.head(3)

train = pd.read_csv("../input/train.csv")
#train.head(3)


# In[5]:


# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.tree import  DecisionTreeClassifier as dt
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb


# In[6]:


# Summary of target feature
train['Target'].value_counts() 


# In[7]:


#Label encoding 
train['dependency'].value_counts()
map1 = {'yes':0,'no':1}
map1
train['dependency'] = train['dependency'].replace(map1).astype(np.float32)


# In[8]:


#Label encoding 
train['edjefa'].value_counts()
map3 = {'yes':0,'no':1}
map3
train['edjefa'] = train['edjefa'].replace(map3).astype(np.float32)


# In[9]:


#Label encoding 
train['edjefe'].value_counts()
map2 = {'yes':0,'no':1}
map2
train['edjefe'] = train['edjefe'].replace(map2).astype(np.float32)


# In[10]:


test['dependency'] = test['dependency'].map({"yes" : 1, "no" : 0})
test['edjefa'] = test['edjefa'].map({"yes" : 1, "no" : 0})
test['edjefe'] = test['edjefe'].map({"yes" : 1, "no" : 0})


# In[11]:


#CLEANING DATA
#     Transform train and test dataframes
#     replacing '0' with NaN
train.replace(0, np.nan)
test.replace(0,np.nan)
#fillna() to replace missing values with the mean value for each column,

train.fillna(train.mean(), inplace=True);
print(train.isnull().sum());

train.shape


# In[12]:


test.fillna(test.mean(), inplace=True);
print(test.isnull().sum());

test.shape


# In[13]:


#Explore Data
#Check dimenstions, column names and data types. 
def getDetails(data):
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    print(f"Data types: {data.dtypes}")

getDetails(train)


# In[14]:


getDetails (test)


# In[15]:


Dropping unnecesary columns


# In[16]:


ID=test['Id']

train.drop(['Id','idhogar'], inplace = True, axis =1)

test.drop(['Id','idhogar'], inplace = True, axis =1)


# In[17]:


Perform data visualisation


# In[18]:


#Relationship of continous variable with the target--Density plots
import seaborn as sns
target_values = [1,2,3,4]

f, axes = plt.subplots(2, 2, figsize=(7, 7))
for target in target_values:
    # Subset as per target--first 1 , 2 , 3 and then 4
    subset = train[train['Target'] == target]

 # Draw the density plot
    sns.distplot(subset['hhsize'], hist = False, kde = True,
                  label = target, ax = axes[0,0])
    sns.distplot(subset['parentesco1'], hist = False, kde = True,
                  label = target, ax = axes[0,1])
    sns.distplot(subset['r4t3'], hist = False, kde = True,
                  label = target, ax = axes[1,0])
    sns.distplot(subset['v2a1'], hist = False, kde = True,
                  label = target, ax = axes[1,1])
    
plt.show()


# In[19]:


Feature-Target Relationships


# In[20]:


sns.countplot("Target", data=train)


# In[21]:


Feature-Feature Relationships The final important relationship to explore is that of the relationships between the attributes.

We can review the relationships between attributes by looking at the distribution of the interactions of each pair of attributes.

This uses a built function to create a matrix of scatter plots of all attributes versus all attributes. The diagonal where each attribute would be plotted against itself shows the Kernel Density Estimation of the attribute instead.


# In[22]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[23]:


from pandas.plotting import scatter_matrix
scatter_matrix(train.select_dtypes('float'), alpha=0.2, figsize=(26, 20), diagonal='kde')


# In[24]:


The below are Distribution plots using seaborn


# In[25]:


from collections import OrderedDict

plt.figure(figsize = (20, 16))
plt.style.use('fivethirtyeight')

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

# Iterate through the float columns
for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# In[26]:


f, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.boxplot(x="Target", y="hhsize", data=train, ax=axes[0, 0])# size of the house hold 
sns.boxplot(x="Target", y="tamviv", data=train, ax=axes[0, 1])# number of person living in the household
sns.boxplot(x="Target", y="v2a1", data=train, ax=axes[1, 0])# monthly rent payment
sns.boxplot(x="Target", y="rooms", data=train, ax=axes[1, 1])#no of rooms in the house 
plt.show()


# In[27]:


y = train.iloc[:,140]
y.unique()


# In[28]:


X = train.iloc[:,1:141]
X.shape


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# In[30]:


#X_train.shape #((7645, 140)
#X_test.shape #(1912, 140)
#y_train.shape #(7645,)
y_test.shape #(1912,)


# In[31]:


Model 1 : Modelling with XGBoosterClassifier


# In[32]:


modelxgb=XGBClassifier()


# In[33]:


start = time.time()
modelgbm = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[34]:


classes = modelgbm.predict(X_test)

classes


# In[35]:


(classes == y_test).sum()/y_test.size 


# In[36]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[37]:


modelxgb.get_params().keys()


# In[38]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
       #'criterion': ['gini'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
       # 'max_features' : (10,64),             # integer-valued parameter
        #'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[39]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[40]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[41]:


modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=85,
               max_features=47,
               min_weight_fraction_leaf=0.035997,
               n_estimators=178)


# In[42]:


start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[43]:


bayes_cv_tuner.best_score_


# In[44]:


bayes_cv_tuner.score(X_test, y_test)


# In[45]:


bayes_cv_tuner.cv_results_['params']


# In[46]:


Model 2 : Random Forest


# In[47]:


modelrf = rf()


# In[48]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[49]:


classes = modelrf.predict(X_test)
(classes == y_test).sum()/y_test.size 


# In[50]:


f1 = f1_score(y_test, classes, average='macro')
f1 


# In[51]:


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


# In[52]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[53]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[54]:


bayes_cv_tuner.best_score_


# In[55]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[56]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[57]:


modelrfTuned=rf(criterion="gini",
               max_depth=88,
               max_features=41,
               min_weight_fraction_leaf=0.1,
               n_estimators=285)


# In[58]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[59]:


rf_predict=modelrfTuned.predict(X_test)
rf_predict


# In[60]:


scale = ss()
test = scale.fit_transform(test)
rf_predict_test=modelrfTuned.predict(test)


# In[61]:


modelKN = KNeighborsClassifier(n_neighbors=7)


# In[62]:


start = time.time()
modelKN = modelKN.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[63]:


classes = modelKN.predict(X_test)
classes


# In[64]:


(classes == y_test).sum()/y_test.size 


# In[65]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=7         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[66]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[67]:


bayes_cv_tuner.best_params_


# In[68]:


modelKNTuned = KNeighborsClassifier(n_neighbors=7, metric="cityblock")


# In[69]:


start = time.time()
modelKNTuned = modelKNTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[70]:


yneigh=modelKNTuned.predict(X_test)


# In[71]:


yneightest=modelKNTuned.predict(test)


# In[72]:


bayes_cv_tuner.best_score_


# In[73]:


bayes_cv_tuner.score(X_test, y_test)


# In[74]:


bayes_cv_tuner.cv_results_['params']


# In[75]:


modeleETClf = ExtraTreesClassifier()


# In[76]:


start = time.time()
modeleETClf = modeleETClf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[77]:


classes = modeleETClf.predict(X_test)
classes


# In[78]:


(classes == y_test).sum()/y_test.size


# In[79]:


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


# In[80]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[81]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[82]:


modeletfTuned=ExtraTreesClassifier(criterion="gini",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=100)


# In[83]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[84]:


yetf=modeletfTuned.predict(X_test)
yetftest=modeletfTuned.predict(test)


# In[85]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[86]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[87]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[88]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[89]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[90]:


classes = modellgb.predict(X_test)

classes


# In[91]:


(classes == y_test).sum()/y_test.size 


# In[92]:


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


# In[93]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[94]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[95]:


modellgbTuned = lgb.LGBMClassifier(criterion="entropy",
               max_depth=35,
               max_features=14,
               min_weight_fraction_leaf=0.18611,
               n_estimators=148)


# In[96]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[97]:


ylgb=modellgbTuned.predict(X_test)
ylgbtest=modellgbTuned.predict(test)


# In[98]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[99]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[100]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[101]:


NewTrain = pd.DataFrame()

NewTrain['ylgb'] = ylgb.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['yetf'] = yetf.tolist()
NewTrain['rf_predict'] = rf_predict.tolist()


NewTrain.head(5), NewTrain.shape


# In[102]:


NewTest = pd.DataFrame()

NewTest['yetf'] = yetftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()
NewTest['rf_predict_test'] = rf_predict_test.tolist()


NewTest.head(5), NewTest.shape


# In[103]:


NewModel=rf(criterion="entropy",
               max_depth=77,
               max_features=3,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[104]:


start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60


# In[105]:


NewPredict=NewModel.predict(NewTest)


# In[106]:


submit=pd.DataFrame({'Id': ID, 'Target': NewPredict})
submit.head(5)


# In[107]:


submit.to_csv('CostaRicaSubmit.csv', index=False)

