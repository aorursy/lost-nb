#!/usr/bin/env python
# coding: utf-8



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




pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




train.head()




train.info()   




sns.countplot("Target", data=train)




sns.countplot(x="r4t3",hue="Target",data=train)




sns.countplot(x="v18q",hue="Target",data=train)




sns.countplot(x="v18q1",hue="Target",data=train)




sns.countplot(x="tamhog",hue="Target",data=train)




sns.countplot(x="hhsize",hue="Target",data=train)




sns.countplot(x="abastaguano",hue="Target",data=train)




sns.countplot(x="noelec",hue="Target",data=train)




train.select_dtypes('object').head()






yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
    
    




train[["dependency","edjefe","edjefa"]].describe()




# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)




train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)




train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)




train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)




#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)




train.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)

test.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)




train.shape




test.shape




y = train.iloc[:,137]
y.unique()




X = train.iloc[:,1:138]
X.shape





my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
pca = PCA(0.95)
X = pca.fit_transform(X)




X.shape, y.shape




X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)





modelrf = rf()




start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelrf.predict(X_test)




(classes == y_test).sum()/y_test.size 




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modeletf = ExtraTreesClassifier()




start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modeletf.predict(X_test)

classes




(classes == y_test).sum()/y_test.size




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modelneigh = KNeighborsClassifier(n_neighbors=4)




start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelneigh.predict(X_test)

classes




(classes == y_test).sum()/y_test.size 




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modelgbm=gbm()




start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelgbm.predict(X_test)

classes




(classes == y_test).sum()/y_test.size 




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modelxgb=XGBClassifier()




start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelxgb.predict(X_test)

classes




(classes == y_test).sum()/y_test.size 




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)




start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modellgb.predict(X_test)

classes




(classes == y_test).sum()/y_test.size 




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





# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']

