#!/usr/bin/env python
# coding: utf-8



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




pd.options.display.max_columns = 150
# reading data from test 
test = pd.read_csv("../input/test.csv")
test.head(3)

train = pd.read_csv("../input/train.csv")
#train.head(3)




# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.tree import  DecisionTreeClassifier as dt
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb




# Summary of target feature
train['Target'].value_counts() 




#Label encoding 
train['dependency'].value_counts()
map1 = {'yes':0,'no':1}
map1
train['dependency'] = train['dependency'].replace(map1).astype(np.float32)




#Label encoding 
train['edjefa'].value_counts()
map3 = {'yes':0,'no':1}
map3
train['edjefa'] = train['edjefa'].replace(map3).astype(np.float32)




#Label encoding 
train['edjefe'].value_counts()
map2 = {'yes':0,'no':1}
map2
train['edjefe'] = train['edjefe'].replace(map2).astype(np.float32)




test['dependency'] = test['dependency'].map({"yes" : 1, "no" : 0})
test['edjefa'] = test['edjefa'].map({"yes" : 1, "no" : 0})
test['edjefe'] = test['edjefe'].map({"yes" : 1, "no" : 0})




#CLEANING DATA
#     Transform train and test dataframes
#     replacing '0' with NaN
train.replace(0, np.nan)
test.replace(0,np.nan)
#fillna() to replace missing values with the mean value for each column,

train.fillna(train.mean(), inplace=True)
print(train.isnull().sum())

train.shape




test.fillna(test.mean(), inplace=True)
print(test.isnull().sum())

test.shape




#Explore Data
#Check dimenstions, column names and data types. 
def getDetails(data):
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    print(f"Data types: {data.dtypes}")

getDetails(train)




getDetails (test)


ID=test['Id']

train.drop(['Id','idhogar'], inplace = True, axis =1)

test.drop(['Id','idhogar'], inplace = True, axis =1)


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


sns.countplot("Target", data=train)

sns.countplot(x="r4t3",hue="Target",data=train)

from pandas.plotting import scatter_matrix
scatter_matrix(train.select_dtypes('float'), alpha=0.2, figsize=(26, 20), diagonal='kde')

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




f, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.boxplot(x="Target", y="hhsize", data=train, ax=axes[0, 0])# size of the house hold 
sns.boxplot(x="Target", y="tamviv", data=train, ax=axes[0, 1])# number of person living in the household
sns.boxplot(x="Target", y="v2a1", data=train, ax=axes[1, 0])# monthly rent payment
sns.boxplot(x="Target", y="rooms", data=train, ax=axes[1, 1])#no of rooms in the house 
plt.show()




y = train.iloc[:,140]
y.unique()




X = train.iloc[:,1:141]
X.shape




X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)




#X_train.shape #((7645, 140)
#X_test.shape #(1912, 140)
#y_train.shape #(7645,)
y_test.shape #(1912,)

modelxgb=XGBClassifier()




start = time.time()
modelgbm = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelgbm.predict(X_test)

classes




(classes == y_test).sum()/y_test.size 




f1 = f1_score(y_test, classes, average='macro')
f1




modelxgb.get_params().keys()




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




#  Get list of best-parameters
bayes_cv_tuner.best_params_




modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=85,
               max_features=47,
               min_weight_fraction_leaf=0.035997,
               n_estimators=178)




start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60




bayes_cv_tuner.best_score_




bayes_cv_tuner.score(X_test, y_test)




bayes_cv_tuner.cv_results_['params']


modelrf = rf()




start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelrf.predict(X_test)
(classes == y_test).sum()/y_test.size 




f1 = f1_score(y_test, classes, average='macro')
f1 




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




bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




modelrfTuned=rf(criterion="gini",
               max_depth=88,
               max_features=41,
               min_weight_fraction_leaf=0.1,
               n_estimators=285)




start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60




rf_predict=modelrfTuned.predict(X_test)
rf_predict




scale = ss()
test = scale.fit_transform(test)
rf_predict_test=modelrfTuned.predict(test)




modelKN = KNeighborsClassifier(n_neighbors=7)




start = time.time()
modelKN = modelKN.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modelKN.predict(X_test)
classes




(classes == y_test).sum()/y_test.size 




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




# Start optimization
bayes_cv_tuner.fit(X_train, y_train)




bayes_cv_tuner.best_params_




modelKNTuned = KNeighborsClassifier(n_neighbors=7, metric="cityblock")




start = time.time()
modelKNTuned = modelKNTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60




yneigh=modelKNTuned.predict(X_test)




yneightest=modelKNTuned.predict(test)




bayes_cv_tuner.best_score_




bayes_cv_tuner.score(X_test, y_test)




bayes_cv_tuner.cv_results_['params']




modeleETClf = ExtraTreesClassifier()




start = time.time()
modeleETClf = modeleETClf.fit(X_train, y_train)
end = time.time()
(end-start)/60




classes = modeleETClf.predict(X_test)
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




modeletfTuned=ExtraTreesClassifier(criterion="gini",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=100)




start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60




yetf=modeletfTuned.predict(X_test)
yetftest=modeletfTuned.predict(test)




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




modellgbTuned = lgb.LGBMClassifier(criterion="entropy",
               max_depth=35,
               max_features=14,
               min_weight_fraction_leaf=0.18611,
               n_estimators=148)




start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60




ylgb=modellgbTuned.predict(X_test)
ylgbtest=modellgbTuned.predict(test)




#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_




#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)




#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']




NewTrain = pd.DataFrame()

NewTrain['ylgb'] = ylgb.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['yetf'] = yetf.tolist()
NewTrain['rf_predict'] = rf_predict.tolist()


NewTrain.head(5), NewTrain.shape




NewTest = pd.DataFrame()

NewTest['yetf'] = yetftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()
NewTest['rf_predict_test'] = rf_predict_test.tolist()


NewTest.head(5), NewTest.shape




NewModel=rf(criterion="entropy",
               max_depth=77,
               max_features=3,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)




start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60




NewPredict=NewModel.predict(NewTest)




submit=pd.DataFrame({'Id': ID, 'Target': NewPredict})
submit.head(5)




submit.to_csv('CostaRicaSubmit.csv', index=False)

