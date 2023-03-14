#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input")) #this is the directory for inputs

# Any results you write to the current directory are saved as output.

# For notebook plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Data Viz
import matplotlib.pyplot as plt

from time import time
from scipy.stats import randint as sp_randint

# Scikit + xgb library
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Seed for reproducability
seed = 123
np.random.seed(seed)

# Directory
KAGGLE_DIR = '../input/'

# Info about dataset - referenced from another kernel
print('Files and directories: \n{}\n'.format(os.listdir(KAGGLE_DIR)))
print('Within the train directory: \n{}\n'.format(os.listdir(KAGGLE_DIR + 'train')))
print('Within the test directory: \n{}\n'.format(os.listdir(KAGGLE_DIR + 'test')))

print('\n# File sizes')
for file in os.listdir(KAGGLE_DIR):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))
        
print('\n# File sizes in train: ')
for file in os.listdir(KAGGLE_DIR + 'train/'):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + 'train/' + file) / 1000000, 2))))
        
print('\n# File sizes in test: ')
for file in os.listdir(KAGGLE_DIR + 'test/'):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + 'test/' + file) / 1000000, 2))))


# In[ ]:


#kappa score calculator
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
scorer = make_scorer(kappa)


# In[ ]:


#load data
breeds = pd.read_csv('../input/breed_labels.csv') 
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')
target = train['AdoptionSpeed']

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'


# In[ ]:


train.info() #look at basic info
#seems as if some names are missing - but probably irrelavant
#description
#photo and video may be used for additional analysis


# In[ ]:


#look at all data except description
train.drop('Description', axis=1).head() 


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.hist(bins=50, figsize = (20,15))
plt.show()


# In[ ]:


corr_matrix = train.corr()


# In[ ]:


corr_matrix["AdoptionSpeed"].sort_values(ascending=False) #see what is correlated
#seems that there is very little correlation


# In[ ]:


# Data clean for initial model
# Probably need to implement these models for higher results
target = train['AdoptionSpeed']
clean_df = train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed', 'dataset_type'])
clean_test = test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'dataset_type'])


# In[ ]:


#split training set for training and validation
x_train, x_valid, y_train, y_valid = train_test_split(clean_df, 
                                                      target, 
                                                      test_size=0.2, 
                                                      random_state=seed)

# Preparation for XGBoost
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


# In[ ]:


#see if data is clean... seems as if petfinder.my did most of the cleaning for us
clean_df.isnull().values.any()


# In[ ]:


#lets see what clean data looks like
clean_df.head()


# In[ ]:


#check structure of all data that we cleaned

x_train.head()


# In[ ]:


y_train.head()


# In[ ]:


clean_df.head()


# In[ ]:


target.head()


# In[ ]:


# Fit XGBoost - got initial params from another kernel... need to tune this for better 
# performance -> gridsearch

xgb_params = {'objective' : 'multi:softmax',
              'eval_metric' : 'mlogloss',
              'eta' : 0.05,
              'max_depth' : 4,
              'num_class' : 5,
              'lambda' : 0.8
}


print('Fitting XGBoost: ')
bst = xgb.train(xgb_params, 
                d_train, 
                400, 
                watchlist, 
                early_stopping_rounds=50, 
                verbose_eval=0)

#Fit RandomTree
clf_rfc = RandomForestClassifier()
clf_rfc.fit(x_train, y_train)

clf_etc = ExtraTreesClassifier()
clf_etc.fit(x_train, y_train)

clf_ada = AdaBoostClassifier()
clf_ada.fit(x_train, y_train)

clf_gdc = GradientBoostingClassifier()
clf_gdc.fit(x_train, y_train)

#this is the dict struct that will run our models
models = {'XGBoost' : bst , 
         'RandomTree':  clf_rfc ,
         'ExtraTrees' : clf_etc,
         'Adaboost' : clf_ada,
         'GradientBoost' : clf_gdc}

bst.predict(xgb.DMatrix(clean_test)).astype(int)


#training scores report
print('Training set scores... Check for overfitting, underfitting, etc...:\n')
train_scores = []
for name, model in models.items():
    if name == 'XGBoost':
        score = kappa(bst.predict(xgb.DMatrix(clean_df)).astype(int), target)
        print('{} score: {}'.format(str(name), round(score, 5)))
    else:    
        score = kappa(model.predict(x_valid), y_valid)
        print('{} score: {}'.format(str(name), round(score, 5)))
    train_scores.append(score)

print('\nMean Score: {0:10.4f}'.format(np.mean(train_scores)))

print('\nStandard Deviation of Scores: {0:10.4f}'.format(np.std(train_scores)))


# In[ ]:


# Utility function to report best scores for gridsearch/randomsearch
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


#gridsearch / random search for random forest classifier
#RANDOM SEARCH for random forest classifier


# param, distribution options that random search, grid search will run.
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf_rfc, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


# In[ ]:


#GRID SEARCH for random forest classifier
# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf_rfc, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(x_valid, y_valid)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


#Fit RandomTree with modified Params
clf_rfc_modparam = RandomForestClassifier(bootstrap= False, criterion= 'gini', max_depth = None, 
                             max_features= 8, min_samples_split= 8)
clf_rfc_modparam.fit(x_train, y_train)

#report score
score = kappa(clf_rfc_modparam.predict(x_valid), y_valid)
print('{} score: {}'.format("clf_rfc_modparam", round(score, 4)))

#wow this dropped scores


# In[ ]:


test.head()


# In[ ]:


#submitting files

test_pet_ID = test['PetID']
final = bst.predict(xgb.DMatrix(clean_test))

submission = pd.DataFrame(data={'PetID' : test_pet_ID.tolist(), 
                                   'AdoptionSpeed' : final})
submission.AdoptionSpeed = submission.AdoptionSpeed.astype(int)
submission.to_csv("submission.csv", index=False)

