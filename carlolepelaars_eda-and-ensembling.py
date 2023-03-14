#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For notebook plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Standard libraries
import os
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Seed for reproducability
seed = 12345
np.random.seed(seed)

# Directory
KAGGLE_DIR = '../input/'

print('\n# Files and file sizes')
for file in os.listdir(KAGGLE_DIR):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))
        
print('\n# Files and file sizes in train: ')
for file in os.listdir(KAGGLE_DIR + 'train/'):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + 'train/' + file) / 1000000, 2))))
        
print('\n# Files and file sizes in test: ')
for file in os.listdir(KAGGLE_DIR + 'test/'):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + 'test/' + file) / 1000000, 2))))


# In[2]:


# Read in data
train_df = pd.read_csv(KAGGLE_DIR + "train/train.csv")
test_df = pd.read_csv(KAGGLE_DIR + "test/test.csv")
target = train_df['AdoptionSpeed']


# In[3]:


# Stats
print('Data Statistics:')
train_df.describe()


# In[4]:


# Types
print('Info about types and missing values: ')
train_df.info()


# In[5]:


# Overview
print('This dataset has {} rows and {} columns\n'.format(train_df.shape[0], train_df.shape[1]))
print('Example rows:')
train_df.head(2)


# In[6]:


# Type distribution
train_df['Type'].value_counts().rename({1:'Dog',
                                        2:'Cat'}).plot(kind = 'barh',
                                                       figsize = (15,6))

plt.yticks(fontsize = 'xx-large')
plt.title('Type Distribution', fontsize = 'xx-large')


# In[7]:


# Gender distribution
train_df['Gender'].value_counts().rename({1:'Male',
                                          2:'Female',
                                          3:'Mixed (Group of pets)'}).plot(kind = 'barh', 
                                                                           figsize = (15,6))
plt.yticks(fontsize = 'xx-large')
plt.title('Gender distribution', fontsize = 'xx-large')


# In[8]:


# Age distribution 
train_df['Age'][train_df['Age'] < 50].plot(kind = 'hist', 
                                           bins = 100, 
                                           figsize = (15,6), 
                                           title = 'Age distribution')

plt.title('Age distribution', fontsize = 'xx-large')
plt.xlabel('Age in months')


# In[9]:


# Photo amount distribution
train_df['PhotoAmt'].plot(kind = 'hist', 
                          bins = 30, 
                          xticks = list(range(31)), 
                          figsize = (15,6))

plt.title('PhotoAmt distribution', fontsize='xx-large')
plt.xlabel('Photos')


# In[10]:


# Target variable (Adoption Speed)
print('The values are determined in the following way:\n0 - Pet was adopted on the same day as it was listed.\n1 - Pet was adopted between 1 and 7 days (1st week) after being listed.\n2 - Pet was adopted between 8 and 30 days (1st month) after being listed.\n3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.\n4 - No adoption after 100 days of being listed.\n(There are no pets in this dataset that waited between 90 and 100 days).')

# Plot
train_df['AdoptionSpeed'].value_counts().sort_index(ascending = False).plot(kind = 'barh', 
                                                                            figsize = (15,6))
plt.title('Adoption Speed (Target Variable)', fontsize = 'xx-large')


# In[11]:


# Example Description (of Nibble) ^^ 
print('Example Description (of Nibble) ^^ : ')
train_df['Description'][0]


# In[12]:


# Metric used for this competition (Quadratic Weigthed Kappa aka Quadratic Cohen Kappa Score)
def metric(y1,y2):
    return cohen_kappa_score(y1, y2, weights = 'quadratic')

# Make scorer for scikit-learn
scorer = make_scorer(metric)


# In[13]:


# Clean up DataFrames
target = train_df['AdoptionSpeed']
clean_df = train_df.drop(columns = ['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
clean_test = test_df.drop(columns = ['Name', 'RescuerID', 'Description', 'PetID'])


# In[14]:


# Preparation for XGBoost
x_train, x_valid, y_train, y_valid = train_test_split(clean_df, 
                                                      target, 
                                                      test_size = 0.2, 
                                                      random_state = seed)

d_train = xgb.DMatrix(x_train, label = y_train)
d_valid = xgb.DMatrix(x_valid, label = y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


# In[15]:


# Create base models
gbm = GradientBoostingClassifier()
clf = RandomForestClassifier()
clf2 = ExtraTreesClassifier()
clf3 = AdaBoostClassifier()

# Parameters
xgb_params = {'objective' : 'multi:softmax',
              'eval_metric' : 'mlogloss',
              'eta' : 0.05,
              'max_depth' : 4,
              'num_class' : 5,
              'lambda' : 0.8
}

# Create parameters to use for Grid Search
gbm_grid = {
    'loss' : ['deviance'],
    'learning_rate' : [.025, 0.5],
    'max_depth': [5, 8],
    'max_features': ['auto'],
    'min_samples_leaf': [100],
    'min_samples_split': [100],
    'n_estimators': [100],
    'subsample' : [.8],
    'random_state' : [seed]
}

rand_forest_grid = {
    'bootstrap': [True, False],
    'max_depth': [50, 85],
    'max_features': ['auto'],
    'min_samples_leaf': [10, 15],
    'min_samples_split': [10, 15],
    'n_estimators': [150, 200, 215],
    'random_state' : [seed]
}

extra_trees_grid = {
    'bootstrap' : [True, False], 
    'criterion' : ['gini'], 
    'max_depth' : [50, 75], 
    'max_features': ['auto'], 
    'min_samples_leaf': [10, 15], 
    'min_samples_split': [10, 15],
    'n_estimators': [150, 200, 215], 
    'random_state' : [seed]
}

adaboost_grid = {
    'n_estimators' : [150, 200, 225],
    'learning_rate' : [.2],
    'algorithm' : ['SAMME.R'],
    'random_state' : [seed]
}

# Search parameter space
gbm_gridsearch = GridSearchCV(estimator = gbm, 
                              param_grid = gbm_grid, 
                              cv = 3, 
                              n_jobs = -1, 
                              verbose = 1, 
                              scoring = scorer)

rand_forest_gridsearch = GridSearchCV(estimator = clf, 
                                      param_grid = rand_forest_grid, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring = scorer)

extra_trees_gridsearch = GridSearchCV(estimator = clf2, 
                                      param_grid = extra_trees_grid, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring = scorer)

adaboost_gridsearch = GridSearchCV(estimator = clf3, 
                                   param_grid = adaboost_grid, 
                                   cv = 3, 
                                   n_jobs = -1, 
                                   verbose = 1, 
                                   scoring = scorer)


# In[16]:


# Fit XGBoost
print('Fitting XGBoost: ')
bst = xgb.train(xgb_params, 
                d_train, 
                400, 
                watchlist, 
                early_stopping_rounds = 50, 
                verbose_eval = 0)


# In[17]:


# Fit the models
print('Fitting GBM: ')
gbm_gridsearch.fit(clean_df, target)
print('Fitting Random Forest: ')
rand_forest_gridsearch.fit(clean_df, target)
print('Fitting Extra Trees: ')
extra_trees_gridsearch.fit(clean_df, target)
print('Fitting AdaBoost: ')
adaboost_gridsearch.fit(clean_df, target)


# In[18]:


# What are the best parameters for each model
print('Best model parameters:\n')
print('Gradient Boosting model:\n{}\n'.format(gbm_gridsearch.best_params_))
print('Random Forest model:\n{}\n'.format(rand_forest_gridsearch.best_params_))
print('Extra Trees model:\n{}\n'.format(extra_trees_gridsearch.best_params_))
print('Adaboost model:\n{}\n'.format(adaboost_gridsearch.best_params_))


# In[19]:


# Score on training set
models = {'XGBoost' : bst,
          'Gradient Boosting' : gbm_gridsearch, 
          'Random Forest' : rand_forest_gridsearch, 
          'Extra Trees' : extra_trees_gridsearch, 
          'Adaboost' : adaboost_gridsearch}

print('Score on the training set. This allows us to spot overfitting, performance, etc. (Rounded to 4 decimals):\n')
train_scores = []
for name, model in models.items():
    if name == 'XGBoost':
        score = metric(bst.predict(xgb.DMatrix(clean_df)).astype(int), target)
        print('{} score: {}'.format(str(name), round(score, 4)))
    else:    
        score = metric(model.predict(clean_df), target)
        print('{} score: {}'.format(str(name), round(score, 4)))
    train_scores.append(score)

print('\nMean Score: {0:10.4f}'.format(np.mean(train_scores)))

print('\nStandard Deviation of Scores: {0:10.4f}'.format(np.std(train_scores)))


# In[20]:


# Cross validation
val_GBM = list(cross_val_score(gbm_gridsearch, 
                               clean_df, 
                               target, 
                               scoring = scorer, 
                               cv = 5))

val_RF = list(cross_val_score(rand_forest_gridsearch, 
                              clean_df, 
                              target, 
                              scoring = scorer, 
                              cv = 5))

val_ET = list(cross_val_score(extra_trees_gridsearch, 
                              clean_df, 
                              target, 
                              scoring = scorer, 
                              cv = 5))

val_ADA = list(cross_val_score(adaboost_gridsearch, 
                               clean_df, 
                               target, 
                               scoring = scorer, 
                               cv = 5))

# Validation score for XGBoost
val_XGB = metric(bst.predict(xgb.DMatrix(x_valid)).astype(int), y_valid)


# In[21]:


print('Cross validation scores:\n\n')
print('Validation Score XGBoost:\n{}\n\n'.format(val_XGB))

print('Cross validation Gradient Boosting:\n{},\nMean score: {}\nStd of scores: {}\n\n'.format([round(elem, 4) for elem in val_GBM],
                                                                                               round(np.mean(val_GBM), 4),
                                                                                               round(np.std(val_GBM), 4)))

print('Cross validation Random Forest:\n{},\nMean score: {}\nStd of scores: {}\n\n'.format([round(elem, 4) for elem in val_RF],
                                                                                           round(np.mean(val_RF), 4),
                                                                                           round(np.std(val_RF), 4)))

print('Cross validation Extra Trees:\n{},\nMean score: {}\nStd of scores: {}\n\n'.format([round(elem, 4) for elem in val_ET], 
                                                                                        round(np.mean(val_ET), 4),
                                                                                        round(np.std(val_ET), 4)))

print('Cross validation AdaBoost:\n{},\nMean score: {}\nStd of scores: {}\n\n'.format([round(elem, 4) for elem in val_ADA], 
                                                                                      round(np.mean(val_ADA), 4),
                                                                                      round(np.std(val_ADA), 4)))

print('Mean Validation score: {}'.format(round(np.mean([np.mean(val_GBM),
                                                        np.mean(val_RF), 
                                                        np.mean(val_ET), 
                                                        np.mean(val_ADA), 
                                                        val_XGB]), 4)))

print('Standard Deviation of Cross Validation scores: {}'.format(round(np.std([np.mean(val_GBM),
                                                                               np.mean(val_RF), 
                                                                               np.mean(val_ET), 
                                                                               np.mean(val_ADA), 
                                                                               val_XGB]), 4)))


# In[22]:


# Get predictions
pred0 = gbm_gridsearch.predict(clean_test)
pred1 = rand_forest_gridsearch.predict(clean_test)
pred2 = extra_trees_gridsearch.predict(clean_test)
pred3 = adaboost_gridsearch.predict(clean_test)
pred4 = bst.predict(xgb.DMatrix(clean_test)).astype(int)

# Combine predictions
final_predictions = []
# Get average of predictions
for pred in zip(pred0, pred1, pred2, pred3, pred4):
    final_predictions.append(int(round((sum(pred)) / len(pred), 0)))


# In[23]:


# Compare predictions
prediction_df = pd.DataFrame({'PetID' : test_df['PetID'],
                              'Gradient Boosting' : pred0,
                              'Random Forest' : pred1,
                              'Extra Trees' : pred2,
                              'Adaboost' : pred3,
                              'XGBoost' : pred4
})

print('Predictions for each model: ')
prediction_df.head(10)


# In[24]:


# Store predictions for Kaggle Submission
submission_df = pd.DataFrame(data = {'PetID' : test_df['PetID'], 
                                     'AdoptionSpeed' : final_predictions})
submission_df.to_csv('submission.csv', index = False)


# In[25]:


# Check submission
submission_df.head(3)


# In[26]:


# Compare distributions of training set and test set (Adoption Speed)

# Plot 1
plt.figure(figsize = (15,4))
plt.subplot(211)
train_df['AdoptionSpeed'].value_counts().sort_index(ascending = False).plot(kind = 'barh')
plt.title('Target Variable distribution in training set', fontsize = 'large')

# Plot 2
plt.subplot(212)
submission_df['AdoptionSpeed'].value_counts().sort_index(ascending = False).plot(kind = 'barh')
plt.title('Target Variable distribution in predictions')

plt.subplots_adjust(top = 2)

