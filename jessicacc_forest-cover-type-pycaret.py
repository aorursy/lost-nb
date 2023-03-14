#!/usr/bin/env python
# coding: utf-8



# Install PyCaret
get_ipython().system('pip install pycaret')




# install watermark
get_ipython().system('pip install watermark')




import pandas as pd
import numpy as np
import random
import matplotlib as m
import matplotlib.pyplot as plt
import seaborn as sns

#import pycaret 
import pycaret   
from pycaret.classification import *  #import classification module 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# make pandas show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Formating the plots
plt.rcParams.update(plt.rcParamsDefault)
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight')
m.rcParams['axes.labelsize'] = 14
m.rcParams['xtick.labelsize'] = 12
m.rcParams['ytick.labelsize'] = 12
m.rcParams['figure.figsize'] = (15, 5)
m.rcParams['font.size'] = 12
m.rcParams['legend.fontsize'] = 'large'
m.rcParams['figure.titlesize'] = 'medium'
m.rcParams['text.color'] = 'k'
sns.set(rc={'figure.figsize':(15,5)})




# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Forest Cover Type Prediction -- Jessica Cabral" --iversions')
get_ipython().run_line_magic('watermark', '-n -t -z')




np.random.seed(42)
random.seed(42)
random_seed = 42




train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
sample_submission = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')


print('Train: {}'.format(train.shape))
print('test: {}'.format(test.shape))
print('sample_submission: {}'.format(sample_submission.shape))




display(train.head(), test.head())

train.shape, test.shape




# remove de ID column

train = train.drop(columns=['Id'], axis=1)
test = test.drop(columns=['Id'], axis=1)

train.shape, test.shape




# separate training and test dataset

train, validation = train_test_split(train, test_size=0.33, random_state=random_seed)

train.shape, validation.shape




exp_clf = setup(data = train,           # train data
              target = 'Cover_Type',   # feature that we are trying to predict
              train_size = 0.7)     # proportion of training data




get_ipython().run_cell_magic('time', '', '\n# Train the modelos using default params\nbest_model = compare_models()\nprint(best_model)')




tuned_catboost = tune_model(best_model)
print(tuned_catboost)




# Increse the number of iterations (n_iter) to 35. 
# Increasing the n_iter parameter will for sure increase the training time but will 
# give a much better performance.

tuned_catboost_v1 = tune_model(best_model, n_iter = 35)
print(tuned_catboost_v1)

# you can try differents values for n_iter param




tuned_catboost_v1.get_params()




# Let's try a custom grid

# tune hyperparameters with custom_grid
params = {#'early_stopping_rounds': 15,
          'max_depth': list(range(3,10,1)),
          'learning_rate': [0.001, 0.01, 0.015, 0.02, 0.04, 0.1],
          #'n_estimators': list(range(100,300,50)),
          'iterations': [1000, 500, 1500, 800, 1100, 1200],
          }

tuned_catboost_v2 = tune_model(best_model, n_iter = 35, custom_grid = params)
print(tuned_catboost_v2)

# you can try differents values for n_iter param




tuned_catboost_v2.get_params()




#evaluate a model
evaluate_model(tuned_catboost_v1)




# Compare test data predictions and results
plot_model(tuned_catboost_v1, plot='confusion_matrix')




# predict in train dataframe
y_train_pred = predict_model(tuned_catboost_v1)

# predict the test dataframe
y_pred = predict_model(tuned_catboost_v1, data = test)




# view the predictions
display(y_train_pred[['Cover_Type', 'Label']], y_pred['Label'])




# Finalize model
final_tuned_catboost_v1 = finalize_model(tuned_catboost_v1)




# Save model
save_model(final_tuned_catboost_v1, 'final_tuned_catboost_v1_30082020'




#sample_submission
sample_submission['Cover_Type'] = y_pred['Label'].tolist()

# Lets see the head of our submission file
display(sample_submission.head())

# Analyse the % of Cover Types predicted
display(sample_submission['Cover_Type'].value_counts(normalize=True)*100)

# Save the 
file_name = '3-sub_catboost_pycaret' 
sample_submission.to_csv('{}.csv'.format(file_name), index=False)

