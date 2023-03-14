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

#!pip install xgboost 
#from xgboost import XGBClassifier
#Random Forest
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
#Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




#Loading Data into DF
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()




#PREPARE TEST DATA
x_train,x_test,y_train,y_test = train_test_split(df_train.drop(columns=['target']),df_train['target'],test_size=0.20, random_state=42)





# model XGBC Classifier
model = XGBClassifier(max_depth=4,
                           min_child_weight=1,
                           learning_rate=0.1,
                           n_estimators=1000,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)
model.fit(x_train,
          y_train)
#predict test data
predictions = model.predict(x_test)
predictions = [round(p) for p in predictions]

#Metrics
accuracy = accuracy_score(y_test,predictions)
print("X_Test Data Accuracy: %2f" % (accuracy * 100))




#RANDOM FOREST MODEL
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, 
                            random_state = 42,
                            bootstrap=True,
                            min_samples_leaf=4,
                            min_samples_split=5,
                            max_depth=100,
                            max_features='auto')
# Train the model on training data
rf.fit(x_train, y_train);
print('Finish to create the random forest!')




#Hyperparameters tuned for random forest
#k = 3
{'bootstrap': True,
 'max_depth': 100,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 5,
 'n_estimators': 200}




#SCORE Random Forest
predictions = rf.predict(x_test)
print(predictions)
predictions = [round(p) for p in predictions]
#Metrics
accuracy = accuracy_score(y_test,predictions)
print("X_Test Data Accuracy: %2f" % (accuracy * 100))




#feature importance
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.head()




#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear').fit(x_train, y_train)




#SCORE Logistic Regression
predictions = clf.predict(x_test)
predictions = [round(p) for p in predictions]
#Metrics
accuracy = accuracy_score(y_test,predictions)
print("X_Test Data Accuracy: %2f" % (accuracy * 100))




#Comparing Classifier on this dataset

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

# Create classifiers
lr = LogisticRegression(solver='liblinear')
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(x_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(x_test)
        prob_pos =             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value =         calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()




#Predict Validation data
#predictions = model.predict(df_test)
predictions = clf.predict(df_test)
# Submission
df_submission = pd.read_csv('../input/sample_submission.csv')
df_submission['target'] = predictions
df_submission.to_csv('submission.csv',index=False)




#Gaussian Naive bayes with parameter inference
get_ipython().run_line_magic('matplotlib', 'inline')
from warnings import filterwarnings
filterwarnings("ignore")
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(12345)
rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




get_ipython().system('pip install git+https://github.com/pymc-learn/pymc-learn')
import pmlearn
from pmlearn.naive_bayes import GaussianNB
print('Running on pymc-learn v{}'.format(pmlearn.__version__))




#Instantiate Gaussian Naive Bayesian Classifier based on probabilistic ML
model = GaussianNB()
model.fit(x_train,y_train)
print(model)




#Use for predicting
predictions = model.predict_proba(X_test)
model.score(X_test, y_test)

