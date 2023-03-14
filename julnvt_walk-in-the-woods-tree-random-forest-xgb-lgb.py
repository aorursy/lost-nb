#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

import gc

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import make_scorer




rawdata_train = pd.read_csv("../input/train.csv", sep = ',',na_values = -1)
rawdata_test = pd.read_csv("../input/test.csv", sep = ',',na_values = -1)




def describe_missing_values(df):
    na_percent = {}
    N = df.shape[0]
    for column in df:
        na_percent[column] = df[column].isnull().sum() * 100 / N

    na_percent = dict(filter(lambda x: x[1] != 0, na_percent.items()))
    plt.bar(range(len(na_percent)), na_percent.values())
    plt.ylabel('Percent')
    plt.xticks(range(len(na_percent)), na_percent.keys(), rotation='vertical')
    plt.show()




print("Missing values for train set")
describe_missing_values(rawdata_train)
print("Missing values for test set")
describe_missing_values(rawdata_test)




X = rawdata_train.drop({'target','id','ps_car_03_cat','ps_car_05_cat'},axis=1)
Y = rawdata_train['target']
X_test = rawdata_test.drop({'id','ps_car_03_cat','ps_car_05_cat'},axis=1)

cat_cols = [col for col in X.columns if 'cat' in col]
bin_cols = [col for col in X.columns if 'bin' in col]
con_cols = [col for col in X.columns if col not in bin_cols + cat_cols]

for col in cat_cols:
    X[col].fillna(value=X[col].mode()[0], inplace=True)
    X_test[col].fillna(value=X_test[col].mode()[0], inplace=True)
    
for col in bin_cols:
    X[col].fillna(value=X[col].mode()[0], inplace=True)
    X_test[col].fillna(value=X_test[col].mode()[0], inplace=True)
    
for col in con_cols:
    X[col].fillna(value=X[col].mean(), inplace=True)
    X_test[col].fillna(value=X_test[col].mean(), inplace=True)




def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized_score(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

score_gini = make_scorer(gini_normalized_score, greater_is_better=True, needs_threshold = True)




depth_gini = []
for i in range(3,15):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    # Perform 5-fold cross validation
    scores_gini = cross_val_score(clf, X, Y, cv=5, scoring = score_gini)
    depth_gini.append((i,scores_gini.mean()))
plt.plot(*zip(*depth_gini))
plt.xlabel('tree depth')
plt.ylabel('cv gini score')
plt.show()




parameters = {'max_depth': np.arange(3,15)}
clf = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, scoring = score_gini, cv = 5)
clf.fit(X, Y)
print("Best parameters set found on development set:")
print()
print(clf.best_estimator_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() / 2, params))
print()




clf = tree.DecisionTreeClassifier(max_depth=8)
clf = clf.fit(X,Y)
Y_pred_clf = clf.predict(X)
Y_pred_proba_clf = clf.predict_proba(X)
Y_pred_clf = clf.predict(X)




dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph




depth_gini = []
for i in range(3,15):
    rf = RandomForestClassifier(max_depth=i)
    # Perform 5-fold cross validation
    scores_gini = cross_val_score(rf, X, Y, cv=5, scoring = score_gini)
    depth_gini.append((i,scores_gini.mean()))
plt.plot(*zip(*depth_gini))
plt.xlabel('tree depth')
plt.ylabel('cv gini score')
plt.show()




rf = RandomForestClassifier(max_depth=8)
rf = rf.fit(X,Y)
Y_pred_rf = rf.predict(X)
Y_pred_proba_rf = rf.predict_proba(X)
Y_pred_rf = rf.predict(X)




importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()




confusion_matrix(Y,Y_pred_clf)




fpr_clf, tpr_clf, thresholds_clf = roc_curve(Y,Y_pred_proba_clf[:,1],pos_label = 1)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y,Y_pred_proba_rf[:,1],pos_label = 1)
plt.plot(fpr_clf,tpr_clf)
plt.plot(fpr_rf,tpr_rf)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
auc_clf = np.trapz(tpr_clf,fpr_clf)
auc_rf = np.trapz(tpr_rf,fpr_rf)
print(auc_clf)
print(auc_rf)




X = rawdata_train.drop({'target','id'},axis=1)
Y = rawdata_train['target']
X_test = rawdata_test




# Create an XGBoost-compatible metric from Gini
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized_score(labels, preds)
    return [('gini', gini_score)]




params = {'eta': 0.2,
          'max_depth': 4,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'silent': True}

features = X.columns
submission = X_test['id'].to_frame()
submission['target']=0

kfold = 3
skf = StratifiedKFold(n_splits=kfold)
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    Y_train, Y_valid = Y.loc[train_index], Y.loc[test_index]
    d_train = xgb.DMatrix(X_train, Y_train) 
    d_valid = xgb.DMatrix(X_valid, Y_valid) 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=100, 
                        feval=gini_xgb, maximize=True, verbose_eval=100)
    submission['target'] += xgb_model.predict(xgb.DMatrix(X_test[features]), 
                        ntree_limit=xgb_model.best_ntree_limit+50) / (kfold)
gc.collect()
submission.head(2)




parameters = {'max_depth': np.arange(3,7),
            'learning_rate': [0.2],
             'n_estimators': [20,100]}

clf = GridSearchCV(estimator = xgb.XGBClassifier(silent=True), param_grid = parameters, scoring = score_gini, cv = 3, verbose = 10, n_jobs = -1)
clf.fit(X, Y)




def gini_lgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized_score(labels, preds)
    return [('gini', gini_score, True)]




params = {'learning_rate' : 0.2, 'max_depth':6, 'max_bin':10,  'objective': 'binary', 
        'metric': 'auc'}

features = X.columns
submission = X_test['id'].to_frame()
submission['target']=0

kfold = 5
skf = StratifiedKFold(n_splits=kfold)
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    Y_train, Y_valid = Y.loc[train_index], Y.loc[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=Y_train), 400, 
                  lgb.Dataset(X_valid, label=Y_valid), verbose_eval=100, 
                  feval=gini_lgb, early_stopping_rounds=50)
    submission['target'] += lgb_model.predict(X_test[features], 
                        num_iteration=lgb_model.best_iteration) / (kfold)
gc.collect()
submission.head(2)




submission.to_csv("./submission.csv", index=False, float_format='%.5f')

