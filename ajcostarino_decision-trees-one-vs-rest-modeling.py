#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Graphing
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df = pd.read_csv('../input/liverpool-ion-switching-ds/train.csv')



PERIODS = [1 , 2]




PERIODS = [1]

def _signal_shift(signal, periods):
    '''Returns signal shifted for a set number of periods.
    '''
    return signal.shift(periods=periods)


def signal_shifts(df, signal):
    '''Calculates all signal shifts positive (forward) and negative (backwards)
    given the predefined shift periods.
    '''
    for period in PERIODS:
        neg = period
        pos = -period
        df[f'{signal}_shift_pos_{period}'] = _signal_shift(df[signal], pos)
        df[f'{signal}_shift_neg_{period}'] = _signal_shift(df[signal], neg)

    return df


def signal_shift_perc(df, signal):
    '''Calculates the percentage or ratio of the shifted signal relative to
    the current signal.
    '''
    for period in PERIODS:
        df[f'{signal}_shift_pos_{period}_perc'] =             df[f'{signal}_shift_pos_{period}'] / df[signal]

        df[f'{signal}_shift_neg_{period}_perc'] =             df[f'{signal}_shift_neg_{period}'] / df[signal]

    return df


def single_decision_tree(X, y):
    '''Trains single decision tree, prints the F1 macro scores and
    and returns cross validaiton scores in one fold across dataframe.'''
    kf = KFold(n_splits=5, random_state=348, shuffle=True)
    
    # Keep track of the predictions across each fold
    cv            = pd.DataFrame() 
    cv['actual']  = pd.Series(y.values)
    cv['predict'] = pd.Series()
    fold          = 1
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Build basic decision tree with low depth.
        dtc = DecisionTreeClassifier(max_depth=8, class_weight='balanced')
        
        # Fit and predict on the test fold
        dtc.fit(X_train, y_train)
        predict = dtc.predict(X_test)
        
        # Calculate f1 macro on single fold and display
        fold_cv = pd.DataFrame()
        fold_cv['actual']  = pd.Series(y_test.values)
        fold_cv['predict'] = pd.Series(predict)
        
        print(f'Fold {fold} F1 Macro: ', f1_score(fold_cv['actual'], fold_cv['predict'], average='weighted'))
        
        cv.loc[test_index, 'predict'] = predict
        
        fold += 1
    
    print(f'Total F1 Macro: ', f1_score(cv['actual'], cv['predict'], average='weighted'))
    
    return cv


def one_versus_all(X, y):
    '''Trains single decision tree for each class, prints the F1 macro scores and
    returns cross validation scores. Returns a out-of-fold scores and individual
    probability per class.'''
    kf = KFold(n_splits=5, random_state=348, shuffle=True)
    
    # Keep track of the predictions across each fold
    cv            = pd.DataFrame() 
    cv['actual']  = pd.Series(y.values)
    cv['predict'] = pd.Series()
    for class_ in range(0, 11):
        cv[f'proba_class_{class_}'] = pd.Series()
        
    fold          = 1
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Build the one versus all classifiers (decision tree)
        dtc = DecisionTreeClassifier(max_depth=8,
                                     class_weight='balanced')
        clf = OneVsRestClassifier(dtc).fit(X_train, y_train)
        
        # Fit and predict on the test fold
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        predict_proba = clf.predict_proba(X_test)
        
        # Calculate f1 macro on single fold and display
        fold_cv = pd.DataFrame()
        fold_cv['actual']  = pd.Series(y_test.values)
        fold_cv['predict'] = pd.Series(predict)
        
        print(f'Fold {fold} F1 Macro: ', f1_score(fold_cv['actual'], fold_cv['predict'], average='weighted'))
        
        for class_ in clf.classes_:
            cv.loc[test_index, f'proba_class_{class_}'] = predict_proba[:, class_]
            cv.loc[test_index, 'predict'] = predict
        
        fold += 1
    
    print(f'Total F1 Macro: ', f1_score(cv['actual'], cv['predict'], average='weighted'))
    
    return cv




## Take a look at the columns we are working with
for col in df.columns:
    print(col)




cv = single_decision_tree(df.drop(['open_channels'], axis = 1), df['open_channels'])




one_v_all_oof = one_versus_all(df.drop(['open_channels'], axis = 1), df['open_channels'])




plt.figure(figsize=(24,8))

aggs = {}

for class_ in range(0,11):
    aggs[f'proba_class_{class_}'] = 'mean'

sns.heatmap(one_v_all_oof[one_v_all_oof['actual'] != one_v_all_oof['predict']].groupby(['actual']).agg(aggs), cmap='Blues', annot=True)




misclassified = one_v_all_oof[one_v_all_oof['actual'] != one_v_all_oof['predict']]
misclassified_classes = [0, 1, 2, 3, 4, 7]

f, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True)
order = [x for x in range(0, 11)]
axes_list = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
sns.set_style("whitegrid")
sns.despine(left = True)

for class_, axs in zip(misclassified_classes, axes_list):
    mis_graph = misclassified[misclassified['actual'] == class_]                              .groupby(['predict'], as_index=False)                              .agg({'proba_class_0' : 'count'})                              .rename(columns={'proba_class_0' : 'count'})
    
    sns.barplot(mis_graph['predict'], mis_graph['count'], ax=axs, order=order,palette='Blues').set_title(f'Misclassifications for class {class_}')




orig_misclassified_index = misclassified.index.copy() 




df = signal_shifts(df, 'signal_sans_drift_avg_center')
df.columns




one_v_all_oof = one_versus_all(df.drop(['open_channels'], axis = 1).fillna(0), df['open_channels'])




plt.figure(figsize=(24,8))

aggs = {}

for class_ in range(0,11):
    aggs[f'proba_class_{class_}'] = 'mean'

sns.heatmap(one_v_all_oof[one_v_all_oof.isin(orig_misclassified_index)].groupby(['actual']).agg(aggs), cmap='Blues', annot=True)






