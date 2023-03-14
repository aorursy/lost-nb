#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')




data




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split




class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing_intencity=10):
        self.smoothing_intencity = smoothing_intencity
        self.value_means = None
        self.total_mean = None

    def fit(self, x, y):
        self.total_mean = y.mean()
        self.value_means = []
        for column in x.values.T:
            column_value_means = {}
            for value in np.unique(column):
                value_mask = column == value
                column_value_means[value] = (y[value_mask].sum() + self.total_mean * self.smoothing_intencity) /                         (np.sum(value_mask) + self.smoothing_intencity)
            self.value_means.append(column_value_means)
        return self
    
    def transform(self, x):
        encoding = []
        for column, value_mean_dict in zip(x.values.T, self.value_means):
            column_encoding = np.zeros(column.shape, dtype=np.float64)
            unseen = np.ones(column.shape, dtype=np.bool)
            for value, mean_target in value_mean_dict.items():
                value_mask = column == value
                column_encoding[value_mask] = mean_target
                unseen &= ~value_mask
            column_encoding[unseen] = self.total_mean
            encoding.append(column_encoding)
        return np.vstack(encoding).T




from itertools import chain


class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods
        
    def fit(self, *args):
        return self
    
    def transform(self, x):
        phases = (x.values / self.periods * 2 * np.pi)
        return np.vstack(list(chain((np.sin(i), np.cos(i)) for i in phases.T))).T




BIN_FEATURES = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
bin_transformer = OneHotEncoder(drop='first')
# Бинарные факторы кодируются флагами. OneHotEncoder удобно использовать, чтобы выставлять флаги для разных типов входных данных




NOM_LOW_CARD_FEATURES = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
nom_low_card_transformer = OneHotEncoder(drop='first')
# Факторы с небольшим множеством значений можно кодировать one-hot




NOM_HIGH_CARD_FEATURES = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
nom_high_card_transformer = MeanTargetEncoder(smoothing_intencity=20)
# При большом количестве возможных значений фактора по отдельному значению может не быть достаточно примеров. 




# Постараемся сохранить порядок в порядковых факторах
ORDINAL_ALPHABET_FEATURES = ['ord_0', 'ord_3', 'ord_4', 'ord_5']
ordinal_alphabet_transformer = OrdinalEncoder()
# Будем считать, что алфавитный порядок, выбираемый OrdinalEncoding по умолчанию, правильный

ORDINAL_ENUM_FEATURES = ['ord_1', 'ord_2']
ordibal_enum_transformer = OrdinalEncoder(categories=[
    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
])




CYCLIC_FEATURES = ['day', 'month']
cyclic_transformer = CyclicEncoder([7, 12])
# Кодировка синусом и косинусом позволяет сохранить близость не только между последовательными значениями, 
# но и между первыми и последними.




x, y = data.drop('target', axis=1), data['target'].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2)




feature_transformer = ColumnTransformer([
    ('bin', bin_transformer, BIN_FEATURES), 
    ('nom_low_card', nom_low_card_transformer, NOM_LOW_CARD_FEATURES),
    ('nom_high_card', nom_high_card_transformer, NOM_HIGH_CARD_FEATURES),
    ('ord_alphabet', ordinal_alphabet_transformer, ORDINAL_ALPHABET_FEATURES),
    ('ord_enum', ordibal_enum_transformer, ORDINAL_ENUM_FEATURES),
    ('cyclic', cyclic_transformer, CYCLIC_FEATURES)
])
x_train_preproc = feature_transformer.fit_transform(x_train, y_train)
x_val_preproc = feature_transformer.transform(x_val)




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score




lr_model = LogisticRegression(C=.1).fit(x_train_preproc, y_train)
pred_lr_val = lr_model.predict_proba(x_val_preproc)[:, 1]
roc_auc_score(y_val, pred_lr_val)




from sklearn.svm import SVC

svm_model = SVC(probability=True, C=10).fit(x_train_preproc[:20000], y_train[:20000])
pred_svm_val = svm_model.predict_proba(x_val_preproc)[:, 1]
roc_auc_score(y_val, pred_svm_val)




from xgboost import XGBClassifier

xgb_model = XGBClassifier(min_child_weight=2000, alpha=10).fit(x_train_preproc, y_train)
pred_xgb_val = xgb_model.predict_proba(x_val_preproc)[:, 1]
roc_auc_score(y_val, pred_xgb_val)




x_test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')




x_test_preproc = feature_transformer.transform(x_test)
pred_xgb_test = xgb_model.predict_proba(x_test_preproc)[:, 1]




submission = pd.DataFrame({'id': x_test['id'], 'target': pred_xgb_test})




submission.to_csv('submission.csv', index=False)

