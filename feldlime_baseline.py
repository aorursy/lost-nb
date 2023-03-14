#!/usr/bin/env python
# coding: utf-8



from os.path import join as pjoin
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns




PATH_TO_DATA = '../input/lab12-classification-problem'




data_train = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))
data_train.shape




data_train.head()




data_test = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))
data_test.shape




data_test.head()




words = data_train['Word']
labels = data_train['Label']




labels.value_counts()




def explore_words_length(data):
    word_len = data['Word'].str.len()
    print(f'Max length: {word_len.max()}')
    print(f'Mean length: {word_len.mean()}')
    print(f'Median length: {word_len.median()}')    




explore_words_length(data_train)




explore_words_length(data_test)




def label_distplots(values, labels, kde=True, hist=True):
    sns.distplot(values[labels == 1], kde=kde, hist=hist, label='Label=1', norm_hist=True)
    sns.distplot(values[labels == 0], kde=kde, hist=hist, label='Label=0', norm_hist=True)
    plt.legend();




words_len = words.str.len()
label_distplots(words_len, labels, hist=False)




is_first_letter_capital = words.str.slice(0, 1).str.isupper()
pd.crosstab(is_first_letter_capital, labels).plot(kind='bar')




is_all_letters_capital = words.str.isupper()
pd.crosstab(is_all_letters_capital, labels, margins=True, normalize=True)









from collections import Counter
concatenated_words = ''.join(words.str.lower().values)
counter = Counter(concatenated_words)
counter




is_symbol_in_word = words.str.contains('\W')
pd.crosstab(is_symbol_in_word, labels)









vowels = 'аеёиоуыэюя'
consonants = 'бвгджзйклмнпрстфхчцшщъь'




n_vowels = words.str.lower().str.count(f'[{vowels}]')
label_distplots(n_vowels, labels, kde=False)




n_consonants = words.str.lower().str.count(f'[{consonants}]')
label_distplots(n_vowels, labels, kde=False)














plt.rcParams['figure.figsize'] = 6, 4




def plot_labels_ratio_bar(values, labels):
    plt.rcParams['figure.figsize'] = 12, 5
    crosstab  = pd.crosstab(values, labels)
    crosstab['ratio'] = crosstab[1] / np.maximum(crosstab[0].values, 1)
    crosstab.query('ratio > 0', inplace=True)
    ratios = crosstab.sort_values('ratio', ascending=False)['ratio']
    ratios.plot(kind='bar')
    plt.ylabel('Labels ratio')
    plt.rcParams['figure.figsize'] = 6, 4




last_letter = words.str.slice(-1, None).str.lower()
last_letter.name = 'last_letter'
plot_labels_ratio_bar(last_letter, labels)




first_letter = words.str.slice(0, 1).str.lower()
first_letter.name = 'first_letter'
plot_labels_ratio_bar(first_letter, labels)




from scipy  import sparse

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier




def extract_features(words):
    data = pd.DataFrame()
    data['word_len'] = words.str.len()
    
    is_first_letter_capital = words.str.slice(0, 1).str.isupper()
    data['is_first_letter_capital'] = is_first_letter_capital
    
    is_symbol_in_word = words.str.contains('\W')
    data['is_symbol_in_word'] = is_symbol_in_word

    last_letter = words.str.slice(-1, None).str.lower()
    data['last_letter'] = last_letter

    letter_before_last = words.str.slice(-2, -1).str.lower()
    data['letter_before_last'] = letter_before_last

    second_letter_before_last = words.str.slice(-3, -2).str.lower()
    data['second_letter_before_last'] = second_letter_before_last
    
    third_letter_before_last = words.str.slice(-4, -3).str.lower()
    data['third_letter_before_last'] = third_letter_before_last

    first_letter = words.str.slice(0, 1).str.lower()
    data['first_letter'] = first_letter

    n_vowels = words.str.lower().str.count(f'[{vowels}]')
    data['n_vowels'] = n_vowels

    n_consonants = words.str.lower().str.count(f'[{consonants}]')
    data['n_consonants'] = n_consonants
    
    return data




def get_cat_features(X: pd.DataFrame) -> List[str]:
    return X.columns[X.dtypes == 'O'].tolist()


def label_encode(
    X: pd.DataFrame, 
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:

    X = X.copy()
    encoders = encoders or {}
    for col in get_cat_features(X):
        if col not in encoders:
            encoder = LabelEncoder().fit(X[col])
            encoders[col] = encoder
        else:
            encoder = encoders[col]
        X[col] = encoder.transform(X[col])
    return X, encoders


def one_hot_encode(
    X: pd.DataFrame, 
    encoders: Optional[Dict[str, OneHotEncoder]] = None,
) -> Tuple[sparse.csr_matrix, Dict[str, OneHotEncoder]]:
    cat_features = get_cat_features(X)
    feature_matrices = []
    encoders = encoders or {}
    for col in X.columns:
        if col in cat_features:
            if col not in encoders:
                encoder = OneHotEncoder().fit(X[[col]])
                encoders[col] = encoder
            else:
                encoder = encoders[col]
            feature_matrix = encoder.transform(X[[col]])
        else:
            feature_matrix = sparse.csr_matrix((
                X[col].values, 
                (
                    np.arange(X.shape[0], dtype=int), 
                    np.zeros(X.shape[0], dtype=int),
                ),
            ))
        feature_matrices.append(feature_matrix)
    features = sparse.hstack(feature_matrices, format='csr')
    return features, encoders  




def calc_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    pred_proba: Union[np.ndarray, pd.Series], 
    threshold: float = 0.5,
) -> Dict[str, float]:
    res = {}
    pred = np.zeros_like(pred_proba)
    pred[pred_proba > threshold] = 1
    res['accuracy'] = accuracy_score(y_true, pred)
    res['auc'] = roc_auc_score(y_true, pred_proba)
    res['f1'] = f1_score(y_true, pred)
    res['precision'] = precision_score(y_true, pred)
    res['recall'] = recall_score(y_true, pred)
    return res




X_train = extract_features(data_train['Word'])
y_train = data_train['Label']




X_train_ohe, one_hot_encoders = one_hot_encode(X_train)




X_train_le, label_encoders = label_encode(X_train)




lr = LogisticRegression(solver='liblinear', penalty='l2', random_state=1)
gscv_lr = GridSearchCV(
    estimator=lr,
    param_grid={'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    scoring='roc_auc',
    n_jobs=1,
    cv=3,
    refit=False,
    return_train_score=True,
    verbose=True,
)

gscv_lr.fit(X_train_ohe, y_train);




gscv_lr.best_params_, gscv_lr.best_score_









dt = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)
gscv_dt = GridSearchCV(
    estimator=dt,
    param_grid={'max_depth': [6, 8, 10, 12, 14], 'min_samples_leaf': [100, 200, 300]},
    scoring='roc_auc',
    n_jobs=1,
    cv=3,
    refit=False,
    return_train_score=True,
    verbose=True,
)

gscv_dt.fit(X_train_le, y_train);




gscv_dt.best_params_, gscv_dt.best_score_









gb = GradientBoostingClassifier(random_state=1)
gscv_gb = GridSearchCV(
    estimator=gb,
    param_grid={
        'n_estimators': [50, 100, 200], 
        'max_depth': [3, 6, 9],
        'min_samples_leaf': [100, 200, 300],
    },
    scoring='roc_auc',
    n_jobs=-2,
    cv=3,
    refit=False,
    return_train_score=True,
    verbose=True,
)

gscv_gb.fit(X_train_le, y_train);




gscv_gb.best_params_, gscv_gb.best_score_




cv_results = pd.DataFrame(gscv_gb.cv_results_)
cv_results[['params', 'mean_fit_time', 'mean_train_score', 'mean_test_score']]     .sort_values('mean_test_score', ascending=False)




cb = CatBoostClassifier(
    cat_features=get_cat_features(X_train),
    eval_metric='AUC',
    random_seed=1,
    nan_mode='Forbidden',
    task_type='CPU',
    verbose=False,
)


gscv_cb = GridSearchCV(
    estimator=cb,
    param_grid={
        'n_estimators': [50, 100, 150], 
        'max_depth': [5, 6, 7],
    },
    scoring='roc_auc',
    n_jobs=2,
    cv=3,
    refit=False,
    return_train_score=True,
    verbose=True,
)

gscv_cb.fit(X_train, y_train);




gscv_cb.best_params_, gscv_cb.best_score_




cv_results = pd.DataFrame(gscv_cb.cv_results_)
cv_results[['params', 'mean_fit_time', 'mean_train_score', 'mean_test_score']]     .sort_values('mean_test_score', ascending=False)






X_train_train, X_train_test, y_train_train, y_train_test = \
    train_test_split(X_train, y_train, test_size=0.3, random_state=1)


classifier = CatBoostClassifier(
    cat_features=get_cat_features(X_train),
    eval_metric='AUC',
    random_seed=1,
    nan_mode='Forbidden',
    task_type='CPU',
    verbose=True,
    n_estimators=150,
    max_depth=6,
)
classifier.fit(X_train, y_train);









X_test = extract_features(data_test['Word'])




pred_train = classifier.predict_proba(X_train)[:, 1]
calc_metrics(y_train, pred_train)




pd.DataFrame({
    'column': X_train.columns,
    'importance': classifier.feature_importances_,
}).sort_values('importance', ascending=False)









pred_test = classifier.predict_proba(X_test)[:, 1]
res = pd.DataFrame({'Id': data_test.index, 'Prediction': pred_test})
res.to_csv('res_150est_d6.csv', index=False)














pr, rec, thr = precision_recall_curve(y_train, pred_train)
f1 = 2 * (pr * rec) / (pr + rec)
plt.plot(thr, f1[:-1])
plt.xlabel('threshold')
plt.ylabel('F1')
plt.grid()




best_thr = thr[f1.argmax() - 1]
best_thr, f1.max()






