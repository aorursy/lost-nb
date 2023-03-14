#!/usr/bin/env python
# coding: utf-8



import time
notebookstart = time.time()
notebookstart




import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)




import platform
import sys
import importlib
import multiprocessing
import random




import numpy as np
import pandas as pd

random.seed(321)
np.random.seed(321)

pd.options.display.max_columns = 9999




import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rc('figure', figsize=(15, 12))
plt.figure(figsize=(15, 12))
plt.rcParams['figure.facecolor'] = 'azure'
mpl.style.use('seaborn')
plt.style.use('seaborn')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')




import seaborn as sns

sns.set(rc={'figure.figsize': (15, 12)})
sns.set(context='notebook', style='darkgrid', font='sans-serif',
        font_scale=1.1, rc={'figure.facecolor': 'azure',
        'axes.facecolor': 'azure', 'grid.color': 'steelblue'})
sns.color_palette(mcolors.TABLEAU_COLORS);




import missingno as msno




import scikitplot as skplt




import sklearn

from sklearn.model_selection import train_test_split,     cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit,     StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler,     RobustScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelBinarizer, label_binarize

from sklearn.metrics import accuracy_score, confusion_matrix,     classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score,     precision_recall_fscore_support

from sklearn.utils import shuffle, resample
from sklearn.base import BaseEstimator, ClassifierMixin




from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier




import xgboost as xgb
from xgboost import XGBClassifier




import lightgbm as lgbm
from lightgbm import LGBMClassifier




import catboost
from catboost import CatBoostClassifier




import skopt
from skopt import BayesSearchCV




import imblearn
from imblearn.over_sampling import SMOTE, SMOTENC




print('Operating system version........', platform.platform())
print('Python version is............... %s.%s.%s' % sys.version_info[:3])
print('scikit-learn version is.........', sklearn.__version__)
print('pandas version is...............', pd.__version__)
print('numpy version is................', np.__version__)
print('matplotlib version is...........', mpl.__version__)
print('seaborn version is..............', sns.__version__)
print('scikit-plot version is..........', skplt.__version__)
print('missingno version is............', msno.__version__)
print('xgboost version is..............', xgb.__version__)
print('catboost version is.............', catboost.__version__)
print('lightgbm version is.............', lgbm.__version__)
print('scikit-optimize version is......', skopt.__version__)
print('imblearn version is.............', imblearn.__version__)




def getDatasetInformation(csv_filepath, is_corr_required=True):
    """
    Read CSV (comma-separated) file into DataFrame
    
    Returns,
    - DataFrame
    - DataFrame's shape
    - DataFrame's data types
    - DataFrame's describe
    - DataFrame's sorted unique value count
    - DataFrame's missing or NULL value count
    - DataFrame's correlation between numerical columns
    """

    dataset_tmp = pd.read_csv(csv_filepath)

    dataset_tmp_shape = pd.DataFrame(list(dataset_tmp.shape),
            index=['No of Rows', 'No of Columns'], columns=['Total'])
    dataset_tmp_shape = dataset_tmp_shape.reset_index()

    dataset_tmp_dtypes = dataset_tmp.dtypes.reset_index()
    dataset_tmp_dtypes.columns = ['Column Names', 'Column Data Types']

    dataset_tmp_desc = pd.DataFrame(dataset_tmp.describe())
    dataset_tmp_desc = dataset_tmp_desc.transpose()

    dataset_tmp_unique = dataset_tmp.nunique().reset_index()
    dataset_tmp_unique.columns = ['Column Name', 'Unique Value(s) Count'
                                  ]

    dataset_tmp_missing = dataset_tmp.isnull().sum(axis=0).reset_index()
    dataset_tmp_missing.columns = ['Column Names',
                                   'NULL value count per Column']
    dataset_tmp_missing =         dataset_tmp_missing.sort_values(by='NULL value count per Column'
            , ascending=False)

    if is_corr_required:
        dataset_tmp_corr = dataset_tmp.corr(method='spearman')
    else:
        dataset_tmp_corr = pd.DataFrame()

    return [
        dataset_tmp,
        dataset_tmp_shape,
        dataset_tmp_dtypes,
        dataset_tmp_desc,
        dataset_tmp_unique,
        dataset_tmp_missing,
        dataset_tmp_corr,
        ]




def getHighlyCorrelatedColumns(dataset, NoOfCols=6):
    df_corr = dataset.corr()

    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones

    df_corr *= np.tri(k=-1, *df_corr.values.shape).T
    df_corr = df_corr.stack()
    df_corr =         df_corr.reindex(df_corr.abs().sort_values(ascending=False).index).reset_index()
    return df_corr.head(NoOfCols)




def createFeatureEngineeredColumns(dataset):
    dataset_tmp = pd.DataFrame()

    dataset_tmp['CountOfZeroValues'] = (dataset == 0).sum(axis=1)
    dataset_tmp['CountOfNonZeroValues'] = (dataset != 0).sum(axis=1)

    weight = ((dataset != 0).sum() / len(dataset)).values
    dataset_tmp['WeightedCount'] = (dataset * weight).sum(axis=1)

    dataset_tmp['SumOfValues'] = dataset.sum(axis=1)

    dataset_tmp['VarianceOfValues'] = dataset.var(axis=1)
    dataset_tmp['MedianOfValues'] = dataset.median(axis=1)
    dataset_tmp['MeanOfValues'] = dataset.mean(axis=1)
    dataset_tmp['StandardDeviationOfValues'] = dataset.std(axis=1)
    #dataset_tmp['ModeOfValues'] = dataset.mode(axis=1)
    dataset_tmp['SkewOfValues'] = dataset.skew(axis=1)
    dataset_tmp['KurtosisOfValues'] = dataset.kurtosis(axis=1)

    dataset_tmp['MaxOfValues'] = dataset.max(axis=1)
    dataset_tmp['MinOfValues'] = dataset.min(axis=1)
    dataset_tmp['DiffOfMinMaxOfValues'] =         np.subtract(dataset_tmp['MaxOfValues'],
                    dataset_tmp['MinOfValues'])

    dataset_tmp['QuantilePointFiveOfValues'] = dataset[dataset
            > 0].quantile(0.5, axis=1)

    dataset = pd.concat([dataset, dataset_tmp], axis=1)

    return dataset




def getZeroStdColumns(dataset):
    columnsWithZeroStd = dataset.columns[dataset.std() == 0].tolist()
    return columnsWithZeroStd




def getUniqueValueColumns(dataset, valueToCheck=0):
    columnsWithUniqueValue = dataset.columns[dataset.nunique()
            == valueToCheck].tolist()
    return columnsWithUniqueValue




def plotCategoricalVariableDistributionGraph(target_value, title='', xticksrotation=0):
    tmp_count = target_value.value_counts()
    
    fig=plt.figure()
    fig.suptitle(title, fontsize=18)
    
    ax1=fig.add_subplot(221)
    sns.pointplot(x=tmp_count.index, y=tmp_count, ax=ax1)
    ax1.set_title('Distribution Graph')
    plt.xticks(rotation=xticksrotation)
    
    ax2=fig.add_subplot(222)
    sns.barplot(x=tmp_count.index, y=tmp_count, ax=ax2)
    ax2.set_title('Distribution Graph - Bar')
    plt.xticks(rotation=xticksrotation)
    
    ax3=fig.add_subplot(212)
    ax3.pie(tmp_count, labels=tmp_count.index, autopct="%1.1f%%", shadow=True, startangle=195)
    ax3.axis('equal')
    ax3.set_title('Distribution Graph - Pie')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.show()




def plot_distplot(dataset):
    colors = mcolors.TABLEAU_COLORS

    dataset_fordist = dataset.select_dtypes([np.int, np.float])
    number_of_subplots = len(dataset_fordist.columns)
    number_of_columns = 3

    number_of_rows = number_of_subplots // number_of_columns
    number_of_rows += number_of_subplots % number_of_columns

    postion = range(1, number_of_subplots + 1)

    fig = plt.figure(1)
    for k in range(number_of_subplots):
        ax = fig.add_subplot(number_of_rows, number_of_columns,
                             postion[k])
        sns.distplot(dataset_fordist.iloc[:, k],
                     color=random.choice(list(colors.keys())), ax=ax)
    fig.tight_layout()
    plt.show()




def convertIntFloatToInt(dictObj):
    for (k, v) in dictObj.items():
        if float('Inf') == v:
            pass
        elif int(v) == v and isinstance(v, float):
            dictObj[k] = int(v)
    return dictObj




(
    dataset_sctp_train,
    df_train_shape,
    df_train_dtypes,
    df_train_describe,
    df_train_unique,
    df_train_missing,
    df_train_corr,
    ) = getDatasetInformation('../input/train.csv', False)

(
    dataset_sctp_test,
    df_test_shape,
    df_test_dtypes,
    df_test_describe,
    df_test_unique,
    df_test_missing,
    df_test_corr,
    ) = getDatasetInformation('../input/test.csv', False)




dataset_sctp_train.head()




df_train_shape




df_train_dtypes




df_train_describe




df_train_unique




df_train_missing




msno.matrix(dataset_sctp_train, color=(33 / 255, 102 / 255, 172 / 255));




dataset_sctp_test.head()




df_test_shape




df_test_dtypes




df_test_describe




df_test_unique




df_test_missing




msno.matrix(dataset_sctp_test, color=(33 / 255, 102 / 255, 172 / 255));




del(df_train_shape, df_train_dtypes, df_train_describe, df_train_unique, df_train_missing, df_train_corr)
del(df_test_shape, df_test_dtypes, df_test_describe, df_test_unique, df_test_missing, df_test_corr)




plot_distplot(dataset_sctp_train.iloc[:, 2:29])




plot_distplot(dataset_sctp_train.iloc[:, 29:56])




plot_distplot(dataset_sctp_train.iloc[:, 56:83])




plot_distplot(dataset_sctp_train.iloc[:, 83:110])




plot_distplot(dataset_sctp_train.iloc[:, 110:137])




plot_distplot(dataset_sctp_train.iloc[:, 137:164])




plot_distplot(dataset_sctp_train.iloc[:, 164:191])




plot_distplot(dataset_sctp_train.iloc[:, 191:204])




dataset_sctp_train.target.unique()




dataset_sctp_train.target.value_counts()




plotCategoricalVariableDistributionGraph(dataset_sctp_train.target, 'Target (feature) - Distribution', xticksrotation=90)




dataset_sctp_train_majority = dataset_sctp_train[dataset_sctp_train.target==0]
dataset_sctp_train_minority = dataset_sctp_train[dataset_sctp_train.target==1]

dataset_sctp_train_minority_upsampled = resample(dataset_sctp_train_minority, replace=True, n_samples=100000,)

dataset_sctp_train_upsampled = pd.concat([dataset_sctp_train_majority, dataset_sctp_train_minority_upsampled])
dataset_sctp_train_upsampled.target.value_counts()




plotCategoricalVariableDistributionGraph(dataset_sctp_train_upsampled.target, 'Target (feature) - Distribution', xticksrotation=90)




#y=dataset_sctp_train['target']
#X=dataset_sctp_train.drop(['ID_code', 'target'], axis=1)

y=dataset_sctp_train_upsampled['target']
X=dataset_sctp_train_upsampled.drop(['ID_code', 'target'], axis=1)

dataset_sctp_test_ID_code = dataset_sctp_test['ID_code']
dataset_sctp_test.drop(['ID_code'], axis=1, inplace=True)

(X.shape, y.shape, dataset_sctp_test.shape)

df_train_corr = X.corr(method='spearman')
df_train_corr#mask = np.zeros_like(df_train_corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    df_train_corr,
    cmap='rainbow',
    annot=False,
    fmt='.2f',
    center=0,
    square=False,
    linewidths=.75,
    #mask=mask,
    );
plt.title('Correlation Matrix', fontsize=18)
plt.show()


getHighlyCorrelatedColumns(X, 20)

upper = df_train_corr.where(np.triu(np.ones(df_train_corr.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f'Columns to drop from Train and Test datasets are {to_drop}.')

X.drop(columns=to_drop, axis=1, inplace=True)
dataset_sctp_test.drop(columns=to_drop, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)


X=createFeatureEngineeredColumns(X)




dataset_sctp_test=createFeatureEngineeredColumns(dataset_sctp_test)




(X.shape, dataset_sctp_test.shape)




columnsWithZeroStdToRemove = getZeroStdColumns(X)
print(f'Columns with Zero STD to drop from Train and Test dataset(s) are {columnsWithZeroStdToRemove}.')

X.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)
dataset_sctp_test.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)




X_columns_one_unique_value = getUniqueValueColumns(X, 1)
print(f'Columns with only 1 as value to drop from Train and Test datasets are {X_columns_one_unique_value}.')

X.drop(X_columns_one_unique_value, axis=1, inplace=True)
dataset_sctp_test.drop(X_columns_one_unique_value, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)




X_columns_zero_unique_value = getUniqueValueColumns(X, 0)
print(f'Columns with only 0 as value to drop from Train and Test datasets are {X_columns_zero_unique_value}.')

X.drop(X_columns_zero_unique_value, axis=1, inplace=True)
dataset_sctp_test.drop(X_columns_zero_unique_value, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)

del(df_train_corr)


n_cpus_avaliable = multiprocessing.cpu_count()

print(f'We\'ve got {n_cpus_avaliable} cpus to work with.')




model_scores = pd.DataFrame(columns=['Classification_Type', 'Model_Name',
                            'Accuracy_Score'])




(X_train, X_test, y_train, y_test) = train_test_split(X, y,
        test_size=0.20)

def lgbm_status_print_twoclass(optimal_result):
    all_models = pd.DataFrame(lgbm_bayes_cv_tuner_twoclass.cv_results_)
    best_params = pd.Series(lgbm_bayes_cv_tuner_twoclass.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(len(all_models),
            np.round(lgbm_bayes_cv_tuner_twoclass.best_score_, 4),
            lgbm_bayes_cv_tuner_twoclass.best_params_)
         )


lgbm_bayes_cv_tuner_twoclass = BayesSearchCV(
    estimator=lgbm.LGBMClassifier(n_jobs=n_cpus_avaliable,
                                  objective='binary',
                                  metric='binary_logloss',
                                  #is_unbalance=True,
                                  class_weight='balanced',
                                  silent=True),
    search_spaces={
        #'boosting_type': ['gbdt', 'dart', 'rf'],
        'num_leaves': (1, 50),
        'max_depth': (1, 40),
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'n_estimators': (100, 300),
        'min_split_gain': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0.01, 1.0, 'uniform'),
        'min_child_samples': (1, 10),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (1, 50),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        #'bagging_fraction': (0.01, 1.0, 'uniform'),     # OR subsample
        #'feature_fraction': (0.01, 1.0, 'uniform'),     # OR colsample_bytree
        },
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=13, shuffle=True),
    n_jobs=n_cpus_avaliable,
    n_iter=9,
    refit=True,
    verbose=0,
    )


lgbm_result_twoclass = lgbm_bayes_cv_tuner_twoclass.fit(X, y, 
        callback=lgbm_status_print_twoclass)

lgbm_twoclass_model = lgbm_result_twoclass.best_estimator_

print(lgbm_twoclass_model)y_pred = lgbm_twoclass_model.predict(X_test)
y_pred_proba = lgbm_twoclass_model.predict_proba(X_test)

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the LightGBM (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': lgbm_twoclass_model.__class__.__name__ + ' - BayesSearchCV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (LightGBM Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );
print('\n')

skplt.metrics.plot_precision_recall(y_test, y_pred_proba,
                                    title='Binomial Classification (LightGBM Precision-Recall Curve)',
                                   );
print('\n')

skplt.metrics.plot_roc(y_test, y_pred_proba,
                       title='Binomial Classification (LightGBM ROC Curves)',
                      );feature_importance = pd.DataFrame({'imp': lgbm_twoclass_model.feature_importances_, 'col': X_train.columns})
feature_importance = feature_importance.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
feature_importance.plot(kind='barh', x='col', y='imp', color=mcolors.TABLEAU_COLORS);
plt.title('Binomial Classification (LightGBM - Feature Importance(s))', fontsize=18)
plt.show()


dataset_sctp_train_lgbm = lgbm.Dataset(data=X, label=y)
dataset_sctp_test_lgbm = lgbm.Dataset(data=X_test, label=y_test)
watchlist = [dataset_sctp_train_lgbm, dataset_sctp_test_lgbm]

evaluation_results = {}

lgbmparams = {
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',
    'colsample_bytree': 0.8220008732467731,
    'importance_type': 'split',
    'learning_rate': 0.6161285857013439,
    'max_depth': 33,
    'metric': 'binary_logloss',
    'min_child_samples': 9,
    'min_child_weight': 0.7823020109077508,
    'min_split_gain': 0.7211200712627109,
    'n_estimators': 295,
    'n_jobs': n_cpus_avaliable,
    'num_leaves': 41,
    'objective': 'binary',
    'reg_alpha': 0.0,
    'reg_lambda': 0.002053077897015527,
    'silent': False,
    'subsample': 0.6820394902275151,
    'subsample_for_bin': 200000,
    'subsample_freq': 8,
    }

lgbm_twoclass_model = lgbm.train(
    lgbmparams,
    train_set=dataset_sctp_train_lgbm,
    num_boost_round=5000,
    valid_sets=watchlist,
    early_stopping_rounds=120,
    evals_result=evaluation_results,
    verbose_eval=100,
    )




y_pred_proba = lgbm_twoclass_model.predict(X_test)
y_pred = [round(value) for value in y_pred_proba]

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the LightGBM (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': lgbm_twoclass_model.__class__.__name__ + ' - BayesSearchCV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (LightGBM Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );




ax = lgbm.plot_metric(evaluation_results)
plt.show()




ax = lgbm.plot_importance(lgbm_twoclass_model, max_num_features=30, color=mcolors.TABLEAU_COLORS)
plt.show()

def status_print_twoclass(optimal_result):
    all_models = pd.DataFrame(bayes_cv_tuner_twoclass.cv_results_)
    best_params = pd.Series(bayes_cv_tuner_twoclass.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(len(all_models), 
            np.round(bayes_cv_tuner_twoclass.best_score_, 4), 
            bayes_cv_tuner_twoclass.best_params_)
         )


bayes_cv_tuner_twoclass = BayesSearchCV(
    estimator=xgb.XGBClassifier(
        n_jobs=n_cpus_avaliable,
        objective='binary:logistic',
        eval_metric='auc',
        silent=1,
        tree_method='approx',
        device='cpu',
        ),
    search_spaces={
        #'booster': ['gbtree', 'dart'],
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_delta_step': (0, 20),
        'max_depth': (0, 40),
        'min_child_weight': (0, 20),
        'n_estimators': (200, 300),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        },
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=13, shuffle=True),
    n_jobs=n_cpus_avaliable,
    n_iter=9,
    refit=True,
    verbose=0,
    )

result_twoclass = bayes_cv_tuner_twoclass.fit(X, y,
        callback=status_print_twoclass)

xgb_twoclass_model = result_twoclass.best_estimator_

print(xgb_twoclass_model)y_pred = xgb_twoclass_model.predict(X_test)
y_pred_proba = xgb_twoclass_model.predict_proba(X_test)

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the XGBoost (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': xgb_twoclass_model.__class__.__name__ + ' - BayesSearchCV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (XGBoost Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );
print('\n')

skplt.metrics.plot_precision_recall(y_test, y_pred_proba,
                                    title='Binomial Classification (XGBoost Precision-Recall Curve)',
                                   );
print('\n')

skplt.metrics.plot_roc(y_test, y_pred_proba,
                       title='Binomial Classification (XGBoost ROC Curves)',
                      );feature_importance = pd.DataFrame({'imp': xgb_twoclass_model.feature_importances_, 'col': X_train.columns})
feature_importance = feature_importance.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
feature_importance.plot(kind='barh', x='col', y='imp', color=mcolors.TABLEAU_COLORS);
plt.title('Binomial Classification (XGBoost - Feature Importance(s))', fontsize=18)
plt.show()


dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, 'train'), (dtest, 'valid')]

xgbparams = {
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 0.48973134715974026,
    'colsample_bytree': 0.46131821596707184,
    'device': 'cpu',
    'eval_metric': 'auc',
    'gamma': 8.0762512124703e-06,
    'learning_rate': 0.054722068464825926,
    'max_delta_step': 3,
    'max_depth': 29,
    'min_child_weight': 14,
    'missing': 'None',
    'n_estimators': 274,
    'n_jobs': n_cpus_avaliable,
    'objective': 'binary:logistic',
    'reg_alpha': 0,
    'reg_lambda': 2.1021696940800796,
    'scale_pos_weight': 43.8858626195784,
    'silent': True,
    'subsample': 0.8113392946402368,
    'tree_method': 'approx',
    }
xgb_twoclass_model = xgb.train(
    xgbparams,
    dtrain,
    5000,
    watchlist,
    verbose_eval=200,
    early_stopping_rounds=100,
    )




y_pred_proba = xgb_twoclass_model.predict(dtest)
y_pred = [round(value) for value in y_pred_proba]

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the XGBoost (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': xgb_twoclass_model.__class__.__name__ + ' - CV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (XGBoost Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );




model_scores.Accuracy_Score = model_scores.Accuracy_Score.astype('float32')
model_scores




sns.barplot(x='Model_Name', y='Accuracy_Score', data=model_scores);
plt.xticks(rotation=90)
plt.show()




sctp_predictions_lgbm = lgbm_twoclass_model.predict(dataset_sctp_test)
sctp_predictions_lgbm = [round(value) for value in sctp_predictions_lgbm]
sctp_predictions_lgbm = list(map(int, sctp_predictions_lgbm))
sctp_predictions_lgbm[:20]




dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_lgbm
dataset_submission.to_csv('lgbm_twoclass_model_submission.csv', index=False)




dataset_sctp_test_xgb = xgb.DMatrix(dataset_sctp_test)
sctp_predictions_xgb = xgb_twoclass_model.predict(dataset_sctp_test_xgb)
sctp_predictions_xgb = [round(value) for value in sctp_predictions_xgb]
sctp_predictions_xgb = list(map(int, sctp_predictions_xgb))
sctp_predictions_xgb[:20]




dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_xgb
dataset_submission.to_csv('xgb_twoclass_model_submission.csv', index=False)




print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

