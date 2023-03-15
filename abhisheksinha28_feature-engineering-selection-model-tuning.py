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




import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df):
    print(f'Original size of data: {return_size(df)} gb.')
    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    print(f'New size of data: {return_size(df)} gb.')
    return df
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
plt.style.use('fivethirtyeight')




train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})

test['TARGET'] = np.nan
data = train.append(test, ignore_index=True, sort=True)

data = convert_types(data)
bureau = convert_types(bureau)
bureau_balance = convert_types(bureau_balance)

import gc
gc.enable()
del train, test
gc.collect()




def kde_target(var_name,df):
    corr = df['TARGET'].corr(df[var_name])
    sns.kdeplot(df.ix[df['TARGET']==0, var_name], label = 'target ==0')
    sns.kdeplot(df.ix[df['TARGET']==1, var_name], label= 'target ==1')
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend()
 




def agg_numeric(df, group_var, df_name):
    
    
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    _, idx = np.unique(agg,axis=1,return_index=True)
    agg= agg.iloc[:,idx]
    
    return agg




def target_corrs(df):
    corrs = []
    for col in df :
        print(col)
        if col!= 'TARGET':
            corr = df['TARGET'].corr(df[col])
            corrs.append((col,corr))
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    return corrs




def agg_categorical(df,group_var,df_name) :
    categorical = pd.get_dummies(df.select_dtypes('category'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum','count','mean'])
    
    column_names = []
    
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum','count','mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    _,idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:,idx]
    return categorical




import gc

def agg_child(df,parent_var,df_name):
    df_agg = agg_numeric(df,parent_var,df_name)
    df_cat = agg_categorical(df,parent_var,df_name)
    df_combined = df_agg.merge(df_cat,on=parent_var, how='outer')
    
    _,idx = np.unique(df_combined,axis=1,return_index=True)
    df_combined = df_combined.iloc[:,idx]
    
    gc.enable()
    del df_agg,df_cat
    gc.collect()
    
    return df_combined




def agg_grandchild(df, parent_df,parent_var, grandparent_var, df_name):
    parent_df = parent_df[[parent_var,grandparent_var]].copy().set_index(parent_var)
    df_agg = agg_numeric(df,parent_var,'%s_LOAN' % df_name)
    df_agg = df_agg.merge(parent_df, on = parent_var, how = 'left')
    df_agg_client = agg_numeric(df_agg,grandparent_var,'%s_CLIENT' %df_name)
    if any(df.dtypes == 'category'):
    
        # Aggregate the categorical variables at the parent level
        df_agg_cat = agg_categorical(df, parent_var, '%s_LOAN' % df_name)
        df_agg_cat = df_agg_cat.merge(parent_df,
                                      on = parent_var, how = 'left')

        # Aggregate the categorical variables at the grandparent level
        df_agg_cat_client = agg_numeric(df_agg_cat, grandparent_var, '%s_CLIENT' % df_name)
        df_info = df_agg_client.merge(df_agg_cat_client, on = grandparent_var, how = 'outer')
        
        gc.enable()
        del df_agg, df_agg_client, df_agg_cat, df_agg_cat_client
        gc.collect()
        
    else:
        df_info =df_agg_client.copy()
        gc.enable()
        del df_agg_client,df_agg
        gc.collect()
        
    _, idx = np.unique(df_info, axis=1, return_index =True)
    df_info = df_info.iloc[:,idx]
    return df_info
            




data['LOAN_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT'] 
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['EMPLOYED_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['EXT_SOURCE_SUM'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
data['AMT_REQ_SUM'] = data[[x for x in data.columns if 'AMT_REQ_' in x]].sum(axis = 1)




bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']




bureau_info = agg_child(bureau, 'SK_ID_CURR', 'BUREAU')
bureau_info.head()




bureau_info.shape




bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'




bureau_balance_info = agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
del bureau_balance, bureau
bureau_balance_info.head()




bureau_balance_info.shape




data = data.set_index('SK_ID_CURR')
data = data.merge(bureau_info, on = 'SK_ID_CURR', how = 'left')
del bureau_info
data.shape




data = data.merge(bureau_balance_info, on = 'SK_ID_CURR', how = 'left')
del bureau_balance_info
data.shape




previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
previous = convert_types(previous)
previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']




previous_info = agg_child(previous, 'SK_ID_CURR', 'PREVIOUS')
previous_info.shape




data = data.merge(previous_info, on = 'SK_ID_CURR', how = 'left')
del previous_info
data.shape




installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})
installments = convert_types(installments)
installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']




installments_info = agg_grandchild(installments,previous,'SK_ID_PREV', 'SK_ID_CURR', 'IN')
del installments
installments_info.shape




data = data.merge(installments_info, on = 'SK_ID_CURR', how = 'left')
del installments_info
data.shape




cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
cash = convert_types(cash)
cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']




cash_info = agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
del cash
cash_info.shape




data = data.merge(cash_info, on='SK_ID_CURR', how='left')
del cash_info
data.shape




credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
credit = convert_types(credit)
credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
credit['LATE'] = credit['SK_DPD'] > 0.0




credit_info = agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
del credit, previous
credit_info.shape




gc.collect()
gc.enable()





data = data.merge(credit_info, on = 'SK_ID_CURR', how = 'left')
del credit_info




data.shape




print('After manual feature engineering, there are {} features.'.format(data.shape[1] - 2))




gc.enable()
gc.collect()




print(f'Final size of data {return_size(data)}')




#data.to_csv('clean_manual_features.csv', chunksize = 100)




data.reset_index(inplace = True)
train, test = data[data['TARGET'].notnull()].copy(), data[data['TARGET'].isnull()].copy()
gc.enable()
del data
gc.collect()




train_labels = np.array(train.pop('TARGET')).reshape((-1, ))

test_ids = list(test.pop('SK_ID_CURR'))
test = test.drop(columns = ['TARGET'])
train = train.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)




train_df = train[:1000]
test_df = test[:1000]




train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Match the columns in the dataframes
train, test = train.align(test, join = 'inner', axis = 1)

train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Match the columns in the dataframes
train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)




cols_with_id = [x for x in train_df.columns if 'SK_ID_CURR' in x]
cols_with_bureau_id = [x for x in train_df.columns if 'SK_ID_BUREAU' in x]
cols_with_previous_id = [x for x in train_df.columns if 'SK_ID_PREV' in x]
print('There are %d columns that contain SK_ID_CURR' % len(cols_with_id))
print('There are %d columns that contain SK_ID_BUREAU' % len(cols_with_bureau_id))
print('There are %d columns that contain SK_ID_PREV' % len(cols_with_previous_id))

train = train.drop(columns = cols_with_id)
test = test.drop(columns = cols_with_id)
print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)




threshold = 0.9

# Absolute value correlation matrix
corr_matrix = train_df.corr().abs()
corr_matrix.head()




upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()




to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))




train = train.drop(columns = to_drop)
test = test.drop(columns = to_drop)

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)




train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)
train_missing.head()




test_missing = (test.isnull().sum() / len(test)).sort_values(ascending = False)
test_missing.head()




train_missing = train_missing.index[train_missing > 0.75]
test_missing = test_missing.index[test_missing > 0.75]

all_missing = list(set(set(train_missing) | set(test_missing)))
print('There are %d columns with more than 75%% missing values' % len(all_missing))




#train_labels = train["TARGET"]
#train_ids = train['SK_ID_CURR']
#test_ids = test['SK_ID_CURR']

train = pd.get_dummies(train.drop(columns = all_missing))
test = pd.get_dummies(test.drop(columns = all_missing))

train, test = train.align(test, join = 'inner', axis = 1)

print('Training set full shape: ', train.shape)
print('Testing set full shape: ' , test.shape)




import lightgbm as lgb
feature_importances = np.zeros(train.shape[1])

# Create the model with several hyperparameters
model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')




train1 = pd.read_csv('../input/application_train.csv')
test1 = pd.read_csv('../input/application_test.csv')
train_labels = train1["TARGET"]
train_ids = train1['SK_ID_CURR']
test_ids = test1['SK_ID_CURR']




del train1 , test1
gc.collect()




for i in range(2):
    
    # Split into training and validation set
    train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size = 0.25, random_state = i)
    
    # Train using early stopping
    model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
              eval_metric = 'auc', verbose = 200)
    
    # Record the feature importances
    feature_importances += model.feature_importances_




feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

feature_importances.head()




zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
feature_importances.tail()




train = train.drop(columns = zero_features)
test = test.drop(columns = zero_features)




def plot_feature_importances(df, n = 15, threshold = None):
    """
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be "feature" and "importance"
    
    n : int, default = 15
        Number of most important features to plot
    
    threshold : float, default = None
        Threshold for cumulative importance plot. If not provided, no plot is made
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    Note
    --------
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
    
    """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', edgecolor = 'k', figsize = (12, 8),
                            legend = False)

    plt.xlabel('Normalized Importance', size = 18)
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18)
        
        # Number of features needed for threshold cumulative importance
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.2, linestyles = '--', colors = 'red')
        plt.show()
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 100 * threshold))
    
    return df




plot_feature_importances(feature_importances, n=15, threshold = 0.9)




train['TARGET'] = train_labels
train['SK_ID_CURR'] = train_ids
test['SK_ID_CURR'] = test_ids




features = train

# Sample 16000 rows (10000 for training, 6000 for testing)
features = features.sample(n = 16000, random_state = 42)

# Only numeric features
features = features.select_dtypes('number')

# Extract the labels
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])

# Split into training and testing data
train_features, test_features, trainlabels, testlabels = train_test_split(features, labels, test_size = 6000, random_state = 50)




print("Training features shape: ", train_features.shape)
print("Testing features shape: ", test_features.shape)




train_set = lgb.Dataset(data = train_features, label = trainlabels)
test_set = lgb.Dataset(data = test_features, label = testlabels)




default_params = model.get_params()

N_FOLDS = 5
MAX_EVALS = 5
# Remove the number of estimators because we set this to 10000 in the cv call
del default_params['n_estimators']

# Cross validation with early stopping
cv_results = lgb.cv(default_params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = N_FOLDS, seed = 42)




print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.'.format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('The optimal number of boosting rounds (estimators) was {}.'.format(len(cv_results['auc-mean'])))




from sklearn.metrics import roc_auc_score
model.n_estimators = len(cv_results['auc-mean'])

# Train and make predicions with model
model.fit(train_features, trainlabels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(testlabels, preds)

print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))




def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 42)
    
    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    return [score, hyperparameters, iteration]




score, params, iteration = objective(default_params, 1)

print('The cross-validation ROC AUC was {:.5f}.'.format(score))




# Create a default model
model = lgb.LGBMModel()
model.get_params()




param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}




import random

random.seed(50)

# Randomly sample a boosting type
boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

# Set subsample depending on boosting type
subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

print('Boosting type: ', boosting_type)
print('Subsample ratio: ', subsample)




import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Learning rate histogram
plt.hist(param_grid['learning_rate'], bins = 20, color = 'r', edgecolor = 'k')
plt.xlabel('Learning Rate', size = 14); plt.ylabel('Count', size = 14); plt.title('Learning Rate Distribution', size = 18)




random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))

grid_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))




import itertools

def grid_search(param_grid, max_evals = MAX_EVALS):
    """Grid search algorithm (with limit on max evals)"""
    
    # Dataframe to store results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))
    
    # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
    keys, values = zip(*param_grid.items())
    
    i = 0
    
    # Iterate through every possible combination of hyperparameters
    for v in itertools.product(*values):
        
        # Create a hyperparameter dictionary
        hyperparameters = dict(zip(keys, v))
        
        # Set the subsample ratio accounting for boosting type
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        
        # Evalute the hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
        
        i += 1
        
        # Normally would not limit iterations
        if i > MAX_EVALS:
            break
        # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results




''''
grid_results = grid_search(param_grid)

print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')
grid_results = grid_results.drop(columns='index')
grid_results.to_csv('grid_results.csv', header= ['score', 'hyperparameters', 'iteration'] )
import pprint
pprint.pprint(grid_results.loc[0, 'params'])
''''




''''
grid_search_params = grid_results.loc[0, 'params']

# Create, train, test model
model = lgb.LGBMClassifier(**grid_search_params, random_state=42)
model.fit(train_features, trainlabels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from grid search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(testlabels, preds)))
''''




random.seed(50)

# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

random_params




def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 




random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])
random_results = random_results.drop(columns='index')
random_results.to_csv('random_results.csv', header= ['score', 'hyperparameters', 'iteration'] )




random_results




def evaluate(results, name):
    """Evaluate model on test data using hyperparameters in results
       Return dataframe of hyperparameters"""
        
    # Sort with best values on top
    results = results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, results.loc[0, 'score'], results.loc[0, 'iteration']))
    
    # Use best hyperparameters to create a model
    hyperparameters = results.loc[0, 'params']
    model = lgb.LGBMClassifier(**hyperparameters)
    
    # Train and make predictions
    model.fit(train_features, trainlabels)
    preds = model.predict_proba(test_features)[:, 1]
    
    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(testlabels, preds)))
    
    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(results.loc[0, 'params'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(results['params']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True)
        
    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = results['iteration']
    hyp_df['score'] = results['score']
    
    return hyp_df




random_hyp = evaluate(random_results, name = 'random search')




test_ids = test['SK_ID_CURR']
train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))

train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)




train_set = lgb.Dataset(train, label = train_labels)

hyperparameters = dict(**random_results.loc[0, 'params'])
del hyperparameters['n_estimators']

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameters, train_set,
                    num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = N_FOLDS)




print('The cross validation score on the full dataset = {:.5f} with std: {:.5f}.'.format(
    cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('Number of estimators = {}.'.format(len(cv_results['auc-mean'])))




# Train the model with the optimal number of estimators from early stopping
model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)
model.fit(train, train_labels)
                        
# Predictions on the test data
preds = model.predict_proba(test)[:, 1]




submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
submission.to_csv('submission_simple_features_random.csv', index = False)

