#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score

import warnings
warnings.filterwarnings('ignore')

import eli5
from eli5.sklearn import PermutationImportance

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_feather("../input/ion-switching-expriment-lag-and-lead-features/train_lag30.feather")
test = pd.read_feather("../input/ion-switching-expriment-lag-and-lead-features/test_lag30.feather")
sample_submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train.columns


# In[5]:


## Preparation

drop_cols = [
    'time','open_channels'
    ]

train['open_channels'] = train['open_channels'].astype(int)

X = train.drop(drop_cols, axis = 1)
X_test = test.drop(drop_cols, axis = 1)

y = train['open_channels']

print(f'Number of features = {len(X.columns)}')

del train
del test
gc.collect()


# In[6]:


#--------------------------------------------------------
# Parameter Setting
#--------------------------------------------------------

# I also use 'tracking'thanks Rob!!
#
TOTAL_FOLDS = 5
MODEL_TYPE = 'LGBM'
SHUFFLE = True
NUM_BOOST_ROUND = 2_500
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 500
RANDOM_SEED = 31452

##
calc_feature_imp = True
calc_perm_imp = True  ##!!　Calculation takes time.

# prams for validation
params = {
    'learning_rate': 0.03, 
    'max_depth': -1,
    'num_leaves': 2**8+1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.82,
    'bagging_freq': 0,
    'n_jobs': 8,
    'random_state': 31452,
    'metric': 'rmse',
    'objective' : 'regression'
    }

params2 = {
    #'eval_metric': 'rmse', #This did not work.4/26
    'n_estimators': 50000,
    'early_stopping_rounds': 50
    }

params.update(params2)


# In[7]:


#--------------------------------------------------------
# Validation
#--------------------------------------------------------

fold = StratifiedKFold(n_splits=TOTAL_FOLDS, shuffle=SHUFFLE, random_state=RANDOM_SEED)

df_feature_importance = pd.DataFrame()
oof_pred = np.ones(len(X))*-1
models = []


for i, (train_index, valid_index) in enumerate(fold.split(X, y)):
    print(f'Fold-{i+1} started at {time.ctime()}')
    X_train, X_valid = X.iloc[train_index, :], X.iloc[valid_index, :]
    y_train, y_valid = y[train_index], y[valid_index]
    
    model = lgb.LGBMRegressor()
    model.set_params(**params)
    model.fit(X_train, y_train,
              eval_set = [(X_train, y_train),(X_valid, y_valid)],
              verbose = 500
              )
    
    y_pred_valid = model.predict(X_valid, num_iteration = model.best_iteration_)
    y_pred_valid = np.round(np.clip(y_pred_valid, 0, 10)).astype(int)
    f1_valid = f1_score(y_valid, y_pred_valid, average = 'macro')
    
    oof_pred[valid_index] = y_pred_valid
    #len(oof_pred[oof_pred >= 0])
    print(f'Fold-{i+1} f1 score = {f1_valid:0.5f}')
    
    models.append(model)
    
    #del X_train, y_train
    gc.collect()
    
    ## Feateure Importance
    
    if i == 0: # for time reducing
        if calc_feature_imp:
            #model.feature_importances_
            fold_importance = pd.DataFrame()
            fold_importance['feature'] = X.columns.tolist()
            fold_importance['fold'] = i + 1
            fold_importance['split_importance'] = model.booster_.feature_importance(importance_type='split')
            fold_importance['gain_importance'] = model.booster_.feature_importance(importance_type='gain')


        if calc_feature_imp * calc_perm_imp:
            # from corochann's notebooks
            print("Calculating permutation importance... ")
            perm = PermutationImportance(model, random_state = 1, cv = 'prefit')
            perm.fit(X_valid, y_valid)
            fold_importance['permutation_importance'] = perm.feature_importances_

        if calc_feature_imp:
            df_feature_importance =                pd.concat([df_feature_importance, fold_importance], axis= 0)

    del X_valid, y_valid
    gc.collect()
    
    break # <- !!! fold1 only, add 5/2


# In[8]:


# Only fold1 calculation, so commented out here.

#oof_f1 = f1_score(y, oof_pred, average = "macro")
#print(f'f1_score_oof = {oof_f1:0.5f}')


# In[9]:


def plot_feature_importance_by3(df_feature_importance, null_imp = False):
    
    fig, ax = plt.subplots(1, 3, figsize = (18, 14))
    plt.rcParams["font.size"] = 15

    if not null_imp:
        col_name = ["split_importance", "gain_importance", "permutation_importance"]
    else:
        col_name = ["p_by_split_distribution", "p_by_gain_distribution", "dummy"]


    for i, importance_name in enumerate(col_name):
        try:
            cols = (df_feature_importance[["feature", importance_name]]                    .groupby("feature").mean()                    .sort_values(by = importance_name, ascending=False)[:100].index)

            best_features = df_feature_importance.loc[df_feature_importance.feature.isin(cols), ['feature', importance_name]]
            sns.barplot(x = importance_name, y ="feature", ax=ax[i], 
                        data = best_features.sort_values(by = importance_name, ascending=False))
            plt.tight_layout()
            ax[i].set_title(f'{importance_name} (averaged over folds)')
        except:
            pass
    plt.tight_layout()


# In[10]:


#Name shortening
st1 = df_feature_importance[df_feature_importance['feature'].str.contains("_10s")]['feature'].str.replace("_10s", "")
df_feature_importance.loc[df_feature_importance['feature'].str.contains("_10s"), 'feature'] = st1

plot_feature_importance_by3(df_feature_importance)


# In[11]:


df_lags = df_feature_importance[df_feature_importance['feature']                      .str.startswith("lag")]
plot_feature_importance_by3(df_lags)


# In[12]:


df_leads = df_feature_importance[df_feature_importance['feature']                      .str.startswith("lead")]
plot_feature_importance_by3(df_leads)


# In[13]:


#--------------------------------------------------------
# Test Predoction and Submission
#--------------------------------------------------------  

flg_submit = False

if flg_submit:
    
    test_preds = pd.DataFrame()
    for i, model in enumerate(models):
        print(f'Predictinig {i+1}th model...')
        test_pred = model.predict(X_test, num_iteration = model.best_iteration_)
        test_pred = np.round(np.clip(test_pred, 0, 10)).astype(int)
        test_preds[f'Fold{i+1}'] = test_pred

    sample_submission['open_channels'] = test_preds.median(axis=1).astype(int)
    #sample_submission.open_channels.value_counts()

    save_sub_name = 'submission.csv'

    sample_submission.to_csv(save_sub_name,
            index=False,
            float_format='%0.4f')


# In[14]:


#dir()
del df_lags, df_leads, fold_importance, models, sample_submission, oof_pred
gc.collect()


# In[15]:


#--------------------------------------
# Calculation null importnce
#--------------------------------------

## re-define params

n_est = model.booster_.best_iteration

# prams for validation
params_null = {
    'learning_rate': 0.03, 
    'max_depth': -1,
    'num_leaves': 2**8+1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.82,
    'bagging_freq': 0,
    'n_jobs': 8,
    'random_state': 31452,
    'metric': 'rmse',
    'objective' : 'regression'
    }

params_null2 = {
    #'eval_metric': 'rmse', #This did not work.4/26
    'n_estimators': n_est #,
    #'early_stopping_rounds': 50
    }

params_null.update(params_null2)


n_round = 30 #!!

df_null_importance = pd.DataFrame()

for i in range(n_round):
    if i % 10 == 0:
        print(f'Calculating null importance round {i}')
    y_null = y_train.copy().sample(frac = 1.0)
    
    model_null = lgb.LGBMRegressor()
    model_null.set_params(**params_null)
    
    model_null.fit(X_train, y_null, 
                   eval_set = [(X_train, y_null)], #,(X_valid, y_valid)],
                   verbose = 300)
    
    tmp_importance = pd.DataFrame()
    tmp_importance['feature'] = X.columns.tolist()
    tmp_importance['round'] = i + 1 
    tmp_importance['split_importance'] =        model_null.booster_.feature_importance(importance_type = 'split')
    tmp_importance['gain_importance'] =        model_null.booster_.feature_importance(importance_type = 'gain')
    
    df_null_importance =        pd.concat([df_null_importance, tmp_importance], axis = 0)

#Name shotening
st1 = df_null_importance[df_null_importance['feature'].str.contains("_10s")]['feature'].str.replace("_10s", "")
df_null_importance.loc[df_null_importance['feature'].str.contains("_10s"), 'feature'] = st1


# In[16]:


def plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp):

    fig, ax = plt.subplots(3, 4, figsize = (15, 10))

    k = 0
    for i in range(3):
        for j in range(4):
            try:
                disp_col = X_cols[k]

                act_imp =                    df_feature_importance[df_feature_importance['feature'] == disp_col][imp].values
                null_imp =                    df_null_importance[df_null_importance['feature'] == disp_col][imp]

                f = ax[i, j].hist(null_imp, alpha = 0.8, color = "dodgerblue", 
                             label = "Null importance")
                y_max = np.max(f[0])

                ax[i, j].plot([act_imp, act_imp], [0.0, y_max], linewidth = 7, color ="magenta", label = "Real target")
                ax[i, j].set_title(disp_col.replace("_10s", ""),  fontsize = 16)
                k += 1
            except:
                pass
    plt.tight_layout()
    fig.suptitle("Distribution of " + imp + ": actual(magenta), target_permutation(blue) ", fontsize=20)
    plt.subplots_adjust(top=0.9)
    plt.show()


# In[17]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("mean|sd")]['feature'].unique()
imp = "split_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[18]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("lag")]['feature'].unique()
imp = "split_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[19]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("lead")]['feature'].unique()
imp = "split_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[20]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("mean|sd")]['feature'].unique()
imp = "gain_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[21]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("lag")]['feature'].unique()
imp = "gain_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[22]:


X_cols = df_feature_importance[df_feature_importance['feature'].str.contains("lead")]['feature'].unique()
imp = "gain_importance"
plot_null_dist(df_feature_importance, df_null_importance, X_cols, imp)


# In[23]:


## evaluation of null_importance

#df_null_p = df_null_importance.groupby('feature').agg(["mean", "std"], sort = False).reset_index()
df_null_mean = df_null_importance.groupby('feature', sort = False).mean().reset_index()
df_null_mean.drop('round', axis = 1, inplace = True)
df_null_mean.rename(columns = {'split_importance':'split_importance_mean', 
                               'gain_importance':'gain_importance_mean'}, inplace = True)

df_null_sd = df_null_importance.groupby('feature', sort = False).std().reset_index()
df_null_sd.drop('round', axis = 1, inplace = True)
df_null_sd.rename(columns = {'split_importance':'split_importance_sd', 
                               'gain_importance':'gain_importance_sd'}, inplace = True)

df_act = df_feature_importance.groupby('feature', sort =False).mean().reset_index()
df_act.drop('fold', axis = 1, inplace = True)

df_null_summary = df_act.merge(df_null_mean, how = "left", on = 'feature')
df_null_summary = df_null_summary.merge(df_null_sd, how = "left", on = 'feature')
df_null_summary['z_split'] = (df_null_summary['split_importance']-df_null_summary['split_importance_mean'])/df_null_summary['split_importance_sd'] 
df_null_summary['z_gain'] = (df_null_summary['gain_importance']-df_null_summary['gain_importance_mean'])/df_null_summary['gain_importance_sd'] 

from scipy.stats import norm
df_null_summary['p_by_split_distribution'] = norm.cdf(x = df_null_summary['z_split'], loc = 0, scale = 1)
df_null_summary['p_by_gain_distribution'] = norm.cdf(x = df_null_summary['z_gain'], loc = 0, scale = 1)

#df_null_summary.columns


# In[24]:


df_null_summary_tmp =df_null_summary[~df_null_summary['feature'].str.contains("lag2[0-9]|lead2[0-9]|lag3[0-9]|lead3[0-9]")]
plot_feature_importance_by3(df_null_summary_tmp, null_imp = True)


# In[25]:


df_null_summary_tmp = df_null_summary[df_null_summary['feature'].str.contains("lag")]
plot_feature_importance_by3(df_null_summary_tmp, null_imp = True)


# In[26]:


df_null_summary_tmp = df_null_summary[df_null_summary['feature'].str.contains("lead")]
plot_feature_importance_by3(df_null_summary_tmp, null_imp = True)

