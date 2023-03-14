#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns

rcParams['figure.figsize'] = (8, 5)

train = pd.read_csv('../input/train/train.csv')
test  = pd.read_csv('../input/test/test.csv')
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()

cat_vars = ['type', 'breed1', 'breed2', 'gender', 'color1', 'color2',
            'color3', 'vaccinated', 'dewormed', 'sterilized', 'health', 'state']

# no missing values in categorical variables
assert not train[cat_vars].isnull().any().any()
assert not test[cat_vars].isnull().any().any()

train[cat_vars].nunique()




def encode_onehot(train, test, categ_variables):
    df_onehot = pd.get_dummies(pd.concat([train[cat_vars], test[cat_vars]]).astype(str))
    return df_onehot[:len(train)], df_onehot[len(train):]

train_onehot, test_onehot = encode_onehot(train, test, cat_vars)
train_onehot.shape




encod_type = train.groupby('type')['adoptionspeed'].mean()
print(encod_type)
train.loc[:, 'type_mean_enc'] = train['type'].map(encod_type)
train[['type','type_mean_enc']].head()




(train.groupby('breed1').size() / len(train)).nlargest(10)




def encode_target_smooth(data, target, categ_variables, smooth):
    """    
    Apply target encoding with smoothing.
    
    Parameters
    ----------
    data: pd.DataFrame
    target: str, dependent variable
    categ_variables: list of str, variables to encode
    smooth: int, number of observations to weigh global average with
    
    Returns
    --------
    encoded_dataset: pd.DataFrame
    code_map: dict, mapping to be used on validation/test datasets 
    defaul_map: dict, mapping to replace previously unseen values with
    """
    train_target = data.copy()
    code_map = dict()    # stores mapping between original and encoded values
    default_map = dict() # stores global average of each variable
    
    for v in categ_variables:
        prior = data[target].mean()
        n = data.groupby(v).size()
        mu = data.groupby(v)[target].mean()
        mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
        
        train_target.loc[:, v] = train_target[v].map(mu_smoothed)        
        code_map[v] = mu_smoothed
        default_map[v] = prior        
    return train_target, code_map, default_map




train_target_smooth, target_map, default_map = encode_target_smooth(train, 'adoptionspeed', cat_vars, 500)
test_target_smooth = test.copy()
for v in cat_vars:
    test_target_smooth.loc[:, v] = test_target_smooth[v].map(target_map[v])




train_target_smooth[cat_vars].head()




def impact_coding_leak(data, feature, target, n_folds=20, n_inner_folds=10):
    from sklearn.model_selection import StratifiedKFold
    '''
    ! Using oof_default_mean for encoding inner folds introduces leak.
    
    Source: https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features
    
    Changelog:    
    a) Replaced KFold with StratifiedFold due to class imbalance
    b) Rewrote .apply() with .map() for readability
    c) Removed redundant apply in the inner loop
    '''
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # KFold in the original
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature], data[target]):

        kf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_oof_mean_cv = pd.DataFrame()
        oof_default_inner_mean = data.iloc[infold][target].mean()
        
        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold], data.loc[infold, target]):
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()

            # Also populate mapping (this has all group -> mean for all inner CV folds)
            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
            inner_split += 1

        # compute mean for each value of categorical value across oof iterations
        inner_oof_mean_cv_map = inner_oof_mean_cv.mean(axis=1)

        # Also populate mapping
        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
        oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
        split += 1

        feature_mean = data.loc[oof, feature].map(inner_oof_mean_cv_map).fillna(oof_default_mean)
        impact_coded = impact_coded.append(feature_mean)
            
    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean


def impact_coding(data, feature, target, n_folds=20, n_inner_folds=10):
    from sklearn.model_selection import StratifiedKFold
    '''
    ! Using oof_default_mean for encoding inner folds introduces leak.
    
    Source: https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features
    
    Changelog:    
    a) Replaced KFold with StratifiedFold due to class imbalance
    b) Rewrote .apply() with .map() for readability
    c) Removed redundant apply in the inner loop
    d) Removed global average; use local mean to fill NaN values in out-of-fold set
    '''
    impact_coded = pd.Series()
        
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # KFold in the original
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature], data[target]):

        kf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_oof_mean_cv = pd.DataFrame()
        oof_default_inner_mean = data.iloc[infold][target].mean()
        
        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold], data.loc[infold, target]):
                    
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
            
            # Also populate mapping (this has all group -> mean for all inner CV folds)
            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
            inner_split += 1

        # compute mean for each value of categorical value across oof iterations
        inner_oof_mean_cv_map = inner_oof_mean_cv.mean(axis=1)

        # Also populate mapping
        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
        oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True) # <- local mean as default
        split += 1

        feature_mean = data.loc[oof, feature].map(inner_oof_mean_cv_map).fillna(oof_default_inner_mean)
        impact_coded = impact_coded.append(feature_mean)
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean

def encode_target_cv(data, target, categ_variables, impact_coder=impact_coding):
    """Apply original function for each <categ_variables> in  <data>
    Reduced number of validation folds
    """
    train_target = data.copy() 
    
    code_map = dict()
    default_map = dict()
    for f in categ_variables:
        train_target.loc[:, f], code_map[f], default_map[f] = impact_coder(train_target, f, target)
        
    return train_target, code_map, default_map




train_target_cv, code_map, default_map = encode_target_cv(train, 'adoptionspeed', cat_vars, impact_coder=impact_coding)




train_target_cv[cat_vars].head()




corr_map = dict()
for v in cat_vars:
    corr_map[v] = np.corrcoef(train_target_cv[v], train_target_smooth[v])[0, 1]    
reg_correl = pd.Series(corr_map)

num_categories = train[cat_vars].nunique()




reg_correl.plot(kind='barh', color='green', alpha=0.3)
_ = plt.title('Correlation between mean-encoded-variables\n using smoothing and CV', fontsize=16)




fig = plt.figure(figsize=(10, 5))
_ = sns.kdeplot(train_target_smooth['breed1'], label='simple smoothing')
_ = sns.kdeplot(train_target_cv['breed1'], label='cross-validation')
_ = plt.title('Cross-validation regularisation introduced more variation than simple smoothing')




train[cat_vars].nunique().plot(kind='barh')
_ = plt.title('Number of unique categories')




fig, ax = plt.subplots()
_ = ax.scatter(num_categories, reg_correl)
_ = ax.set_xlabel('Number of unique categories in a variable', fontsize=14)
_ = ax.set_ylabel('Correlation between 2 regularisations', fontsize=14)
for i, txt in enumerate(num_categories.index):
    ax.annotate(txt, (num_categories[i], reg_correl[i]))




train.groupby('health').size()




def get_categor_spread(data, categ_variables):
    spread = dict()
    for v in categ_variables:
        dist = data.groupby(v).size()
        spread[v] = dist.max() / dist.min() / len(data)
    return spread




spread = pd.Series(get_categor_spread(train, cat_vars))
spread.plot(kind='barh')
_ = plt.title('Larger spread indicates bigger difference between value\n with highest and lowest number of observations')




fig, ax = plt.subplots()
_ = ax.scatter(spread, reg_correl)
_ = ax.set_xlabel('Number of unique categories in a variable', fontsize=14)
_ = ax.set_ylabel('Correlation between 2 regularisations', fontsize=14)
for i, txt in enumerate(num_categories.index):
    ax.annotate(txt, (num_categories[i], reg_correl[i]))




from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer

gbc = GradientBoostingClassifier(n_estimators=10, random_state=20190301)

skf = StratifiedKFold(n_splits=10, random_state=20190301)

kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')




def cross_validate_encoder(X, target, categorical_vars, encoder,
                           model, n_splits=10, **kwargs):
    """Evaluate perfomance of encoding categorical varaibles with <encoder> by fitting 
    <model> and measuring average kappa on <n_samples> cross validation.
    
    Make sure to apply mean encoding only to infold set.
            
    Parameters
    ----------
    X: pd.DataFrame, train data
    target: str, response variable
    categorical_vars: list of str, categorical variables to encode
    encoder: custom function to apply
    model: sklearn model to fit
    n_splits: number of cross-validation folds
    **kwargs: key-word arguments to encoder
    
    Returns
    ----------
    metric_cvs: np.array of float, metrics computed on the held-out fold
    """     
    skf = StratifiedKFold(n_splits=n_splits, random_state=20190301)
    metric_cvs = list()
    
    for fold_idx, val_idx in skf.split(X=X, y=X[target]):
        train_fold, valid_fold = X.loc[fold_idx].reset_index(drop=True),                                  X.loc[val_idx].reset_index(drop=True)
        
        # apply encoding to k-th fold and validation set 
        train_fold, code_map, default_map = encoder(train_fold, target, categorical_vars, **kwargs)
        for v in categorical_vars:
            valid_fold.loc[:, v] = valid_fold[v].map(code_map[v]).fillna(default_map[v])
        
        # fit model on training fold
        model.fit(train_fold[categorical_vars], train_fold[target])
        # predict out-of-fold
        oof_pred = model.predict(valid_fold[categorical_vars])
        metric_cvs.append(cohen_kappa_score(valid_fold[target], oof_pred, weights='quadratic'))        
    return np.array(metric_cvs)




print('Evaluating one-hot... ')
score_onehot_gbc = cross_val_score(estimator=gbc, X=train_onehot, y=train.adoptionspeed,
                                   cv=skf, scoring=kappa_scorer)

print('Evaluating mean encoding with smoothing... ')
score_target_smooth_gbc = cross_validate_encoder(train, 'adoptionspeed', cat_vars, 
                                                 encode_target_smooth, gbc, smooth=500)

print('Evaluating mean encoding with cross-validation...')
score_target_cv_gbc = cross_validate_encoder(train, 'adoptionspeed', cat_vars, 
                                             encode_target_cv, gbc)




summary_gb = pd.DataFrame({'kappa_cv10_mean': [score_onehot_gbc.mean(), 
                                               score_target_smooth_gbc.mean(), 
                                               score_target_cv_gbc.mean()],
                           'kappa_cv10_std': [score_onehot_gbc.std(), 
                                              score_target_smooth_gbc.std(),
                                              score_target_cv_gbc.std()]},
                          index=['GradientBoosting - one hot', 
                                 'GradientBoosting - mean (smoothing)',
                                 'GradientBoosting - mean (cross-validation)']
                         )

summary_gb




summary_gb['kappa_cv10_mean'].plot(kind='barh', color='lightblue', xerr=summary_gb.kappa_cv10_std, ecolor='red')
_ = plt.xlabel('10-CV kappa average', fontsize=14)
_ = plt.title('Comparison of encoding schemes', fontsize=16, y=1.01)




from sklearn.metrics import roc_auc_score




def generate_data(n_obs=10000, n_lev=500, variance=1):
    np.random.seed(20190325)
    n_vars = 4
    df = np.empty([n_obs, n_vars])
    for i in range(n_vars):
        df[:, i] = np.random.choice(np.arange(n_lev), size=n_obs, replace=True)
    df = pd.DataFrame(df).astype(int)
    cat_cols = ['x_bad1', 'x_bad2', 'x_good1', 'x_good2']
    df.columns = cat_cols
    
    # y depends only x_good1 and x_good2
    y = (0.2 * np.random.normal(size=n_obs) 
         + 0.5 * np.where(df.x_good1 > n_lev / 2, 1, -1) 
         + 0.3 * np.where(df.x_good2 > n_lev / 2, 1, -1)
         + np.random.normal(scale=variance, size=n_obs)
        )
    df.loc[:, 'y'] = y > 0
    df.loc[:, 'split_group'] = np.random.choice(('cal','train','test'), 
                                                size=n_obs, 
                                                replace=True, 
                                                p=(0.6, 0.2, 0.2))
    
    df.loc[:, cat_cols] = df[cat_cols].astype(str) + '_level'
    return df




df = generate_data()
df.head()




df_train = df.loc[df.split_group!='test'].reset_index(drop=True)
df_test = df.loc[df.split_group=='test'].reset_index(drop=True)
cat_cols = ['x_bad1', 'x_bad2', 'x_good1', 'x_good2']




def encode_onehot(train, test, categ_variables):
    df_onehot = pd.get_dummies(pd.concat([train[categ_variables], test[categ_variables]]).astype(str))
    return df_onehot[:len(train)], df_onehot[len(train):]

train_onehot, test_onehot = encode_onehot(df_train, df_test, cat_cols)
train_onehot.shape




gbc = GradientBoostingClassifier(n_estimators=10, random_state=20190325)
gbc.fit(train_onehot, df_train['y'])
print(gbc.classes_)

# train
print(roc_auc_score(df_train['y'], gbc.predict_proba(train_onehot)[:, 1])) # taking True class
# test
print(roc_auc_score(df_test['y'], gbc.predict_proba(test_onehot)[:, 1]))

# import features
pd.Series(index=train_onehot.columns, data=gbc.feature_importances_).nlargest(10)




def encode_target_naive(train, test):
    df_train_naive = train.copy()
    df_test_naive = test.copy()
    
    default = df_train['y'].mean()
    for v in cat_cols:    
        encod_map = df_train.groupby(v)['y'].mean()
        df_train_naive.loc[:, v] = df_train_naive[v].map(encod_map).fillna(default)
        df_test_naive.loc[:, v] = df_test_naive[v].map(encod_map).fillna(default)
    return df_train_naive, df_test_naive




df_train_naive, df_test_naive = encode_target_naive(df_train, df_test)




gbc = GradientBoostingClassifier(random_state=20190325, n_estimators=10)
gbc.fit(df_train_naive[cat_cols], df_train_naive['y'])
pd.Series(gbc.feature_importances_, index=cat_cols)




pd.Series(gbc.feature_importances_, index=cat_cols)




# train
roc_auc_score(y_true=df_train_naive['y'], 
              y_score = gbc.predict_proba(df_train_naive[cat_cols])[:, 1])




# test
roc_auc_score(y_true=df_test_naive['y'], 
              y_score = gbc.predict_proba(df_test_naive[cat_cols])[:, 1])




gbc = GradientBoostingClassifier(random_state=20190325)
gbc.fit(df_train_naive[cat_cols], df_train_naive['y'])
pd.Series(gbc.feature_importances_, index=cat_cols)




# train
roc_auc_score(y_true=df_train_naive['y'], 
              y_score = gbc.predict_proba(df_train_naive[cat_cols])[:, 1])




# test
roc_auc_score(y_true=df_test_naive['y'], 
              y_score = gbc.predict_proba(df_test_naive[cat_cols])[:, 1])




from sklearn.metrics import roc_auc_score
def cross_validate_encoder(X, target, categorical_vars, encoder,
                           model, n_splits=10, **kwargs):
    """Evaluate perfomance of encoding categorical varaibles with <encoder> by fitting 
    <model> and measuring average kappa on <n_samples> cross validation.
    
    Make sure to apply mean encoding only to infold set.
            
    Parameters
    ----------
    X: pd.DataFrame, train data
    target: str, response variable
    categorical_vars: list of str, categorical variables to encode
    encoder: custom function to apply
    model: sklearn model to fit
    n_splits: number of cross-validation folds
    **kwargs: key-word arguments to encoder
    
    Returns
    ----------
    metric_cvs: np.array of float, metrics computed on the held-out fold
    """     
    skf = StratifiedKFold(n_splits=n_splits, random_state=20190301)
    metric_cvs = list()
    
    for fold_idx, val_idx in skf.split(X=X, y=X[target]):
        train_fold, valid_fold = X.loc[fold_idx].reset_index(drop=True),                                  X.loc[val_idx].reset_index(drop=True)
        
        # apply encoding to k-th fold and validation set 
        train_fold, code_map, default_map = encoder(train_fold, target, categorical_vars, **kwargs)
        for v in categorical_vars:
            valid_fold.loc[:, v] = valid_fold[v].map(code_map[v]).fillna(default_map[v])
        
        # fit model on training fold
        model.fit(train_fold[categorical_vars], train_fold[target])
        # predict out-of-fold
        oof_pred = model.predict_proba(valid_fold[categorical_vars])[:, 1]
        metric_cvs.append(roc_auc_score(valid_fold[target], oof_pred))
    return np.array(metric_cvs)




gbc = GradientBoostingClassifier(n_estimators=10, random_state=20190325)

print('Evaluating mean encoding with smoothing... ')
score_target_smooth_gbc = cross_validate_encoder(df_train, 'y', cat_cols, encode_target_smooth, gbc, smooth=500)

print('Evaluating mean encoding with cross-validation...')
score_target_cv_gbc = cross_validate_encoder(df_train, 'y', cat_cols, encode_target_cv, gbc)




score_target_smooth_gbc.mean(), score_target_cv_gbc.mean()




score_target_smooth_gbc.mean(), score_target_cv_gbc.mean()




gbc = GradientBoostingClassifier(random_state=20190325, n_estimators=10)
df_train_smooth, code_map, default_map = encode_target_smooth(df_train, 'y', cat_cols, smooth=500)

df_test_smooth = df_test.copy()
for v in cat_cols:
    df_test_smooth.loc[:, v] = df_test_smooth[v].map(code_map[v]).fillna(default_map[v])
    
gbc.fit(df_train_smooth[cat_cols], df_train_smooth['y'])




pd.Series(gbc.feature_importances_, index=cat_cols)




# train
roc_auc_score(y_true=df_train_smooth['y'], 
              y_score = gbc.predict_proba(df_train_smooth[cat_cols])[:, 1])




# test
roc_auc_score(y_true=df_test_smooth['y'], y_score = gbc.predict_proba(df_test_smooth[cat_cols])[:, 1])




gbc = GradientBoostingClassifier(random_state=20190325, n_estimators=10)
df_train_cv, code_map, default_map = encode_target_cv(df_train, 'y', cat_cols)

df_test_cv = df_test.copy()
for v in cat_cols:
    df_test_cv.loc[:, v] = df_test_cv[v].map(code_map[v]).fillna(default_map[v])

gbc.fit(df_train_cv[cat_cols], df_train_cv['y'])    




pd.Series(gbc.feature_importances_, index=cat_cols)




roc_auc_score(y_true=df_train_cv['y'], y_score = gbc.predict_proba(df_train_cv[cat_cols])[:, 1])




roc_auc_score(y_true=df_test_cv['y'], y_score = gbc.predict_proba(df_test_cv[cat_cols])[:, 1])

