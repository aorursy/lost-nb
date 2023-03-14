#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge

from catboost import CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

get_ipython().system('pip install git+http://github.com/brendanhasz/dsutils.git')
from dsutils.encoding import TargetEncoderCV
from dsutils.optimization import optimize_params
from dsutils.ensembling import EnsembleRegressor, StackedRegressor

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set()


# In[2]:


# Load data containing all the features
fname = '../input/elo-feature-selection/card_features_top100.feather'
cards = pd.read_feather(fname)
cards.set_index('card_id', inplace=True)


# In[3]:


# Test indexes
test_ix = cards['target'].isnull()

# Test data
X_test = cards[test_ix]#TODO: .copy()
del X_test['target']

# Training data
cards_train = cards[~test_ix]#TODO: .copy()
y_train = cards_train['target']#TODO: .copy()
X_train = cards_train.copy()
del X_train['target']

# Clean up
del cards_train
del cards
gc.collect()


# In[4]:


def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error regression loss"""
    return np.sqrt(np.mean(np.square(y_true-y_pred)))


# In[5]:


rmse_scorer = make_scorer(root_mean_squared_error)


# In[6]:


root_mean_squared_error(np.mean(y_train), y_train)


# In[7]:


get_ipython().run_cell_magic('time', '', "\n# Categorical columns\ncat_cols = [c for c in X_train if 'mode' in c] \n\n# Regression pipeline\nmodel = Pipeline([\n    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),\n    ('scaler',    RobustScaler()),\n    ('imputer',   SimpleImputer(strategy='median')),\n    ('regressor', CatBoostRegressor(verbose=False))\n])\n\n# Cross-validated performance\nscores = cross_val_score(model, X_train, y_train, \n                         cv=3, scoring=rmse_scorer)\nprint('Cross-validated MSE: %0.3f +/- %0.3f'\n      % (scores.mean(), scores.std()))")


# In[8]:


# Show histogram of target
y_train.hist(bins=30)
plt.xlabel('target')
plt.ylabel('count')
plt.show()


# In[9]:


y_train[y_train<-20].unique()


# In[10]:


print('Percent of targets which are outliers:', 100*np.mean(y_train<-20))


# In[11]:


def cross_val_metric(model, X, y, cv=3, 
                     metric=root_mean_squared_error, 
                     train_subset=None, test_subset=None, 
                     shuffle=False, display=None):
    """Compute a cross-validated metric for a model.
    
    Parameters
    ----------
    model : sklearn estimator or callable
        Model to use for prediction.  Either an sklearn estimator (e.g. a 
        Pipeline), or a function which takes 3 arguments: 
        (X_train, y_train, X_test), and returns y_pred.  X_train and X_test
        should be pandas DataFrames, and y_train and y_pred should be 
        pandas Series.
    X : pandas DataFrame
        Features.
    y : pandas Series
        Target variable.
    cv : int
        Number of cross-validation folds
    metric : sklearn.metrics.Metric
        Metric to evaluate.
    train_subset : pandas Series (boolean)
        Subset of the data to train on. 
        Must be same size as y, with same index as X and y.
    test_subset : pandas Series (boolean)
        Subset of the data to test on.  
        Must be same size as y, with same index as X and y.
    shuffle : bool
        Whether to shuffle the data. Default = False
    display : None or str
        Whether to print the cross-validated metric.
        If None, doesn't print.
    
    Returns
    -------
    metrics : list
        List of metrics for each test fold (length cv)
    preds : pandas Series
        Cross-validated predictions
    """
    
    # Use all samples if not specified
    if train_subset is None:
        train_subset = y.copy()
        train_subset[:] = True
    if test_subset is None:
        test_subset = y.copy()
        test_subset[:] = True
    
    # Perform the cross-fold evaluation
    metrics = []
    TRix = y.copy()
    TEix = y.copy()
    all_preds = y.copy()
    kf = KFold(n_splits=cv, shuffle=shuffle)
    for train_ix, test_ix in kf.split(X):
        
        # Indexes for samples in training fold and in train_subset
        TRix[:] = False
        TRix.iloc[train_ix] = True
        TRix = TRix & train_subset
        
        # Indexes for samples in test fold and in test_subset
        TEix[:] = False
        TEix.iloc[test_ix] = True
        TEix = TEix & test_subset
        
        # Predict using a function
        if callable(model):
            preds = model(X.loc[TRix,:], y[TRix], X.loc[TEix,:])
        else:
            model.fit(X.loc[TRix,:], y[TRix])
            preds = model.predict(X.loc[TEix,:])
        
        # Store metrics for this fold
        metrics.append(metric(y[TEix], preds))
        
        # Store predictions for this fold
        all_preds[TEix] = preds

    # Print the metric
    metrics = np.array(metrics)
    if display is not None:
        print('Cross-validated %s: %0.3f +/- %0.3f'
              % (display, metrics.mean(), metrics.std()))
        
    # Return a list of metrics for each fold
    return metrics, all_preds


# In[12]:


get_ipython().run_cell_magic('time', '', "\n# Compute which samples are outliers\nnonoutliers = y_train>-20\n\n# Cross-validated performance training only on non-outliers\ncross_val_metric(model, X_train, y_train, cv=3, \n                 metric=root_mean_squared_error, \n                 train_subset=nonoutliers,\n                 display='RMSE')")


# In[13]:


# Classification pipeline
classifier = Pipeline([
    ('targ_enc',   TargetEncoderCV(cols=cat_cols)),
    ('scaler',     RobustScaler()),
    ('imputer',    SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier())
])

# Regression pipeline
regressor = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor())
])


# In[14]:


def classifier_regressor(X_tr, y_tr, X_te):
    """Classify outliers, and set to outlier value if a predicted outlier
    else use a regressor trained on non-outliers
    
    Parameters
    ----------
    X_tr : pandas DataFrame
        Training features
    y_tr : pandas Series
        Training target
    X_te : pandas DataFrame
        Test features
        
    Returns
    -------
    y_pred : pandas Series
        Predicted y values for samples in X_te
    """
    
    # Fit the classifier to predict outliers
    outliers = y_tr<-20
    fit_classifier = classifier.fit(X_tr, outliers)
    
    # Samples which do not have outlier target values
    X_nonoutlier = X_tr.loc[~outliers, :]
    y_nonoutlier = y_tr[~outliers]

    # Fit the regressor to estimate non-outlier targets
    fit_regressor = regressor.fit(X_nonoutlier, y_nonoutlier)
    
    # Predict outlier probability
    pred_outlier = fit_classifier.predict_proba(X_te)[:,1]

    # Estimate target
    y_pred = fit_regressor.predict(X_te)
    
    # Estimate number of outliers in test data
    outlier_num = int(X_te.shape[0]*np.mean(outliers))

    # Set that proportion of top estimated outliers to outlier value
    thresh = np.sort(pred_outlier)[-outlier_num]
    y_pred[pred_outlier>thresh] = -33.21928024
    
    # Return predictions
    return y_pred


# In[15]:


get_ipython().run_cell_magic('time', '', "\n# Performance of mixed classifier + regressor\ncross_val_metric(classifier_regressor, \n                 X_train, y_train,\n                 metric=root_mean_squared_error, \n                 cv=3, display='RMSE')")


# In[16]:


# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', BayesianRidge(alpha_1=1e-6,
                                alpha_2=1e-6,
                                lambda_1=1e-6,
                                lambda_2=1e-6))
])


# In[17]:


# Parameter bounds
bounds = {
    'regressor__alpha_1':  [1e-7, 1e-5, float],
    'regressor__alpha_2':  [1e-7, 1e-7, float],
    'regressor__lambda_1': [1e-7, 1e-7, float],
    'regressor__lambda_2': [1e-7, 1e-7, float],
}


# In[18]:


"""
%%time
# Find the optimal parameters
opt_params, gpo = optimize_params(
    X_train, y_train, xgb_pipeline, bounds,
    n_splits=2, max_evals=300, n_random=100,
    metric=root_mean_squared_error)
    
# Show the expected best parameters
print(opt_params)
"""

# Output:
#    {'regressor__alpha_1': 2.76e-6,
#     'regressor__alpha_2': 7.96e-7,
#     'regressor__lambda_1': 7.32e-7,
#     'regressor__lambda_2': 9.69e-6}
#    Wall time: 5h 35min 34s


# In[19]:


"""
%%time

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', XGBRegressor(max_depth=3,
                               learning_rate=0.1,
                               n_estimators=100))
])

# Parameter bounds
bounds = {
    'regressor__max_depth': [2, 3, int],
    'regressor__learning_rate': [0.01, 0.5, float],
    'regressor__n_estimators': [50, 200, int],
}

# Find the optimal parameters
opt_params, gpo = optimize_params(
    X_train, y_train, xgb_pipeline, bounds,
    n_splits=2, max_evals=100, n_random=20,
    metric=root_mean_squared_error)
    
# Show the expected best parameters
print(opt_params)
"""

# Output:
#    {'regressor__max_depth': 5, 
#     'regressor__learning_rate': 0.0739,
#     'regressor__n_estimators': 158}
#    Wall time: 5h 20min 3s


# In[20]:


"""
%%time

# LightGBM pipeline
lgbm_pipeline = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', LGBMRegressor(num_leaves=31,
                                max_depth=-1,
                                learning_rate=0.1,
                                n_estimators=100))
])

# Parameter bounds
bounds = {
    'regressor__num_leaves': [10, 50, int],
    'regressor__max_depth': [2, 8, int],
    'regressor__learning_rate': [0.01, 0.2, float],
    'regressor__n_estimators': [50, 200, int],
}

# Find the optimal parameters
opt_params, gpo = optimize_params(
    X_train, y_train, lgbm_pipeline, bounds,
    n_splits=2, max_evals=100, n_random=20,
    metric=root_mean_squared_error)
    
# Show the expected best parameters
print(opt_params)
"""

# Output
#    {'regressor__num_leaves': 30,
#     'regressor__max_depth': 6,
#     'regressor__learning_rate': 0.0439,
#     'regressor__n_estimators': 160}
#    Wall time: 5h 19min 45s


# In[21]:


"""
%%time

# CatBoost pipeline
cb_pipeline = Pipeline([
    ('targ_enc', TargetEncoderCV(cols=cat_cols)),
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor(depth=6,
                                    learning_rate=0.03,
                                    iterations=1000,
                                    l2_leaf_reg=3.0))
])

# Parameter bounds
bounds = {
    'regressor__depth': [4, 10, int],
    'regressor__learning_rate': [0.01, 0.1, float],
    'regressor__iterations': [500, 1500, int],
    'regressor__l2_leaf_reg': [1.0, 5.0, float],
}

# Find the optimal parameters
opt_params, gpo = optimize_params(
    X_train, y_train, cb_pipeline, bounds,
    n_splits=2, max_evals=100, n_random=20,
    metric=root_mean_squared_error)
        
# Show the expected best parameters
print(opt_params)
"""

# Output
#    {'regressor__depth': 9, 
#     'regressor__learning_rate': 0.0240, 
#     'regressor__iterations': 650, 
#     'regressor__l2_leaf_reg': 1.47}
#    Wall time: 5h 25min 22s


# In[22]:


# Bayesian ridge regression
ridge = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge(alpha_1=3e-6,
                                alpha_2=1e-6,
                                lambda_1=1e-6,
                                lambda_2=1e-5))
])

# XGBoost
xgboost = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', XGBRegressor(max_depth=5,
                               learning_rate=0.07,
                               n_estimators=150, 
                               n_jobs=2))
])

# LightGBM
lgbm = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='mean')),
    ('regressor', LGBMRegressor(num_leaves=30,
                                max_depth=6,
                                learning_rate=0.05,
                                n_estimators=160))
])

# CatBoost
catboost = Pipeline([
    ('targ_enc',  TargetEncoderCV(cols=cat_cols)),
    ('scaler',    RobustScaler()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor(verbose=False, 
                                    depth=7,
                                    learning_rate=0.02,
                                    iterations=1000,
                                    l2_leaf_reg=2.0))
])


# In[23]:


get_ipython().run_cell_magic('time', '', "\n# Bayesian Ridge Regression\nrmse_br, preds_br = cross_val_metric(\n    ridge, X_train, y_train,\n    metric=root_mean_squared_error,\n    cv=3, display='RMSE')")


# In[24]:


get_ipython().run_cell_magic('time', '', "\n# XGBoost\nrmse_xgb, preds_xgb = cross_val_metric(\n    xgboost, X_train, y_train,\n    metric=root_mean_squared_error,\n    cv=3, display='RMSE')")


# In[25]:


get_ipython().run_cell_magic('time', '', "\n# LightGBM\nrmse_lgb, preds_lgb = cross_val_metric(\n    lgbm, X_train, y_train,\n    metric=root_mean_squared_error,\n    cv=3, display='RMSE')")


# In[26]:


get_ipython().run_cell_magic('time', '', "\n# CatBoost\nrmse_cb, preds_cb = cross_val_metric(\n    catboost, X_train, y_train,\n    metric=root_mean_squared_error,\n    cv=3, display='RMSE')")


# In[27]:


# Construct a DataFrame with each model's predictions
pdf = pd.DataFrame()
pdf['Ridge'] = preds_br
pdf['XGB'] = preds_xgb
pdf['LGBM'] = preds_lgb
pdf['CatBoost'] = preds_cb

# Plot the correlation matrix
corr = pdf.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")
sns.heatmap(corr, mask=mask, annot=True,
            square=True, linewidths=.5,
            cbar_kws={'label': 'Correlation coefficient'})


# In[28]:


# Compute the averaged predictions
mean_preds = pdf.mean(axis=1)

# Compute RMSE of averaged predictions
root_mean_squared_error(y_train, mean_preds)


# In[29]:


# Compute the averaged predictions
mean_preds = pdf[['XGB', 'LGBM', 'CatBoost']].mean(axis=1)

# Compute RMSE of averaged predictions
root_mean_squared_error(y_train, mean_preds)


# In[30]:


get_ipython().run_cell_magic('time', '', "\n# Create the ensemble regressor\nmodel = StackedRegressor([ridge, xgboost, catboost, lgbm],\n                         meta_learner=BayesianRidge())\n\n# Performance of ensemble\ncross_val_metric(model, X_train, y_train,\n                 metric=root_mean_squared_error,\n                 cv=3, display='RMSE')")


# In[31]:


get_ipython().run_cell_magic('time', '', "\n# Create the ensemble regressor\nmodel = StackedRegressor([xgboost, catboost, lgbm],\n                         meta_learner=BayesianRidge())\n\n# Performance of ensemble\ncross_val_metric(model, X_train, y_train,\n                 metric=root_mean_squared_error,\n                 cv=3, display='RMSE')")


# In[32]:


# Fit model on training data
fit_model = model.fit(X_train, y_train)

# Predict on test data
predictions = fit_model.predict(X_test)

# Write predictions to file
df_out = pd.DataFrame()
df_out['card_id'] = X_test.index
df_out['target'] = predictions
df_out.to_csv('predictions.csv', index=False)

