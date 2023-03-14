#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set()

get_ipython().system('pip install git+http://github.com/brendanhasz/dsutils.git')
from dsutils.encoding import one_hot_encode
from dsutils.encoding import TargetEncoderCV
from dsutils.printing import print_table
from dsutils.evaluation import permutation_importance_cv
from dsutils.evaluation import plot_permutation_importance




# Load data containing all the features
fname = '../input/elo-feature-engineering-and-feature-selection/card_features_all.feather'
cards = pd.read_feather(fname)
cards.set_index('card_id', inplace=True)

# Test data
test = cards['target'].isnull()
X_test = cards[test].copy()
del X_test['target']

# Training data
y_train = cards.loc[~test, 'target'].copy()
X_train = cards[~test].copy()
del X_train['target']

# Clean up
del cards
gc.collect()




def mutual_information(xi, yi, res=20):
    """Compute the mutual information between two vectors"""
    ix = ~(np.isnan(xi) | np.isinf(xi) | np.isnan(yi) | np.isinf(yi))
    x = xi[ix]
    y = yi[ix]
    N, xe, ye = np.histogram2d(x, y, res)
    Nx, _ = np.histogram(x, xe)
    Ny, _ = np.histogram(y, ye)
    N = N / len(x) #normalize
    Nx = Nx / len(x)
    Ny = Ny / len(y)
    Ni = np.outer(Nx, Ny)
    Ni[Ni == 0] = np.nan
    N[N == 0] = np.nan
    return np.nansum(N * np.log(N / Ni))




# Show mutual information vs correlation
x = 5*np.random.randn(1000)
y = [x + np.random.randn(1000),
     2*np.sin(x) + np.random.randn(1000),
     x + 10*np.random.randn(1000)]
plt.figure(figsize=(10, 4))
for i in range(3):    
    plt.subplot(1, 3, i+1)
    plt.plot(x, y[i], '.')
    rho, _ = spearmanr(x, y[i])
    plt.title('Mutual info: %0.3f\nCorr coeff: %0.3f'
              % (mutual_information(x, y[i]), rho))
    plt.gca().tick_params(labelbottom=False, labelleft=False)




def quantile_transform(v, res=101):
    """Quantile-transform a vector to lie between 0 and 1"""
    x = np.linspace(0, 100, res)
    prcs = np.nanpercentile(v, x)
    return np.interp(v, prcs, x/100.0)
    
    
def q_mut_info(x, y):
    """Mutual information between quantile-transformed vectors"""
    return mutual_information(quantile_transform(x),
                              quantile_transform(y))




get_ipython().run_cell_magic('time', '', "\n# Compute the mutual information\ncols = []\nmis = []\nfor col in X_train:\n    mi = q_mut_info(X_train[col], y_train)\n    cols.append(col)\n    mis.append(mi)\n    \n# Print mut info of each feature\nprint_table(['Column', 'Mutual_Information'],\n            [cols, mis])")




# Create DataFrame with scores
mi_df = pd.DataFrame()
mi_df['Column'] = cols
mi_df['mut_info'] = mis

# Sort by mutual information
mi_df = mi_df.sort_values('mut_info', ascending=False)
top200 = mi_df.iloc[:200,:]
top200 = top200['Column'].tolist()

# Keep only top 200 columns
X_train = X_train[top200]
X_test = X_test[top200]




# Regression pipeline
cat_cols = [c for c in X_train if 'mode' in c] 
reg_pipeline = Pipeline([
    ('target_encoder', TargetEncoderCV(cols=cat_cols)),
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor(loss_function='RMSE', 
                                    verbose=False))
])




get_ipython().run_cell_magic('time', '', "\n# Compute the cross-validated feature importance\nimp_df = permutation_importance_cv(\n    X_train, y_train, reg_pipeline, 'rmse', n_splits=2)")




# Plot the feature importances
plt.figure(figsize=(8, 100))
plot_permutation_importance(imp_df)
plt.show()




# Get top 100 most important features
df = pd.melt(imp_df, var_name='Feature', value_name='Importance')
dfg = (df.groupby(['Feature'])['Importance']
       .aggregate(np.mean)
       .reset_index()
       .sort_values('Importance', ascending=False))
top100 = dfg['Feature'][:100].tolist()




# Save file w/ 100 most important features
cards = pd.concat([X_train[top100], X_test[top100]])
cards['target'] = y_train
cards.reset_index(inplace=True)
cards.to_feather('card_features_top100.feather')

